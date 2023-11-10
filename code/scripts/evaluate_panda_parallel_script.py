import diffuser.utils as utils
from ml_logger import logger
import torch
from copy import deepcopy
import numpy as npco
import os
import gym
import gymnasium as gym
import panda_gym
from config.locomotion_config import Config
from diffuser.utils.arrays import to_torch, to_np, to_device
from diffuser.datasets.d4rl import suppress_output
from collections import OrderedDict, defaultdict
import numpy as np
from pathlib import Path
import pickle
from numpngw import write_apng
from IPython.display import Image

def evaluate(**deps):
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

    RUN._update(deps)
    Config._update(deps)

    logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))

    Config.device = 'cuda:4'

    if Config.predict_epsilon:
        prefix = f'predict_epsilon_{Config.n_diffusion_steps}_1000000.0'
    else:
        prefix = f'predict_x0_{Config.n_diffusion_steps}_1000000.0'

    loadpath = os.path.join(Config.bucket, logger.prefix, 'checkpoint')
    if Config.save_checkpoints:
        loadpath = os.path.join(loadpath, f'state_670000.pt')
    else:
        loadpath = os.path.join(loadpath, 'state.pt')
    print(loadpath)
    state_dict = torch.load(loadpath, map_location=Config.device)

    # Load configs
    torch.backends.cudnn.benchmark = True
    Config.seed = 2342123421
    utils.set_seed(Config.seed)

    dataset_config = utils.Config(
        Config.loader,
        savepath='dataset_config.pkl',
        env=Config.dataset,
        horizon=Config.horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        use_padding=Config.use_padding,
        max_path_length=Config.max_path_length,
        include_returns=Config.include_returns,
        returns_scale=Config.returns_scale,
        skill_dataset=Config.skill_dataset,
    
    )

    #render_config = utils.Config(
    #    Config.renderer,
    #    savepath='render_config.pkl',
    #    env=Config.dataset,
    #)

    dataset = dataset_config()
    #renderer = render_config()

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    if Config.diffusion == 'models.GaussianInvDynDiffusion' or 'GaussianInvDynDiffusionSkills':
        transition_dim = observation_dim
    else:
        transition_dim = observation_dim + action_dim

    model_config = utils.Config(
        Config.model,
        savepath='model_config.pkl',
        horizon=Config.horizon,
        transition_dim=transition_dim,
        cond_dim=observation_dim,
        dim_mults=Config.dim_mults,
        dim=Config.dim,
        returns_condition=Config.returns_condition,
        skills_condition=Config.skills_condition,
        device=Config.device,
        attention=Config.attention,
    )

    Config.condition_guidance_w =1.2
    diffusion_config = utils.Config(
        Config.diffusion,
        savepath='diffusion_config.pkl',
        horizon=Config.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=Config.n_diffusion_steps,
        loss_type=Config.loss_type,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        hidden_dim=Config.hidden_dim,
        ## loss weighting
        action_weight=Config.action_weight,
        loss_weights=Config.loss_weights,
        loss_discount=Config.loss_discount,
        returns_condition=Config.returns_condition,
        skills_condition=Config.skills_condition,
        device=Config.device,
        condition_guidance_w=Config.condition_guidance_w,
    )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath='trainer_config.pkl',
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        log_freq=Config.log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        save_parallel=Config.save_parallel,
        bucket=Config.bucket,
        n_reference=Config.n_reference,
        train_device=Config.device,
    )

    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, None)
    logger.print(utils.report_parameters(model), color='green')
    trainer.step = state_dict['step']
    trainer.model.load_state_dict(state_dict['model'])
    trainer.ema_model.load_state_dict(state_dict['ema'])

  
    num_eval = 100
    device = Config.device

    env_list = [gym.make(Config.dataset) for _ in range(num_eval)]
    dones = [0 for _ in range(num_eval)]
    episode_rewards = [0 for _ in range(num_eval)]

    assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w
    skills = to_device(torch.tensor([[0,1]],dtype=torch.float32).repeat(num_eval,1), device)
    t = 0
    obs_list = [env.reset()[0] for env in env_list]
    obs_list_new = []
    for i in range(num_eval):
        dg = obs_list[i]['desired_goal'].tolist()
        ag = obs_list[i]['achieved_goal'].tolist()
        o = obs_list[i]['observation'].tolist()
        extra = np.append(dg,ag)
        obs_list_new.append([np.append(o,extra)])
    obs = np.concatenate(obs_list_new, axis=0)
    success_rate = 0
    images_3 = [env_list[3].render()]
    images_2 = [env_list[2].render()]
    while sum(dones) <  num_eval:
        obs = dataset.normalizer.normalize(obs, 'observations')
        conditions = {0: to_torch(obs, device=device)}
        samples = trainer.ema_model.conditional_sample(conditions, skills=skills)
        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
        obs_comb = obs_comb.reshape(-1, 2*observation_dim)
        action = trainer.ema_model.inv_model(obs_comb)

        samples = to_np(samples)
        action = to_np(action)

        action = dataset.normalizer.unnormalize(action, 'actions')

        obs_list = []
        for i in range(num_eval):
            this_obs, this_reward, this_terminated,this_truncated, info = env_list[i].step(action[i])
            images_3.append(env_list[3].render())
            images_2.append(env_list[2].render())
            dg = this_obs['desired_goal'].tolist()
            ag = this_obs['achieved_goal'].tolist()
            o = this_obs['observation'].tolist()
            extra = np.append(dg,ag)
            obs_list.append([np.append(o,extra)])

            if this_terminated or this_truncated:
                if dones[i] == 1:
                    pass
                else:
                    if info['is_success']:
                        success_rate += 1
                    dones[i] = 1
                    episode_rewards[i] += this_reward
                    logger.print(f"Episode ({i}): {episode_rewards[i]}", color='green')
            else:
                if dones[i] == 1:
                    pass
                else:
                    episode_rewards[i] += this_reward

        obs = np.concatenate(obs_list, axis=0)
        t += 1
      
    write_apng(f"/home/fernandi/projects/decision-diffuser/code/images/sucess_push_0.png", images_3, delay=40)  
    Image(filename=f"/home/fernandi/projects/rl-baselines3-zoo/rl_zoo3/images/sucess_push_0.png")
    write_apng(f"/home/fernandi/projects/decision-diffuser/code/images/fail_push_0.png", images_3, delay=40)  
    Image(filename=f"/home/fernandi/projects/rl-baselines3-zoo/rl_zoo3/images/fail_push_0.png")
    print(f"Success rate: {success_rate/num_eval}")

    
