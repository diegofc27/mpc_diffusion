from pathlib import Path
import pickle
import diffuser.utils as utils
from ml_logger import logger
import torch
from copy import deepcopy
import numpy as np
import os
import gym
import gymnasium as gym
import panda_gym
from config.locomotion_config import Config
from diffuser.utils.arrays import to_torch, to_np, to_device
from diffuser.datasets.d4rl import suppress_output
from collections import OrderedDict
from rl_zoo3 import create_test_env
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
    Config.seed = 980
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
    )

    Config.condition_guidance_w =[1.2]
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

    num_eval = 1
    device = Config.device

    hyperparams = OrderedDict([('batch_size', 2048), ('buffer_size', 1000000), ('env_wrapper', 'sb3_contrib.common.wrappers.TimeFeatureWrapper'), ('gamma', 0.95), ('learning_rate', 0.001), ('n_timesteps', 3000000.0), ('normalize', False), ('policy', 'MultiInputPolicy'), ('policy_kwargs', 'dict(net_arch=[512, 512, 512], n_critics=2)'), ('replay_buffer_class', 'HerReplayBuffer'), ('replay_buffer_kwargs', "dict(goal_selection_strategy='future', n_sampled_goal=4,)"), ('tau', 0.05)])
    env = create_test_env(
        "PandaPushDense-v3",
        n_envs=10,
        stats_path='/home/fernandi/projects/decision-diffuser/code/scripts',
        seed=Config.seed,
        log_dir=None,
        should_render=False,
        hyperparams=hyperparams,
        env_kwargs={},
    )
     
    env_list = [env]
    dones = [0 for _ in range(num_eval)]
    episode_rewards = [0 for _ in range(num_eval)]

    assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w
    skills = to_device(torch.tensor([[0,1]],dtype=torch.float32).repeat(num_eval,1), device)
    # #change dtype os skills tensor to float32
    # skills = skills.type(torch.float32)
    
    t = 0
    obs_list = [env.reset() for env in env_list]
    
    extra = np.append(obs_list[0]['desired_goal'].tolist(),obs_list[0]['achieved_goal'].tolist())
    obs_list = [np.append(obs_list[0]['observation'].tolist(),extra)]
    obs = np.concatenate(obs_list, axis=0)
    
    recorded_obs = [deepcopy(obs[:, None])]
    rewards_list = []
    done = False
    images = [env.render()]
    target_path = Path(f"/home/fernandi/projects/decision-diffuser/code/skills/target_198740.pickle")
    with open(target_path, 'rb') as f:
        target = pickle.load(f)

    while not done:
        if t==0:
            print("obs: ",obs.tolist())
        obs = dataset.normalizer.normalize(obs, 'observations')
        conditions = {0: to_torch(obs, device=device)}
        samples = trainer.ema_model.conditional_sample(conditions, skills=skills)
        observations_nonorm = dataset.normalizer.unnormalize(samples[:, :, :].cpu(), 'observations')
        #print("observations_nonorm: ",observations_nonorm)
        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
        obs_comb = obs_comb.reshape(-1, 2*observation_dim)
        action = trainer.ema_model.inv_model(obs_comb)
        samples = to_np(samples)
        action = to_np(action)
        #action = dataset.normalizer.unnormalize(action, 'actions')

        if t == 0:
            normed_observations = samples[:, :, :]
            #print("normed_observations: ",normed_observations)
            observations = dataset.normalizer.unnormalize(normed_observations, 'observations')
            #savepath = os.path.join('images', 'sample-planned.png')
            #renderer.composite(savepath, observations)

        obs_list = []
                    
        i=0 
        env_list[i].render()
        #print("action: ",action[i])
        this_obs, this_reward, done, info = env_list[i].step([action[i]])
        images.append(env_list[i].render())
        print("")
        # print("action: ",action[i])
        # print("reward: ",this_reward)
        # print("done: ",done)
        # print("obs: ",this_obs['observation'].tolist()[0][:3])
        achieved_goal = this_obs['achieved_goal']
        desired_goal = this_obs['desired_goal']
        difference = np.linalg.norm(achieved_goal - desired_goal)
        print("difference: ",difference)
        extra = np.append(this_obs['desired_goal'].tolist(),this_obs['achieved_goal'].tolist())
        obs_list = [np.append(this_obs['observation'].tolist(),extra)]
        obs = np.concatenate(obs_list, axis=0)

        assert obs.shape[0] == 25
        if info[0]['is_success']:
            print("Sucess: ",info[0]['is_success'])
        rewards_list.append(this_reward)
        # obs = np.concatenate(obs_list, axis=0)
        # recorded_obs.append(deepcopy(obs[:, None]))
        
        t += 1

    recorded_obs = np.concatenate(recorded_obs, axis=1)
    #savepath = os.path.join('images', f'sample-executed.png')
    #renderer.composite(savepath, recorded_obs)
    episode_rewards = np.array(episode_rewards)
    write_apng(f"/home/fernandi/projects/decision-diffuser/code/images/diff_push_equal{Config.seed}.png", images, delay=40) 
    Image(filename=f"/home/fernandi/projects/decision-diffuser/code/images/diff_push_equal{Config.seed}.png")
    logger.print(f"average_ep_reward: {np.mean(episode_rewards)}, std_ep_reward: {np.std(episode_rewards)}", color='green')
    logger.log_metrics_summary({'average_ep_reward':np.mean(episode_rewards), 'std_ep_reward':np.std(episode_rewards)})
