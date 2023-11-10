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
from rl_zoo3 import create_test_env
import numpy as np
from pathlib import Path
import pickle

def eval_diffusion(diffusion, dataset, config):
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

    Config._update(config)

    # Load configs
    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)

    if Config.predict_epsilon:
        prefix = f'predict_epsilon_{Config.n_diffusion_steps}_1000000.0'
    else:
        prefix = f'predict_x0_{Config.n_diffusion_steps}_1000000.0'

    loadpath = os.path.join(Config.bucket, logger.prefix, 'checkpoint')
    if Config.save_checkpoints:
        loadpath = os.path.join(loadpath, f'state_0.pt')
    else:
        loadpath = os.path.join(loadpath, 'state.pt')
    print(loadpath)
    state_dict = torch.load(loadpath, map_location=Config.device)

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
    trainer = trainer_config(diffusion, dataset, None)
    logger.print(utils.report_parameters(model), color='green')
    trainer.step = state_dict['step']
    trainer.model.load_state_dict(state_dict['model'])
    trainer.ema_model.load_state_dict(state_dict['ema'])

    num_eval = 100
    device = Config.device

    hyperparams = OrderedDict([('batch_size', 2048), ('buffer_size', 1000000), ('env_wrapper', 'sb3_contrib.common.wrappers.TimeFeatureWrapper'), ('gamma', 0.95), ('learning_rate', 0.001), ('n_timesteps', 3000000.0), ('normalize', False), ('policy', 'MultiInputPolicy'), ('policy_kwargs', 'dict(net_arch=[512, 512, 512], n_critics=2)'), ('replay_buffer_class', 'HerReplayBuffer'), ('replay_buffer_kwargs', "dict(goal_selection_strategy='future', n_sampled_goal=4,)"), ('tau', 0.05)])
    env = create_test_env(
        "PandaPushDense-v3",
        n_envs=num_eval,
        stats_path='/home/fernandi/projects/decision-diffuser/code/scripts',
        seed=Config.seed,
        log_dir=None,
        should_render=False,
        hyperparams=hyperparams,
        env_kwargs={},
    )
    
    assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w
    skills = to_device(torch.tensor([[0,1]],dtype=torch.float32).repeat(num_eval,1), device)
    # #change dtype os skills tensor to float32
    # skills = skills.type(torch.float32)
    
    t = 0
    obs_list = env.reset() 
    obs_list_new = []
    success_ids = []
    for i in range(num_eval):
        dg = obs_list['desired_goal'][i].tolist()
        ag = obs_list['achieved_goal'][i].tolist()
        o = obs_list['observation'][i].tolist()
        extra = np.append(dg,ag)
        obs_list_new.append([np.append(o,extra)])
    obs = np.concatenate(obs_list_new, axis=0)
   
    rewards_list = []
    done = False
    images = defaultdict(list)

    target_path = Path(f"/home/fernandi/projects/decision-diffuser/code/skills/target_198740.pickle")
    with open(target_path, 'rb') as f:
        target = pickle.load(f)
    success_rate=0
    while not done:

        print("step: ",t)
        obs = dataset.normalizer.normalize(obs, 'observations')
        conditions = {0: to_torch(obs, device=device)}
        samples = trainer.ema_model.conditional_sample(conditions, skills=skills)
        observations_nonorm = dataset.normalizer.unnormalize(samples[:, :, :].cpu(), 'observations')
        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
        obs_comb = obs_comb.reshape(-1, 2*observation_dim)
        action = trainer.ema_model.inv_model(obs_comb)
        samples = to_np(samples)
        action = to_np(action)
        if t == 0:
            normed_observations = samples[:, :, :]
            observations = dataset.normalizer.unnormalize(normed_observations, 'observations')

        obs_list = []
   
        this_obs, this_reward, done, info = env.step(action)
        done = done.all()
        print("")

        obs_list_new = []
        for i in range(num_eval):
            if info[i]['is_success']:
                print("Sucess: #",i)              
                #if id is not in success_ids
                if i not in success_ids:          
                    success_rate+=1
                    print(f"Sucess rate:{success_rate}/{num_eval}")
                success_ids.append(i)
            dg = this_obs['desired_goal'][i].tolist()
            ag = this_obs['achieved_goal'][i].tolist()
            o = this_obs['observation'][i].tolist()
            extra = np.append(dg,ag)
            obs_list_new.append([np.append(o,extra)])
        obs = np.concatenate(obs_list_new, axis=0)
        
        if t > 51:
            done = True
        t += 1
    sucecss_porcentage = (success_rate/num_eval)
    env.close()    

    return sucecss_porcentage