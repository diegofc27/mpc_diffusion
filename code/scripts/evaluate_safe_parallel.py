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
from omnisafe.envs.safety_gymnasium_env import SafetyGymnasiumEnv
from gymnasium.utils.save_video import save_video
import json


def eval_diffusion(diffusion, dataset, config,eval_steps=150,idx=0):
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
    device = Config.device
    num_eval = 1
    env_kwargs = {
        'env_id': Config.dataset,
        'num_envs': num_eval,
        'render_mode': "rgb_array",
        'camera_id': None,
        'camera_name': "track",
        'width': 500,
        'height': 500,
    }
    envs =SafetyGymnasiumEnv(**env_kwargs)
    # episode_rewards = np.array([0 for _ in range(num_eval)])
    # episode_cost = np.array([0 for _ in range(num_eval)])
    episode_rewards = np.array([])
    episode_cost = np.array([])
    frames = []
    seed = 69
    obs = envs.reset(seed)[0]
    envs._env.render_parameters.mode = "rgb_array"
    frames.append(envs.render())

    assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w
    if Config.diffusion == 'models.GaussianDiffusionReturns':
        returns = to_device(0.9 * torch.ones(num_eval, 1), device)
    elif Config.diffusion == 'models.GaussianDiffusionSkills':
        if Config.include_returns and Config.include_costs:
            skills = to_device(torch.tensor([[0.9,0.1]],dtype=torch.float32).repeat(num_eval,1,1), device)
        else:
            skills = to_device(torch.tensor([[1.0,0.0],[0.0,1.0]],dtype=torch.float32).repeat(num_eval,1,1), device)
    elif Config.diffusion == 'models.GaussianDiffusionCosts':
        costs = to_device(0.1 * torch.ones(num_eval, 1), device)

    t=0
    horizon = eval_steps
    while t <  horizon:
        obs = dataset.normalizer.normalize(obs, 'observations')
        if num_eval == 1:
            obs = obs[None]
        conditions = {0: to_torch(obs, device=device)}
        if Config.diffusion == 'models.GaussianDiffusionReturns':
            samples = trainer.ema_model.conditional_sample(conditions, returns=returns)
        elif Config.diffusion == 'models.GaussianDiffusionSkills':
            samples = trainer.ema_model.conditional_sample(conditions, skills=skills)
        elif Config.diffusion == 'models.GaussianDiffusionCosts':
            samples = trainer.ema_model.conditional_sample(conditions, costs=costs)
        if Config.diffusion == 'models.GaussianDiffusionSkills' or 'models.GaussianDiffusionReturns' or 'models.GaussianDiffusionCosts':
            action = samples[:, :, :2][0,0]
        else:
            obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
            obs_comb = obs_comb.reshape(-1, 2*observation_dim)
            action = trainer.ema_model.inv_model(obs_comb)
            if num_eval == 1:
                action = action[0]
        samples = to_np(samples)

        obs,reward,cost,terminated,truncated,info=envs.step(action)
        frames.append(envs.render())
        # episode_rewards = np.vstack([episode_rewards,reward.numpy()[0]])
        # episode_cost = np.vstack([episode_cost,cost.numpy()[0]])
        episode_rewards = np.append(episode_rewards,reward.numpy())
        episode_cost = np.append(episode_cost,cost.numpy())
        t += 1
        if t % 25 == 0:
            print(f"timestep: {t}, rewards: {episode_rewards.sum(axis=0)}, reward: {episode_rewards.sum(axis=0).mean()}, cost: {episode_cost.sum(axis=0).mean()}")
    total_rewards =episode_rewards.sum(axis=0).mean()
    total_cost = episode_cost.sum(axis=0).mean()
    print(f"Avg reward: {total_rewards}")
    print(f"Avg cost: {total_cost}")
    #import json
    save_results_path = os.path.join(Config.bucket, logger.prefix)
    results = {'rewards':total_rewards, 'cost':total_cost}
    save_results_path = os.path.join(save_results_path, f"results_{idx}.json")
    with open(save_results_path, 'w') as f:
        json.dump(results, f)


    #save gif
    save_replay_path = os.path.join(Config.bucket, logger.prefix, 'replay')
    name = f"eval_{idx}"
    save_video(
        frames,
        save_replay_path,
        fps=30,
        episode_trigger=lambda x: True,
        video_length=horizon,
        episode_index=t,
        name_prefix=name,
    )    
    
    return total_rewards, total_cost
