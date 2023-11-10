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

def eval_diffusion(diffusion, dataset, config,eval_steps=150):
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
    num_eval = 10
    device = Config.device

    envs =SafetyGymnasiumEnv(Config.dataset,num_eval)
    dones = [0 for _ in range(num_eval)]
    episode_rewards = np.array([0 for _ in range(num_eval)])
    episode_cost = np.array([0 for _ in range(num_eval)])
    obs = envs.reset()[0]


    assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w
    skills = to_device(torch.tensor([[1.0,0.0],[0.0,1.0]],dtype=torch.float32).repeat(10,1,1), device)
    #skills = to_device(torch.tensor([[1.0,0.0]],dtype=torch.float32).repeat(num_eval,1,1), device)


    t=0
    while t <  eval_steps:
        obs = dataset.normalizer.normalize(obs, 'observations')
        conditions = {0: to_torch(obs, device=device)}
        import pdb; pdb.set_trace()
        samples = trainer.ema_model.conditional_sample(conditions, skills=skills)

        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
        obs_comb = obs_comb.reshape(-1, 2*observation_dim)
        action = trainer.ema_model.inv_model(obs_comb)

        samples = to_np(samples)

        obs,reward,cost,terminated,truncated,info=envs.step(action)
        episode_rewards = np.vstack([episode_rewards,reward.numpy()])
        episode_cost = np.vstack([episode_cost,cost.numpy()])
        t += 1
    total_rewards =episode_rewards.sum(axis=0).mean()
    total_cost = episode_cost.sum(axis=0).mean()
    
    print(f"total_rewards: {episode_rewards}")
    print(f"Avg reward: {total_rewards}")
    print(f"Avg cost: {total_cost}")
    
    return total_rewards, total_cost
