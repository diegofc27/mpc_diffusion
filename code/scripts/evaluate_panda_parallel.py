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

    env_list = [gym.make(Config.dataset) for _ in range(num_eval)]
    dones = [0 for _ in range(num_eval)]
    episode_rewards = [0 for _ in range(num_eval)]
    obs_list = [env.reset()[0] for env in env_list]

    t = 0
    obs_list_new = []
    dg_list = []
    for i in range(num_eval):
        dg = obs_list[i]['desired_goal'].tolist()
        dg_list.append(dg)
        ag = obs_list[i]['achieved_goal'].tolist()
        o = obs_list[i]['observation'].tolist()
        extra = np.append(dg,ag)
        obs_list_new.append([np.append(o,extra)])
    obs = np.concatenate(obs_list_new, axis=0)
    success_rate = 0

    assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w
    if Config.skills_condition:
        skills = to_device(torch.tensor([[0,1]],dtype=torch.float32).repeat(num_eval,1), device)
    elif Config.goal_condition:
        goals = to_device(torch.tensor(np.array(dg_list),dtype=torch.float32), device)
    
    while sum(dones) <  num_eval:
        obs = dataset.normalizer.normalize(obs, 'observations')
        conditions = {0: to_torch(obs, device=device)}
        if Config.skills_condition:
            samples = trainer.ema_model.conditional_sample(conditions, skills=skills)
        elif Config.goal_condition:
            samples = trainer.ema_model.conditional_sample(conditions, skills=goals)
        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
        obs_comb = obs_comb.reshape(-1, 2*observation_dim)
        action = trainer.ema_model.inv_model(obs_comb)

        samples = to_np(samples)
        action = to_np(action)

        action = dataset.normalizer.unnormalize(action, 'actions')

        obs_list = []
        #dg_list = []
        for i in range(num_eval):
            this_obs, this_reward, this_terminated,this_truncated, info = env_list[i].step(action[i])
            dg = this_obs['desired_goal'].tolist()
            #dg_list.append(dg)
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
        #goals = to_device(torch.tensor(np.array(dg_list),dtype=torch.float32), device)

        obs = np.concatenate(obs_list, axis=0)
        t += 1
    print(f"Success rate: {success_rate/num_eval}")

    
    return success_rate/num_eval, np.mean(episode_rewards).item()
