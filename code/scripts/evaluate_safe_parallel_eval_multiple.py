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

def evaluate(**deps):
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

    RUN._update(deps)
    Config._update(deps)

    logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))

    Config.device = 'cuda:1'

    # Load configs
    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)
    reward_cond_value = .1
    print(f"reward_cond_value: {reward_cond_value}")
    if Config.predict_epsilon:
        prefix = f'predict_epsilon_{Config.n_diffusion_steps}_1000000.0'
    else:
        prefix = f'predict_x0_{Config.n_diffusion_steps}_1000000.0'

    loadpath = os.path.join(Config.bucket, logger.prefix, 'checkpoint')
    if Config.save_checkpoints:
        loadpath = os.path.join(loadpath, f'state_0.pt')
    else:
        loadpath = os.path.join(loadpath, 'state_best.pt')
    print(loadpath)
    state_dict = torch.load(loadpath, map_location=Config.device)
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
        include_costs=Config.include_costs,
        returns_scale=Config.returns_scale,
        discount=Config.discount,
        termination_penalty=Config.termination_penalty,
        max_n_episodes=Config.max_n_episodes,
        skill_dataset=Config.skill_dataset,
        point_dataset=Config.point_dataset,
    )

    dataset = dataset_config()

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    if Config.diffusion == 'models.GaussianInvDynDiffusion' or 'GaussianInvDynDiffusionSkills':
        transition_dim = observation_dim
    else:
        transition_dim = observation_dim + action_dim
    if Config.diffusion == 'models.GaussianDiffusionSkills' or  'models.GaussianDiffusionReturns' or 'models.GaussianDiffusionCosts':
        model_config = utils.Config(
            Config.model,
            savepath='model_config.pkl',
            horizon=Config.horizon,
            transition_dim=observation_dim+action_dim,
            cond_dim=observation_dim,
            dim_mults=Config.dim_mults,
            returns_condition=Config.returns_condition,
            skills_condition=Config.skills_condition,
            dim=Config.dim,
            condition_dropout=Config.condition_dropout,
            calc_energy=Config.calc_energy,
            device=Config.device,
            attention=Config.attention,
            include_xy=Config.include_xy,
        )

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
            ar_inv=Config.ar_inv,
            train_only_inv=Config.train_only_inv,
            ## loss weighting
            action_weight=Config.action_weight,
            loss_weights=Config.loss_weights,
            loss_discount=Config.loss_discount,
            returns_condition=Config.returns_condition,
            skills_condition=Config.skills_condition,
            condition_guidance_w=Config.condition_guidance_w,
            goal_condition=Config.goal_condition,
            device=Config.device,
        )
    else:
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
    num_eval = 11
    device = Config.device
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
    dones = [0 for _ in range(num_eval)]
    episode_rewards = np.array([0 for _ in range(num_eval)])
    episode_cost = np.array([0 for _ in range(num_eval)])
    # episode_rewards = np.array([])
    # episode_cost = np.array([])
    frames = []
    seed = 69
    obs = envs.reset(seed)[0]
    # envs._env.render_parameters.mode = "rgb_array"
    # frames.append(envs.render())
    
    
    assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w
    if Config.diffusion == 'models.GaussianDiffusionReturns':
        returns = to_device(reward_cond_value* torch.ones(num_eval, 1), device)
    elif Config.diffusion == 'models.GaussianDiffusionSkills':
        skills = to_device(torch.tensor([[1.0,0.0],[0.0,1.0]],dtype=torch.float32).repeat(num_eval,1,1), device)
    elif Config.diffusion == 'models.GaussianDiffusionCosts':
        costs = to_device(0.1 * torch.ones(num_eval, 1), device)
    
    #skills = to_device(torch.tensor([[1.0,0.0]],dtype=torch.float32).repeat(num_eval,1,1), device)

    t=0
    horizon = 1000
    while t <  horizon:
        obs = dataset.normalizer.normalize(obs, 'observations')
        conditions = {0: to_torch(obs, device=device)}
        if Config.diffusion == 'models.GaussianDiffusionReturns':
            samples = trainer.ema_model.conditional_sample(conditions, returns=returns)
        elif Config.diffusion == 'models.GaussianDiffusionSkills':
            samples = trainer.ema_model.conditional_sample(conditions, skills=skills)
        elif Config.diffusion == 'models.GaussianDiffusionCosts':
            samples = trainer.ema_model.conditional_sample(conditions, costs=costs)
        if Config.diffusion == 'models.GaussianDiffusionSkills' or Config.diffusion == 'models.GaussianDiffusionReturns':
            action = samples[:, 0, :2]
        else:
            obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
            obs_comb = obs_comb.reshape(-1, 2*observation_dim)
            action = trainer.ema_model.inv_model(obs_comb)
            if num_eval == 1:
                action = action[0]
        samples = to_np(samples)
        obs,reward,cost,terminated,truncated,info=envs.step(action)
        #frames.append(envs.render())
        episode_rewards = np.vstack([episode_rewards,reward.numpy()])
        episode_cost = np.vstack([episode_cost,cost.numpy()])
        # episode_rewards = np.append(episode_rewards,reward.numpy())
        # episode_cost = np.append(episode_cost,cost.numpy())
        t += 1
        if t % 25 == 0:
            print(f"timestep: {t}, rewards: {episode_rewards.sum(axis=0)}, reward: {episode_rewards.sum(axis=0).mean()}, cost: {episode_cost.sum(axis=0).mean()}")
            total_rewards =episode_rewards.sum(axis=0).mean()
            total_cost = episode_cost.sum(axis=0).mean()
            print(f"Avg reward: {total_rewards}")
            print(f"Avg cost: {total_cost}")
            #results as json
            import pandas as pd
            reward_list = pd.Series(episode_rewards.sum(axis=0)).to_json(orient='values')
            cost_list = pd.Series(episode_cost.sum(axis=0)).to_json(orient='values')
            save_results_path = os.path.join(Config.bucket, logger.prefix)
            results = {'avg_rewards':total_rewards,'avg_cost':total_cost, 'std_rewards':episode_rewards.sum(axis=0).std(),
                    'reward_list':reward_list,'std_cost':episode_cost.sum(axis=0).std(), 'cost_list':cost_list, "timestep":t}
            if Config.diffusion == 'models.GaussianDiffusionReturns':
                name = f"seeds_10_combined_results_reward_{reward_cond_value}_.json"
            else:
                name = f"seeds_10_combined_results_reward.json"
            with open(os.path.join(save_results_path, name), 'w') as f:
                json.dump(results, f)

    # #save gif
    # save_replay_path = os.path.join(Config.bucket, logger.prefix, 'replay')
    # save_video(
    #     frames,
    #     save_replay_path,
    #     fps=30,
    #     episode_trigger=lambda x: True,
    #     video_length=horizon,
    #     episode_index=t,
    #     name_prefix='eval',
    # )    
    