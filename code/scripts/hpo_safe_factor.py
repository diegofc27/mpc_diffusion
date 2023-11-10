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
import optuna
import joblib
import pandas as pd


def hpo_w(**deps):
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

    RUN._update(deps)
    Config._update(deps)

    logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))

    Config.device = 'cuda:5'
    name_study = "w_hpo_full5"
    print("Name study: ", name_study)

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
    

   

    def objective(trial,name="w_hpo",device=torch.device('cuda:5'),num_eval=10,skills=None):
        #create env
        num_eval = num_eval
        env_kwargs = {
            'env_id': "SafetyPointButton3-v0",
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

        exp_dir = root / f"runs/{name}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(study, exp_dir / f"{name}.pkl")

        hp = defaultdict()
        hp["w_ppo"] = trial.suggest_float("w_ppo", 0,3)
        hp["w_ipo"] = trial.suggest_float("i_ppo", 0,3)
            
        skills = to_device(torch.tensor([[1.0,0.0],[0.0,1.0]],dtype=torch.float32).repeat(num_eval,1,1), device)
        trainer.ema_model.condition_guidance_w = [hp["w_ppo"],hp["w_ipo"]]
        t=0
        episode_len = 1000
        while t <  episode_len:
            obs = dataset.normalizer.normalize(obs, 'observations')
            conditions = {0: to_torch(obs, device=device)}
            samples = trainer.ema_model.conditional_sample(conditions, skills=skills)
            action = samples[:, 0, :2]
            samples = to_np(samples)
            obs,reward,cost,terminated,truncated,info=envs.step(action)
            #frames.append(envs.render())
            episode_rewards = np.vstack([episode_rewards,reward.numpy()])
            episode_cost = np.vstack([episode_cost,cost.numpy()])
            # episode_rewards = np.append(episode_rewards,reward.numpy())
            # episode_cost = np.append(episode_cost,cost.numpy())
            t += 1
            if t % 10 == 0:
                print(f"timestep: {t}, rewards: {episode_rewards.sum(axis=0)}, reward: {episode_rewards.sum(axis=0).mean()}, cost: {episode_cost.sum(axis=0).mean()}")
                total_rewards =episode_rewards.sum(axis=0).mean()
                total_cost = episode_cost.sum(axis=0).mean()
                print(f"Avg reward: {total_rewards}")
                print(f"Avg cost: {total_cost}")
               
        return total_rewards
    
    # HPO
    root =Path("/home/fernandi/projects/decision-diffuser/code/hpo")

    if os.path.isfile(root / f'{name_study}.pkl'):
        study = joblib.load(root / f'{name_study}.pkl')

        print("Loaded study object from the pickle file.")
    else:
        study = optuna.create_study()
    study = optuna.create_study(
        study_name=name_study,
        direction="maximize",
        # pruner=optuna.pruners.HyperbandPruner(
        #     min_resource=2, max_resource=epochs, reduction_factor=3
        # ),
    )
    study.optimize(objective, n_trials=300)
    exp_dir = root / f"runs/{name_study}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(study, exp_dir / f"{name_study}.pkl")

