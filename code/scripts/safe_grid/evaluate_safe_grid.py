import diffuser.utils as utils
from ml_logger import logger
import torch
from copy import deepcopy
import numpy as npco
import os
import gym

from config.locomotion_config import Config
from diffuser.utils.arrays import to_torch, to_np, to_device
from diffuser.datasets.d4rl import suppress_output
from collections import OrderedDict, defaultdict
import numpy as np
from pathlib import Path
import pickle
from diffuser.environments.safe_grid import Safe_Grid
import json
import datetime

def evaluate(**deps):
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

    RUN._update(deps)
    Config._update(deps)

    logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))

    Config.device = 'cuda:5'

    # Load configs
    torch.backends.cudnn.benchmark = True
    utils.set_seed(767)

    if Config.predict_epsilon:
        prefix = f'predict_epsilon_{Config.n_diffusion_steps}_1000000.0'
    else:
        prefix = f'predict_x0_{Config.n_diffusion_steps}_1000000.0'
    #get last part of path of logger.prefix
    name_model = logger.prefix.split('/')[-1]
    loadpath = os.path.join(Config.bucket, logger.prefix, 'checkpoint')
    if Config.save_checkpoints:
        loadpath = os.path.join(loadpath, f'state_0.pt')
    else:
        loadpath = os.path.join(loadpath, 'state.pt')
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
        returns_scale=Config.returns_scale,
        discount=Config.discount,
        termination_penalty=Config.termination_penalty,
        max_n_episodes=Config.max_n_episodes,
        skill_dataset=Config.skill_dataset,
        point_dataset=Config.point_dataset,
    )

    dataset = dataset_config()

    save_results_path = os.path.join(Config.bucket, logger.prefix)
    plots_path = os.path.join(save_results_path, 'plots')
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    transition_dim = observation_dim + action_dim

    if Config.diffusion == 'models.GaussianInvDynDiffusion' or Config.diffusion ==  'GaussianInvDynDiffusionSkills' or Config.diffusion == 'models.GaussianStaticInvDynDiffusion':
        transition_dim = observation_dim  


    print(f"transition_dim: {transition_dim}")
    if Config.diffusion == 'models.GaussianDiffusionSkills':
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
            #skills_condition=Config.skills_condition,
            #condition_guidance_w=Config.condition_guidance_w,
            #goal_condition=Config.goal_condition,
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
            #skills_condition=Config.skills_condition,
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
            #skills_condition=Config.skills_condition,
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
    device = Config.device
    num_eval =100
    envs = [Safe_Grid() for _ in range(num_eval)]
    episode_rewards = [0 for _ in range(num_eval)]
    episode_costs = [0 for _ in range(num_eval)]
    frames = []
    seed = 69
    obs = [envs[i].reset() for i in range(num_eval)]
    dones = [0 for _ in range(num_eval)]
    obs = np.array(obs)

    assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w
    #skills = to_device(torch.tensor([[1.0,0.0],[0.0,1.0]],dtype=torch.float32).repeat(10,1,1), device)
    returns = to_device(.9* torch.ones(num_eval, 1), device)

    t=0
    done = False
    while sum(dones) <  num_eval:
        obs = dataset.normalizer.normalize(obs, 'observations')
        if num_eval == 1:
            obs = obs[None]
        conditions = {0: to_torch(obs, device=device)}

        samples = trainer.ema_model.conditional_sample(conditions, returns=returns)
        #unormed_samples = dataset.normalizer.unnormalize(to_np(samples), 'observations')
        if Config.diffusion == 'models.GaussianDiffusionSkills' or Config.diffusion == 'models.GaussianDiffusionReturns':
            action = samples[:, 0, :2]
            action = dataset.normalizer.unnormalize(action.detach().cpu().numpy(), 'actions')
        elif Config.diffusion == 'models.GaussianStaticInvDynDiffusion':
            samples = dataset.normalizer.unnormalize(to_np(samples), 'observations')
            action = samples[:, 1, :2] - samples[:, 0, :2]
        else:
            obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
            obs_comb = obs_comb.reshape(-1, 2*observation_dim)
            action = trainer.ema_model.inv_model(obs_comb)
            action = action.detach().cpu().numpy()

            action = dataset.normalizer.unnormalize(action, 'actions')

            if num_eval == 1:
                action = action[0]

        #rollout each of the envs and save rewards and costs
        obs_list = []
        for i in range(num_eval):
            obs,reward,cost,done,info=envs[i].step(action[i])
            obs_list.append(obs[None])
            if done:
                if dones[i] == 1:
                    pass
                else:
                    dones[i] = 1
                    episode_rewards[i] += reward
                    episode_costs[i] += cost
                    envs[i].render(path=plots_path,name=f"Safe_Grid_episode_{i}")
            else:
                if dones[i] == 1:
                    pass
                else:
                    episode_rewards[i] += reward
                    episode_costs[i] += cost
        obs = np.concatenate(obs_list, axis=0)
        t += 1


        
    
    avg_rewards =np.array(episode_rewards).mean()
    avg_cost = np.array(episode_costs).mean()
    print(f"Avg reward: {avg_rewards}")
    print(f"Avg cost: {avg_cost}")
    #results as json
    results = {'rewards':avg_rewards, 'cost':avg_cost}
    with open(os.path.join(save_results_path, 'results.json'), 'w') as f:
        json.dump(results, f)

