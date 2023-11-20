import diffuser.utils as utils
from ml_logger import logger
import torch
from copy import deepcopy
import numpy as npco
import os
import gym
import datetime
from config.locomotion_config import Config
from diffuser.utils.arrays import to_torch, to_np, to_device
from diffuser.datasets.d4rl import suppress_output
from collections import OrderedDict, defaultdict
import numpy as np
from pathlib import Path
import pickle
from diffuser.environments.safe_grid import Safe_Grid
import json

def evaluate(diffusion, dataset, config):
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

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
    success_rate = 0
    while sum(dones) <  num_eval:
        obs = dataset.normalizer.normalize(obs, 'observations')
        if num_eval == 1:
            obs = obs[None]
        conditions = {0: to_torch(obs, device=device)}

        samples = trainer.ema_model.conditional_sample(conditions, returns=returns)
        #unormed_samples = dataset.normalizer.unnormalize(to_np(samples), 'observations')
        if Config.diffusion == 'models.GaussianDiffusionSkills' or Config.diffusion == 'models.GaussianDiffusionReturns' or Config.diffusion == 'models.GaussianDiffusionReturnsWithDynamics':
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
                    if info['goal_reached']:
                        success_rate += 1
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
    now = datetime.datetime.now()
    success_rate = success_rate/num_eval
    results = {'rewards':avg_rewards, 'cost':avg_cost, "success_rate":success_rate, 'time':now.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(save_results_path, 'results.json'), 'w') as f:
        json.dump(results, f)

    return avg_rewards, avg_cost, success_rate
