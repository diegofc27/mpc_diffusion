from argparse import ArgumentParser
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
from diffuser.environments.point import Find_Dot
import pandas as pd
from collections import defaultdict

def evaluate(**deps):
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

    RUN._update(deps)
    Config._update(deps)
    ##get args from terminal
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--conditioning", type=str, default="right")
    parser.add_argument("--condition_guidance_w", type=float, nargs='+', default=3.0)
    parser.add_argument("--device", type=str, default='cuda:2')
    args = parser.parse_args()


    logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))

    Config.device = args.device
    Config.condition_guidance_w = args.condition_guidance_w
    print(args.condition_guidance_w)

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
        max_n_episodes=Config.max_n_episodes,
        point_dataset=Config.point_dataset,
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

    env = Find_Dot(max_number_steps=30)
    env_list = [env]
    dones = [0 for _ in range(num_eval)]
    episode_rewards = [0 for _ in range(num_eval)]
    list_obs = []

    assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w

    if args.conditioning =="composition":
        skills = to_device(torch.tensor([[1.0,0.0],[0.0,1.0]],dtype=torch.float32).repeat(num_eval,1), device)
    elif args.conditioning =="right":
        skills = to_device(torch.tensor([[1.0,0.0]],dtype=torch.float32).repeat(num_eval,1), device)
    elif args.conditioning =="up":
        skills = to_device(torch.tensor([[0.0,1.0]],dtype=torch.float32).repeat(num_eval,1), device)
    # #change dtype os skills tensor to float32
    # skills = skills.type(torch.float32)
    t = 0
    #makes 2 list from 0 to 10 with step .1
    list_action = np.arange(1.2, 10, 0.1).tolist()
    list_action2 = np.arange(1.2, 10, 0.1).tolist()
    dif = defaultdict(list)
    count = 0
    #loop over the combinations of actions
    for i in list_action:
        for j in list_action2:
            np.random.seed(args.seed)
            trainer.ema_model.condition_guidance_w =[i,j]
            obs_list = [env.reset() for env in env_list]
            
            obs = np.concatenate(obs_list, axis=0)
            list_obs.append(obs.tolist())
            rewards_list = []
            list_action = []
            done = False
            while not done:
                obs = dataset.normalizer.normalize(obs, 'observations')
                conditions = {0: to_torch(obs, device=device)}
                samples = trainer.ema_model.conditional_sample(conditions, skills=skills)
                obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
                obs_comb = obs_comb.reshape(-1, 2*observation_dim)
                action = trainer.ema_model.inv_model(obs_comb)

                samples = to_np(samples)
                action = to_np(action)
                #action = dataset.normalizer.unnormalize(action, 'actions')

                if t == 0:
                    normed_observations = samples[:, :, :]
                samples_unnorm = dataset.normalizer.unnormalize(samples, 'observations')
                diff_x = samples_unnorm[0][:,0][7]-samples_unnorm[0][:,0][0]
                diff_y = samples_unnorm[0][:,1][7]-samples_unnorm[0][:,1][0]
                dif["rigth_w"].append(i)
                dif["up_w"].append(j)
                dif["diff_x"].append(diff_x)
                dif["diff_y"].append(diff_y)
                #print average of samples
                done = True
                count += 1
                print(f"count: {count}/10000")
                print(f" w: [{i},{j}] diff_x: {diff_x} diff_y: {diff_y}")
        df = pd.DataFrame(dif)
        df.to_csv("/home/fernandi/projects/decision-diffuser/code/skills/dif_straight.csv",index=False)
           
    #save the dictionary with the differences as csv
    df = pd.DataFrame(dif)
    df.to_csv("/home/fernandi/projects/decision-diffuser/code/skills/dif_straight.csv")
   