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
#from rl_zoo3 import create_test_env
from diffuser.environments.point import Find_Dot
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
def evaluate(**deps):
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

    RUN._update(deps)
    Config._update(deps)
    ##get args from terminal
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--conditioning", type=str, default="right")
    parser.add_argument("--cond_w", type=float, nargs='+', default=3.0)
    parser.add_argument("--device", type=str, default='cuda:1')
    args = parser.parse_args()


    logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))

    Config.device = args.device
    Config.condition_guidance_w = args.cond_w
    print(args.cond_w)

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

    env = Find_Dot(max_number_steps=50)
    env_list = [env]
    dones = [0 for _ in range(num_eval)]
    episode_rewards = [0 for _ in range(num_eval)]
    list_obs = []

    assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w

    if args.conditioning =="composition" or args.conditioning =="alternating":
        skills = to_device(torch.tensor([[1.0,0.0],[0.0,1.0]],dtype=torch.float32).repeat(num_eval,1), device)
    elif args.conditioning =="right":
        skills = to_device(torch.tensor([[1.0,0.0]],dtype=torch.float32).repeat(num_eval,1), device)
    elif args.conditioning =="up":
        skills = to_device(torch.tensor([[0.0,1.0]],dtype=torch.float32).repeat(num_eval,1), device)

    # #change dtype os skills tensor to float32
    # skills = skills.type(torch.float32)
    t = 0
    np.random.seed(args.seed)
    obs_list = [env.reset() for env in env_list]
    
    obs = np.concatenate(obs_list, axis=0)
    list_obs.append(obs.tolist())
    rewards_list = []
    list_action = []
    done = False
    sample_all = True
    while not done:
        with torch.no_grad():
            obs = dataset.normalizer.normalize(obs, 'observations')
            conditions = {0: to_torch(obs, device=device).unsqueeze(0)}
            if args.conditioning =="alternating":
                if t % 2 == 0:
                    skills = (torch.tensor([[1.0,0.0,obs[0][0],obs[0][1]]],dtype=torch.float32,device=device))
                    samples = trainer.ema_model.conditional_sample(conditions, skills=skills)
                    print("right")            
                else:
                    skills = (torch.tensor([[0.0,1.0,obs[0][0],obs[0][1]]],dtype=torch.float32,device=device))
                    samples = trainer.ema_model.conditional_sample(conditions, skills=skills)
                    print("up")
                    
            elif args.conditioning =="composition":
                skills = (torch.tensor([[1.0,0.0,obs[0],obs[1]],[0.0,1.0,obs[0],obs[1]]],dtype=torch.float32,device=device))
                samples = trainer.ema_model.conditional_sample(conditions, skills=skills)
            else:
                skills = (torch.tensor([[skills[0][0].item(),skills[0][1].item(),obs[0],obs[1]]],dtype=torch.float32,device=device))
                samples = trainer.ema_model.conditional_sample(conditions, skills=skills)
            
            # obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
            # obs_comb = obs_comb.reshape(-1, 2*observation_dim)
            #action = trainer.ema_model.inv_model(obs_comb)

            samples = to_np(samples)
            
            if(sample_all):
                num_samples = len(samples[0])
                frames = []
                for i in range(num_samples):
                    # samples_uncond = dataset.normalizer.unnormalize(samples[0][0].cpu(), 'observations')
                    # samples_cond1 = dataset.normalizer.unnormalize(samples[1][0].cpu(), 'observations')
                    # samples_cond2 = dataset.normalizer.unnormalize(samples[2][0].cpu(), 'observations')
                    # samples_all= dataset.normalizer.unnormalize(samples[3][0].cpu(), 'observations')
                    # print(samples_all[0][:10][:,0])
                    # plt.plot(samples_all[0][:][:,0],samples_all[0][:][:,1],label="all")
                    # plt.plot(samples_uncond[0][:][:,0],samples_uncond[0][:][:,1],label="uncond")
                    # plt.plot(samples_cond1[0][:][:,0],samples_cond1[0][:][:,1],label="cond1")
                    # plt.plot(samples_cond2[0][:][:,0],samples_cond2[0][:][:,1],label="cond2")
                    samples_uncond = dataset.normalizer.unnormalize(samples[0][i].cpu(), 'observations')
                    samples_cond1 = dataset.normalizer.unnormalize(samples[1][i].cpu(), 'observations')
                    #samples_cond2 = dataset.normalizer.unnormalize(samples[2][i].cpu(), 'observations')
                    samples_all= dataset.normalizer.unnormalize(samples[2][i].cpu(), 'observations')
                    plt.plot(samples_all[0][:][:,0],samples_all[0][:][:,1],label="composition")
                    plt.plot(samples_uncond[0][:][:,0],samples_uncond[0][:][:,1],label="uncond")
                    plt.plot(samples_cond1[0][:][:,0],samples_cond1[0][:][:,1],label="right cond")
                    #plt.plot(samples_cond2[0][:][:,0],samples_cond2[0][:][:,1],label="up cond")
                    #plot legend in upper left corner
                    plt.legend(loc="upper left")
                    plt.xlim(-5,50)
                    plt.ylim(-5,50)
                    plt.grid()
                    plt.show()
                    plt.title(f"Skill: Right. (Denoising Step {i+1})")
                    plt.savefig(f'/home/fernandi/projects/decision-diffuser/code/skills/analysis/reset_50/steps/{args.conditioning}/{i}.png')
                    plt.close()
                
            else:
                samples = dataset.normalizer.unnormalize(samples, 'observations')
                plt.plot(samples[0][:][:,0],samples[0][:][:,1],label="all")
                plt.legend()
                plt.grid()
                plt.show()
                plt.savefig(f'/home/fernandi/projects/decision-diffuser/code/skills/analysis/reset_50/real_multi_{args.conditioning}_{args.seed}_w{Config.condition_guidance_w[0]}_{Config.condition_guidance_w[1]}.png')
                

            import pdb; pdb.set_trace()
            #action = to_np(action)
            #action = dataset.normalizer.unnormalize(action, 'actions')

            if t == 0:
                normed_observations = samples[:, :, :]
            print("samples_unnorm",samples_unnorm)
            diff_x = samples_unnorm[0][:,0][1] -samples_unnorm[0][:,0][0]
            diff_y = samples_unnorm[0][:,1][1] - samples_unnorm[0][:,1][0]
            print(f"(change x) ",samples_unnorm[0][:,0][39]-samples_unnorm[0][:,0][0])
            print(f"(change y) ",samples_unnorm[0][:,1][39]-samples_unnorm[0][:,1][0])
            #print average of samples
            
            obs_list = []
                        
            i=0 
            current_action = np.array([diff_x,diff_y])
            #clip action to be in range [-1,1]
            current_action = np.clip(current_action,-1,1)
            this_obs, this_reward, done, info = env_list[i].step(current_action)
            
            list_action.append(current_action)
            #sum list of actions
            print("SUM ACTION: ", np.sum(np.array(list_action),axis=0))
            print("DONE:",done)
            print("STEP: ",t)
            list_obs.append(this_obs.tolist())
            env_list[i].render(current_action)
            obs = this_obs
            
            rewards_list.append(this_reward)
            # obs = np.concatenate(obs_list, axis=0)
            # recorded_obs.append(deepcopy(obs[:, None]))
            break
            t += 1
    print("Conditioning: ",args.conditioning)
    #savepath = os.path.join('images', f'sample-executed.png')
    #renderer.composite(savepath, recorded_obs)
    episode_rewards = np.array(episode_rewards)
    #plot list_obs
    list_obs = np.array(list_obs)
    list_action = np.array(list_action)
    action_mean = np.mean(list_action, axis=0)
    action_std = np.std(list_action, axis=0)
    list_action = np.append(list_action,[0,0]).reshape(-1,2)
    #insert in pandas dataframe
    df = pd.DataFrame(list_obs, columns=['agent_x','agent_y','goal_x','goal_y'])
    df['action_x'] = list_action[:,0]
    df['action_y'] = list_action[:,1]
    df['agent_x'] = list_obs[:,0]
    df['agent_y'] = list_obs[:,1]
    df['goal_x'] = list_obs[:,2]
    df['goal_y'] = list_obs[:,3]
    df['reward'] = rewards_list.append(0)
    df['action_mean_x'] = action_mean[0]
    df['action_mean_y'] = action_mean[1]
    df['action_std_x'] = action_std[0]
    df['action_std_y'] = action_std[1]
    #save dataframe
    if(len(Config.condition_guidance_w) == 1):
        Config.condition_guidance_w.append(0)
    df.to_csv(f'/home/fernandi/projects/decision-diffuser/code/skills/analysis/reset_50/{args.conditioning}_{args.seed}_w{Config.condition_guidance_w[0]}_{Config.condition_guidance_w[1]}.csv', index=False)
    
    
    plt.plot(list_obs[:,0],list_obs[:,1])
    #save plot
    plt.savefig(f'/home/fernandi/projects/decision-diffuser/code/skills/analysis/reset_50/{args.conditioning}_{args.seed}_w{Config.condition_guidance_w[0]}_{Config.condition_guidance_w[1]}.png')
    plt.show()
    #print avg and std of each dimension of action
    
   