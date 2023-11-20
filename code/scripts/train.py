import os
import diffuser.utils as utils
import torch
import wandb
def main(**deps):
    from ml_logger import logger, RUN
    from config.locomotion_config import Config
    
    RUN._update(deps)
    Config._update(deps)
    # logger.remove('*.pkl')
    # logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))
    logger.log_text("""
                    charts:
                    - yKey: loss
                      xKey: steps
                    - yKey: a0_loss
                      xKey: steps
                    """, filename=".charts.yml", dedent=True, overwrite=True)

    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)
    Config.device = "cuda:4"
    # wandb.init(
    # # set the wandb project where this run will be logged
    #     project=Config.wandb_project,
    #     entity=Config.wandb_entity,
    #     group=Config.wandb_group,
    #     name=Config.wandb_name,
    #     # track hyperparameters and run metadata
    #     config=Config.__dict__
    # )

    # -----------------------------------------------------------------------------#
    # ---------------------------------- dataset ----------------------------------#
    # -----------------------------------------------------------------------------#
    print("Dataset: ", Config.dataset)
    Config.runs = os.path.join(Config.runs, logger.prefix)
    print("Runs: ", Config.runs)

    dataset_config = utils.Config(
        Config.loader,
        savepath=f'{Config.runs}/dataset_config.pkl',
        env=Config.dataset,
        horizon=Config.horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        use_padding=Config.use_padding,
        max_path_length=Config.max_path_length,
        include_returns=Config.include_returns,
        #include_costs=Config.include_costs,
        returns_scale=Config.returns_scale,
        discount=Config.discount,
        termination_penalty=Config.termination_penalty,
        max_n_episodes=Config.max_n_episodes,
        skill_dataset=Config.skill_dataset,
        point_dataset=Config.point_dataset,
    )

    # render_config = utils.Config(
    #     Config.renderer,
    #     savepath='render_config.pkl',
    #     env=Config.dataset,
    # )

    dataset = dataset_config()
    #renderer = render_config()
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#
    if Config.diffusion == 'models.GaussianInvDynDiffusion' or Config.diffusion == 'models.GaussianInvDynDiffusionSkills' or Config.diffusion == 'models.GaussianStaticInvDynDiffusion':
        model_config = utils.Config(
            Config.model,
            savepath=f'{Config.runs}/model_config.pkl',
            horizon=Config.horizon,
            transition_dim=observation_dim,
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
            savepath=f'{Config.runs}/diffusion_config.pkl',
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
            # skills_condition=Config.skills_condition,
            condition_guidance_w=Config.condition_guidance_w,
            # goal_condition=Config.goal_condition,
            device=Config.device,
        )
    elif Config.diffusion == 'models.GaussianDiffusionSkills':
        model_config = utils.Config(
            Config.model,
            savepath=f'{Config.runs}/model_config.pkl',
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
            savepath=f'{Config.runs}/diffusion_config.pkl',
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
    elif Config.loader == "datasets.StateDataset":
        model_config = utils.Config(
            Config.model,
            savepath=f'{Config.runs}/model_config.pkl',
            horizon=Config.horizon,
            transition_dim=observation_dim,
            cond_dim=observation_dim,
            dim_mults=Config.dim_mults,
            returns_condition=Config.returns_condition,
            skills_condition=Config.skills_condition,
            dim=Config.dim,
            condition_dropout=Config.condition_dropout,
            calc_energy=Config.calc_energy,
            device=Config.device,
        )

        diffusion_config = utils.Config(
            Config.diffusion,
            savepath=f'{Config.runs}/diffusion_config.pkl',
            horizon=Config.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_timesteps=Config.n_diffusion_steps,
            loss_type=Config.loss_type,
            clip_denoised=Config.clip_denoised,
            predict_epsilon=Config.predict_epsilon,
            ## loss weighting
            action_weight=Config.action_weight,
            loss_weights=Config.loss_weights,
            loss_discount=Config.loss_discount,
            returns_condition=Config.returns_condition,
            skills_condition=Config.skills_condition,
            condition_guidance_w=Config.condition_guidance_w,
            device=Config.device,
            only_states=True,
        )

    else:
        model_config = utils.Config(
            Config.model,
            savepath=f'{Config.runs}/model_config.pkl',
            horizon=Config.horizon,
            transition_dim=observation_dim + action_dim,
            cond_dim=observation_dim,
            dim_mults=Config.dim_mults,
            returns_condition=Config.returns_condition,
            skills_condition=Config.skills_condition,
            dim=Config.dim,
            condition_dropout=Config.condition_dropout,
            calc_energy=Config.calc_energy,
            device=Config.device,
        )

        diffusion_config = utils.Config(
            Config.diffusion,
            savepath=f'{Config.runs}/diffusion_config.pkl',
            horizon=Config.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_timesteps=Config.n_diffusion_steps,
            loss_type=Config.loss_type,
            clip_denoised=Config.clip_denoised,
            predict_epsilon=Config.predict_epsilon,
            ## loss weighting
            action_weight=Config.action_weight,
            loss_weights=Config.loss_weights,
            loss_discount=Config.loss_discount,
            returns_condition=Config.returns_condition,
            skills_condition=Config.skills_condition,
            condition_guidance_w=Config.condition_guidance_w,
            device=Config.device,
        )
    trainer_config = utils.Config(
        utils.Trainer,
        savepath=f'{Config.runs}/trainer_config.pkl',
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
        save_checkpoints=Config.save_checkpoints,
        test_freq=Config.test_freq,
        config=Config.__dict__,
        
    )

    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#

    model = model_config()

    diffusion = diffusion_config(model)

    trainer = trainer_config(diffusion, dataset, None, wandb=None)
   
    loadpath = os.path.join(Config.bucket, logger.prefix, 'checkpoint')
    loadpath = os.path.join(loadpath, 'state.pt')
    #check of file exists
    # if os.path.exists(loadpath):
    #     print("loading from: ",loadpath)
    #     state_dict = torch.load(loadpath, map_location=Config.device)
    #     trainer.step = state_dict['step']
    #     trainer.model.load_state_dict(state_dict['model'])
    #     trainer.ema_model.load_state_dict(state_dict['ema'])
    # -----------------------------------------------------------------------------#
    # ------------------------ test forward & backward pass -----------------------#
    # -----------------------------------------------------------------------------#

    utils.report_parameters(model)

    # logger.print('Testing forward...', end=' ', flush=True)
    # batch = utils.batchify(dataset[0], Config.device)
    # loss, _ = diffusion.loss(*batch)
    # loss.backward()
    # logger.print('✓')
    # -----------------------------------------------------------------------------#
    # --------------------------------- main loop ---------------------------------#
    # -----------------------------------------------------------------------------#

    n_epochs = int(Config.n_train_steps // Config.n_steps_per_epoch)

    for i in range(n_epochs):
        logger.print(f'Epoch {i} / {n_epochs} | {logger.prefix}')
        trainer.train(n_train_steps=Config.n_steps_per_epoch)

