import torch

from params_proto import ParamsProto, PrefixProto, Proto
#from params_proto.neo_proto import ParamsProto, PrefixProto, Proto
class Config(ParamsProto):
    # misc
    seed = 100
    device = 'cuda:6' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bucket = '/home/fernandi/projects/decision-diffuser/code/weights/'
    runs = '/home/fernandi/projects/decision-diffuser/code/'
    dataset = 'hopper-medium-expert-v2'

    ## model
    model = 'models.TemporalUnet'
    diffusion = 'models.GaussianInvDynDiffusion'
    horizon = 100
    n_diffusion_steps = 200
    action_weight = 10
    loss_weights = None
    loss_discount = 1
    predict_epsilon = True
    dim_mults = (1, 4, 8)
    returns_condition = True
    skills_condition = False
    goal_condition = False
    calc_energy=False
    dim=128
    condition_dropout=0.25
    condition_guidance_w = 1.2
    test_ret=0.9
    renderer = 'utils.MuJoCoRenderer'
    attention = False
    include_xy = False

    ## dataset
    loader = 'datasets.SequenceDataset'
    normalizer = 'CDFNormalizer'
    preprocess_fns = []
    clip_denoised = True
    use_padding = True
    include_returns = True
    include_costs = False
    discount = 0.99
    max_path_length = 1000
    hidden_dim = 256
    ar_inv = False
    train_only_inv = False
    termination_penalty = 0
    returns_scale = 10.0 # Determined using rewards from the dataset
    costs_scale = 7
    max_n_episodes = 1000000
    point_dataset = 'xy_dataset_20'
    skill_dataset = 'xy_dataset_20'

    ## training
    n_steps_per_epoch = 10000
    loss_type = 'l2'
    n_train_steps = 1e6
    batch_size = 32
    learning_rate = 2e-4
    gradient_accumulate_every = 2
    ema_decay = 0.995
    log_freq = 1000
    save_freq = 10000
    sample_freq = 10000
    n_saves = 5
    save_parallel = False
    n_reference = 8
    save_checkpoints = False

    #wandb
    wandb_project = 'decision-diffuser'
    wandb_entity = 'diegofc'
    wandb_group = 'PandaPush-v3'
    wandb_tags = [  'decision-diffuser']
    wandb_name = "test"