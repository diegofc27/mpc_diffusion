# import sys
# sys.path.append("/home/fernandi/projects")
from collections import OrderedDict
from rl_zoo3 import create_test_env
hyperparams = OrderedDict([('batch_size', 2048), ('buffer_size', 1000000), ('env_wrapper', 'sb3_contrib.common.wrappers.TimeFeatureWrapper'), ('gamma', 0.95), ('learning_rate', 0.001), ('n_timesteps', 3000000.0), ('normalize', False), ('policy', 'MultiInputPolicy'), ('policy_kwargs', 'dict(net_arch=[512, 512, 512], n_critics=2)'), ('replay_buffer_class', 'HerReplayBuffer'), ('replay_buffer_kwargs', "dict(goal_selection_strategy='future', n_sampled_goal=4,)"), ('tau', 0.05)])
env = create_test_env(
    "PandaPush-v3",
    n_envs=1,
    stats_path='/home/fernandi/projects/decision-diffuser/code/scripts',
    seed=10,
    log_dir=None,
    should_render=False,
    hyperparams=hyperparams,
    env_kwargs={},
)

obs = env.reset()
import pdb; pdb.set_trace()
     