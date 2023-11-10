import gymnasium as gym
import panda_gym
from collections import OrderedDict
from rl_zoo3 import create_test_env
import numpy as np
num_eval =10
hyperparams = OrderedDict([('batch_size', 2048), ('buffer_size', 1000000), ('env_wrapper', 'sb3_contrib.common.wrappers.TimeFeatureWrapper'), ('gamma', 0.95), ('learning_rate', 0.001), ('n_timesteps', 3000000.0), ('normalize', False), ('policy', 'MultiInputPolicy'), ('policy_kwargs', 'dict(net_arch=[512, 512, 512], n_critics=2)'), ('replay_buffer_class', 'HerReplayBuffer'), ('replay_buffer_kwargs', "dict(goal_selection_strategy='future', n_sampled_goal=4,)"), ('tau', 0.05)])
env = create_test_env(
    "PandaPushDense-v3",
    n_envs=num_eval,
    stats_path='/home/fernandi/projects/decision-diffuser/code/scripts',
    seed=77,
    log_dir=None,
    should_render=False,
    hyperparams=hyperparams,
    env_kwargs={},
)

dones = [0 for _ in range(num_eval)]
episode_rewards = [0 for _ in range(num_eval)]

# #change dtype os skills tensor to float32
# skills = skills.type(torch.float32)

t = 0
obs_list = env.reset() 
import pdb; pdb.set_trace()
for i in range(10):
    print(i)
    dg = obs_list['desired_goal'][i].tolist()
    ag = obs_list['achieved_goal'][i].tolist()
    o = obs_list['observation'][i].tolist()
    extra = np.append(dg,ag)
    obs_list = [np.append(o,extra)]
    import pdb; pdb.set_trace()