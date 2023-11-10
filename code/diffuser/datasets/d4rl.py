import os
import collections
import numpy as np
import gym
#import d4rl
import pdb
# import gymnasium as gym
# import panda_gym
from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)
import pickle
from diffuser.environments.point import Find_Dot

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

# with suppress_output():
#     ## d4rl prints out a variety of warnings
#     import d4rl

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    if name == 'FindDot-v0':
        return Find_Dot(max_number_steps=30)
    if name == 'SafePointExp-v1':
        import safety_gym  # noqa
        return gym.make(name)
    if name == 'SafetyPointButton3-v0':
        from omnisafe.envs.safety_gymnasium_env import SafetyGymnasiumEnv
        return SafetyGymnasiumEnv(name)
    if name == 'SafeGrid-v0':
        from diffuser.environments.safe_grid import Safe_Grid
        return Safe_Grid()
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env

def get_dataset(env,point_dataset="point_1",skill_dataset="PandaPushDense-v3_single_seed_test_123"):
    if(env.__class__.__name__=='Safe_Grid'):
        print(f"Using pickle: {point_dataset}")
        with open(f'/home/fernandi/projects/decision-diffuser/code/trajectories/{point_dataset}.pickle', 'rb') as handle:
            dataset = pickle.load(handle)
    else:
        # if(env.unwrapped.spec.id=='PandaPushDense-v3'):
        #     with open(f'/home/fernandi/projects/decision-diffuser/code/skills/{skill_dataset}.pickle', 'rb') as handle:
        #         dataset = pickle.load(handle)
        #         print("loaded pickle")
        # else:
        with open(f'/home/fernandi/projects/decision-diffuser/code/skills/{skill_dataset}.pickle', 'rb') as handle:
            dataset = pickle.load(handle)
            print("loaded pickle")
            #dataset = env.get_dataset()
    print("episodes")
    print((dataset['terminals']==True).sum())

    if 'antmaze' in str(env).lower():
        ## the antmaze-v0 environments have a variety of bugs
        ## involving trajectory segmentation, so manually reset
        ## the terminal and timeout fields
        dataset = antmaze_fix_timeouts(dataset)
        dataset = antmaze_scale_rewards(dataset)
        get_max_delta(dataset)

    return dataset

def sequence_dataset(env, preprocess_fn,point_dataset="xy_dataset_20",skill_dataset="PandaPushDense-v3_single_seed_test_123"):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    dataset = get_dataset(env,point_dataset,skill_dataset)
    dataset = preprocess_fn(dataset)
    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset
    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            #final_timestep = (episode_step == env._max_episode_steps - 1)
            final_timestep = (episode_step == 1000 - 1)
        for k in dataset:
            if 'metadata' in k: continue
            if 'infos' in k: continue
            data_[k].append(dataset[k][i])
        if done_bool:        
        #if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            # if 'maze2d' in env.name:
            #     episode_data = process_maze2d_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode
