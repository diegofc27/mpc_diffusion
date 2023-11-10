from collections import namedtuple
import numpy as np
import torch
import pdb

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

CostBatch = namedtuple('Batch', 'trajectories conditions costs')
RewardBatch = namedtuple('Batch', 'trajectories conditions returns')
RewardCostBatch = namedtuple('Batch', 'trajectories conditions returns_costs')
SkillBatch = namedtuple('Batch', 'trajectories conditions skills')
Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, discount=0.99, returns_scale=1000, include_returns=False, skill_dataset=None, point_dataset=None,):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.returns_scale = returns_scale
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        itr = sequence_dataset(env, self.preprocess_fn,point_dataset,skill_dataset)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        self.returns_scale = self.fields.rewards.sum(axis=1).max()
        print(f'[ datasets ] Returns scale: {self.returns_scale}')
        self.cost_scale = self.fields.costs.sum(axis=1).max()
        print(f'[ datasets ] Costs scale: {self.cost_scale}')
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns/self.returns_scale], dtype=np.float32)
            batch = RewardBatch(trajectories, conditions, returns)
        else:
            batch = Batch(trajectories, conditions)

        return batch

class SequenceDatasetNew(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=1000000, termination_penalty=0, use_padding=True, discount=0.99, returns_scale=1000, costs_scale=130, include_returns=False,include_skills=False, 
        include_costs=False,point_dataset=None,skill_dataset=None):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.returns_scale = returns_scale
        self.cost_scale = costs_scale
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        self.include_skills = include_skills
        self.include_costs = include_costs
        itr = sequence_dataset(env, self.preprocess_fn,point_dataset,skill_dataset)
        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()
        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        #observations = self.fields.normed_observations[path_ind, start:end]
        #actions = self.fields.normed_actions[path_ind, start:end]
        observations = self.fields.observations[path_ind, start:end]
        actions = self.fields.actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        if self.include_returns and self.include_costs:
            rewards = self.fields.rewards[path_ind, start:]
            costs = self.fields.cost[path_ind, start:]
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            costs = (discounts * costs).sum()
            returns_costs = np.array([returns/self.returns_scale,costs/self.cost_scale], dtype=np.float32)
            batch = RewardCostBatch(trajectories, conditions, returns_costs)
        elif self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns/self.returns_scale], dtype=np.float32)
            batch = RewardBatch(trajectories, conditions, returns)
        elif self.include_costs:
            costs = self.fields.costs[path_ind, start:]
            discounts = self.discounts[:len(costs)]
            costs = (discounts * costs).sum()
            costs = np.array([costs/self.cost_scale], dtype=np.float32)
            batch = CostBatch(trajectories, conditions, costs)
        else:
            batch = Batch(trajectories, conditions)

        return batch
    
class StateDataset(SequenceDataset):
    def __init__(self, *args, include_skills=True, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([observations], axis=-1)
        
        rewards = self.fields.rewards[path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        returns = (discounts * rewards).sum()
        returns = np.array([returns/self.returns_scale], dtype=np.float32)
        batch = RewardBatch(trajectories, conditions, returns)
        

        return batch

    
class ActionDataset(SequenceDataset):
    def __init__(self, *args, include_skills=True, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(actions)
        trajectories = np.concatenate([actions], axis=-1)
        
        rewards = self.fields.rewards[path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        returns = (discounts * rewards).sum()
        returns = np.array([returns/self.returns_scale], dtype=np.float32)
        batch = RewardBatch(trajectories, conditions, returns)
        

        return batch

class SkillsDataset(SequenceDataset):

    def __init__(self, *args, include_skills=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_skills = include_skills
        self.one_hot = [[1.0,0.0],[0.0,1.0]]

    def get_one_hot(self, skill):
        return self.one_hot[skill]

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.actions[path_ind, start:end]
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        if self.include_skills:
            skills = self.fields.skills[path_ind, start:end][0]
            batch = SkillBatch(trajectories, conditions, skills)
        else:
            batch = Batch(trajectories, conditions)
        return batch
    
class SkillsXYDataset(SequenceDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.actions[path_ind, start:end]
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        
        one_hot = self.fields.skills[path_ind, start:end][0]
        skills = np.array([one_hot[0],one_hot[1],observations[0][0],observations[0][1]])
        batch = SkillBatch(trajectories, conditions, skills)

        return batch
    
class GoalsDataset(SequenceDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_one_hot(self, skill):
        return self.one_hot[skill]

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.actions[path_ind, start:end]
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        goal = observations[0][18:21]
        batch = SkillBatch(trajectories, conditions, goal)
        

        return batch

class RewardsDataset(SequenceDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.actions[path_ind, start:end]
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        rewards = self.fields.rewards[path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        returns = (discounts * rewards).sum()
        returns = np.array([returns/self.returns_scale], dtype=np.float32)
        batch = RewardBatch(trajectories, conditions, returns)
        return batch

class CostsDataset(SequenceDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.actions[path_ind, start:end]
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        costs = self.fields.costs[path_ind, start:end]
        discounts = self.discounts[:len(costs)]
        costs = (discounts * costs).sum()
        costs = np.array([costs], dtype=np.float32)
        batch = RewardBatch(trajectories, conditions, costs)
        return batch



class CondSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, discount=0.99, returns_scale=1000, include_returns=False):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.returns_scale = returns_scale
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        t_step = np.random.randint(0, self.horizon)

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        traj_dim = self.action_dim + self.observation_dim

        conditions = np.ones((self.horizon, 2*traj_dim)).astype(np.float32)

        # Set up conditional masking
        conditions[t_step:,:self.action_dim] = 0
        conditions[:,traj_dim:] = 0
        conditions[t_step,traj_dim:traj_dim+self.action_dim] = 1

        if t_step < self.horizon-1:
            observations[t_step+1:] = 0

        trajectories = np.concatenate([actions, observations], axis=-1)

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns/self.returns_scale], dtype=np.float32)
            batch = RewardBatch(trajectories, conditions, returns)
        else:
            batch = Batch(trajectories, conditions)

        return batch

class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }

class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch
