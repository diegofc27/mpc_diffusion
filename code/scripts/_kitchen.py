import gym
import d4rl # Import required to register environments, you may need to also import the submodule
import numpy as np
# Create the environment
env = gym.make('kitchen-complete-v0')

# d4rl abides by the OpenAI gym interface
env.reset()
action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
env.step(action)

# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()
print(dataset['observations'])
import pdb; pdb.set_trace()