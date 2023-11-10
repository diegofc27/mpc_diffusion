from pathlib import Path
import pickle
import numpy as np
from collections import defaultdict

from tqdm import tqdm

#Importing required libraries
import gym
from gym import spaces
import random
import numpy as np

class Find_Dot(gym.Env):
    #add the metadata attribute to your class
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self,render_mode=None,max_number_steps=50, start_x=[0,0], start_y=[50,50]):
        # define the environment's action_space and observation space
        
        '''Box-The argument low specifies the lower bound of each dimension and high specifies the upper bounds
        '''
        self.observation_space= gym.spaces.Box(low=-100, high=100, shape=(4,), dtype=np.float32)
         
        self.action_space= gym.spaces.Box(low=-1,high=1, shape=(2,), dtype=np.float32)
        self.goal_x_rand = 5
        self.goal_y_rand =50
        self.state= np.array([np.random.uniform(0,30),np.random.uniform(0,30),np.random.uniform(self.goal_x_rand,self.goal_y_rand),np.random.uniform(self.goal_x_rand,self.goal_y_rand)])
        self.goal_distance = .5
        self.current_step =0
        self.reset_x = start_x
        self.reset_y = start_y
        self.max_episode_steps =max_number_steps
        self.reward=0
        self.name = 'Find_Dot'
    
    
    def step(self, action):
        '''defines the logic of your environment when the agent takes an actio
        Accepts an action, computes the state of the environment after applying that action
        '''
        done=False
        info={}
      
        #setting the state of the environment based on agent's action
        # rewarding the agent for the action
        action = np.clip(action, -1, 1)
        self.state[0] +=action[0]
        self.state[1] +=action[1]
        distance_to_goal =np.linalg.norm(self.state[0:2]-self.state[2:4])
        self.reward = -np.linalg.norm(self.state[0:2]-self.state[2:4])
        self.current_step +=1
        # define the completion of the episode
 
        if self.current_step>=self.max_episode_steps: #or distance_to_goal<=self.goal_distance:
            done= True
        return self.state, self.reward, done, info
    def render(self,action):
        # Visualize your environment
        print(f"\n Current position:{self.state[0:2]}\n Goal position: {self.state[2:4]} Reward Received:{self.reward} ")
        print(f" Action taken: delta_x: {action[0]}, delta_y: {action[1]}")
        print("==================================================")
    def reset(self):
        #reset your environment
        self.state= np.array([np.random.uniform(self.reset_x[0],self.reset_x[1]),np.random.uniform(self.reset_y[0],self.reset_y[1]),np.random.uniform(self.goal_x_rand,self.goal_y_rand),np.random.uniform(self.goal_x_rand,self.goal_y_rand)])
        self.reward=0
        self.current_step=0
        return self.state
    def close(self):
        # close the nevironment
        self.state=0
        self.reward=0


# def generate_x(x_k):
#   #data generation 
#   noise = np.clip(np.random.normal(.5,1),-1,1)
#   x_k1=noise + x_k
#   action = x_k1 - x_k
#   return x_k1,action

# def generate_y(y_k):
#   d = np.random.choice([-1,1],p=[.2,.8])
#   action = (d *np.random.uniform(.4,.8))
#   y_k1= action + y_k
#   return y_k1, action

def generate_x(x_k):
  #data generation 
  action = np.clip(np.random.normal(1,.5),-1,1)
  x_k1=action + x_k
  return x_k1,action

def generate_y(y_k):
  action = np.clip(np.random.normal(.85,.65),-1,1)
  y_k1= action + y_k
  return y_k1, action

# def generate_x_y(x_k,y_k):
#   noise_x =np.random.normal(.1,.3)
#   noise_y = np.random.normal(.1,.3)
#   action_x = np.clip(np.random.normal(1,.3) + noise_x,-1,1)
#   action_y = np.clip(np.random.normal(.15,.3)+ noise_y,-1,1)
#   x_k +=action_x
#   y_k +=action_y
#   return action_x,action_y,x_k,y_k

# def generate_y_x(x_k,y_k):
#   noise_x =np.random.normal(.1,.3)
#   noise_y = np.random.normal(.1,.3)
#   action_y = np.clip(np.random.normal(1,.3) + noise_x,-1,1)
#   action_x= np.clip(np.random.normal(.15,.3)+ noise_y,-1,1)
#   x_k +=action_x
#   y_k +=action_y
#   return action_x,action_y,x_k,y_k

# def generate_x_y(x_k,y_k):
#   noise_x =np.random.normal(.01,.3)
#   action_x = np.clip(np.random.normal(1,.8) + noise_x,-1,1)
#   action_y = np.clip(np.random.normal(.07,.1),-1,1)
#   x_k +=action_x
#   y_k +=action_y
#   return action_x,action_y,x_k,y_k

# def generate_y_x(x_k,y_k):
#   noise_y = np.random.normal(.01,.3)
#   action_x = np.clip(np.random.normal(.07,.1),-1,1)
#   action_y = np.clip(np.random.normal(1,.8)+ noise_y,-1,1)
#   x_k +=action_x
#   y_k +=action_y
#   return action_x,action_y,x_k,y_k

def generate_x_y(x_k,y_k):
  noise_x =np.random.normal(.01,.2)
  action_x = np.clip(np.random.normal(1.5,.2) + noise_x,-1,1)
  action_y = np.clip(np.random.normal(0,.4),-1,1)
  x_k +=action_x
  y_k +=action_y
  return action_x,action_y,x_k,y_k

def generate_y_x(x_k,y_k):
  noise_y = np.random.normal(.01,.3)
  action_x = np.clip(np.random.normal(0,.2),-1,1)
  action_y = np.clip(np.random.normal(1.5,.2)+ noise_y,-1,1)
  x_k +=action_x
  y_k +=action_y
  return action_x,action_y,x_k,y_k


def collect_x_y_episode(dataset,max_number_steps=20):
  seeds = np.random.randint(0,10000,size=10000) +238
  idx = np.random.randint(0,10000)
  np.random.seed(seeds[idx])
  env = Find_Dot(max_number_steps=max_number_steps, start_x=[0,0], start_y=[0,50])
  done = False
  state = env.reset()
  x_y_list = [state[0:2].tolist()]
  while not done:
    dataset['observations'].append(state.tolist())
    action_x,action_y,x_k,y_k =generate_x_y(state[0],state[1])
    state, reward, done, info = env.step([action_x,action_y]) 
    #x_y_list.append([x_k,y_k])
    dataset['actions'].append([action_x,action_y])
    dataset['rewards'].append(reward)
    dataset['terminals'].append(done)
  env.close()
  #plot the trajectory
  # import matplotlib.pyplot as plt
  # plt.plot(np.array(x_y_list)[:,0],np.array(x_y_list)[:,1])
  # plt.show()
  # plt.savefig('/home/fernandi/projects/decision-diffuser/code/scripts/trajectory_xy.png')
  return dataset

def collect_y_x_episode(dataset,max_number_steps=20):
  seeds = np.random.randint(0,10000,size=10000) +234
  idx = np.random.randint(0,10000)
  np.random.seed(seeds[idx])
  env = Find_Dot(max_number_steps=max_number_steps, start_x=[0,50], start_y=[0,0])
  done = False
  state = env.reset()
  y_x_list = [state[0:2].tolist()]
  while not done:
    dataset['observations'].append(state.tolist())
    action_x,action_y,x_k,y_k =generate_y_x(state[0],state[1])
    state, reward, done, info = env.step([action_x,action_y])
    #y_x_list.append([x_k,y_k]) 
    dataset['actions'].append([action_x,action_y])
    dataset['rewards'].append(reward)
    dataset['terminals'].append(done)
  env.close()
  # import matplotlib.pyplot as plt
  # plt.plot(np.array(y_x_list)[:,0],np.array(y_x_list)[:,1])
  # plt.show()
  # plt.savefig('/home/fernandi/projects/decision-diffuser/code/scripts/trajectory_yx.png')
  
  return dataset

def collect_x_episode(dataset,max_number_steps=20):
  seeds = np.random.randint(0,10000,size=10000) +234
  idx = np.random.randint(0,10000)
  np.random.seed(seeds[idx])

  env = Find_Dot(max_number_steps=max_number_steps,reset_x=0, reset_y=50)
  done = False
  state = env.reset()
  while not done:
    dataset['observations'].append(state.tolist())
    _, action_x =generate_x(state[0])
    state, reward, done, info = env.step([action_x,0]) 
    dataset['actions'].append([action_x,0])
    dataset['rewards'].append(reward)
    dataset['terminals'].append(done)
  env.close()
  return dataset

def collect_y_episode(dataset,max_number_steps=20):
  seeds = np.random.randint(0,10000,size=10000) +234
  idx = np.random.randint(0,10000)
  np.random.seed(seeds[idx])
  env = Find_Dot(max_number_steps=max_number_steps)
  done = False
  state = env.reset()

  while not done:
    dataset['observations'].append(state.tolist())
    _, action_y =generate_y(state[1])
    state, reward, done, info = env.step([0,action_y]) 
    dataset['actions'].append([0,action_y])
    dataset['rewards'].append(reward)
    dataset['terminals'].append(done)
  env.close()
  return dataset

def collect_n_episodes(n,mode='x',max_number_steps=20):

  dataset = defaultdict(list)
  pbar = tqdm(total=n)
  for _ in range(n):
    if mode=='x':
      dataset = collect_x_episode(dataset,max_number_steps)
    elif mode=='y':  
      dataset = collect_y_episode(dataset,max_number_steps)
    elif mode=='xy':
      dataset = collect_x_y_episode(dataset,max_number_steps)
    elif mode=='yx':
      dataset = collect_y_x_episode(dataset,max_number_steps)
    pbar.update(1)
  #import pdb; pdb.set_trace()
  dataset['observations'] = np.array(dataset['observations']).reshape(-1,4)
  dataset['actions'] = np.array(dataset['actions']).reshape(-1,2)
  dataset['rewards'] = np.array(dataset['rewards']).reshape(-1,)
  dataset['terminals'] = np.array(dataset['terminals']).reshape(-1,)
  #save the dataset as pickle file
  path = Path(f"/home/fernandi/projects/decision-diffuser/code/skills/{mode}_{n}_dataset_{max_number_steps}.pickle")
  with open(path, "wb") as fout:
    pickle.dump(dataset, fout)

  print("Num env steps: " ,dataset['observations'].shape[0])
  pbar.close()
  return dataset

def merge_datasets(n,mode="x",max_number_steps=20):
  #LOAD THE pickled dataset
  if mode=='x':
    with open(f'/home/fernandi/projects/decision-diffuser/code/skills/xy_{n}_dataset_{max_number_steps}.pickle', 'rb') as handle: 
      x_dataset = pickle.load(handle)
    with open(f'/home/fernandi/projects/decision-diffuser/code/skills/yx_{n}_dataset_{max_number_steps}.pickle', 'rb') as handle: 
      y_dataset = pickle.load(handle)
  elif mode=='xy':
    with open(f'/home/fernandi/projects/decision-diffuser/code/skills/xy_{n}_dataset_{max_number_steps}.pickle', 'rb') as handle: 
      x_dataset = pickle.load(handle)
    with open(f'/home/fernandi/projects/decision-diffuser/code/skills/yx_{n}_dataset_{max_number_steps}.pickle', 'rb') as handle: 
      y_dataset = pickle.load(handle)
  hot_x = np.array([[1.0,0.0]],dtype=np.float32)
  hot_y = np.array([[0.0,1.0]],dtype=np.float32)
  x_dataset['skills'] = np.repeat(hot_x, x_dataset['observations'].shape[0], axis=0)
  y_dataset['skills'] = np.repeat(hot_y, y_dataset['observations'].shape[0], axis=0)
  #print dimensions
  print(x_dataset['observations'].shape)
  print(x_dataset['skills'].shape)
  print(x_dataset['terminals'].shape)
  print(y_dataset['observations'].shape)
  print(y_dataset['skills'].shape)
  print(y_dataset['terminals'].shape)

  combined_dim = x_dataset['observations'].shape[0] + y_dataset['observations'].shape[0]

  dataset = defaultdict(list)
  for k in x_dataset:
    dataset[k].extend(x_dataset[k])
  for k in y_dataset:
    dataset[k].extend(y_dataset[k])
  #convert to numpy array
  dataset['observations'] = np.array(dataset['observations']).reshape(-1,4)
  dataset['actions'] = np.array(dataset['actions']).reshape(-1,2)
  dataset['rewards'] = np.array(dataset['rewards']).reshape(-1,)
  dataset['terminals'] = np.array(dataset['terminals']).reshape(-1,)
  dataset['skills'] = np.array(dataset['skills']).reshape(-1,2)

  assert dataset['observations'].shape[0] == combined_dim
  #save the dataset as pickle file
  path = Path(f"/home/fernandi/projects/decision-diffuser/code/skills/xy_{n*2}_dataset_{max_number_steps}.pickle")
  with open(path, "wb") as fout:
    pickle.dump(dataset, fout)


  

if __name__ == "__main__":
  max_number_steps = 50 
  collect_n_episodes(500,mode='xy',max_number_steps=max_number_steps)
  collect_n_episodes(500,mode='yx',max_number_steps=max_number_steps)
  merge_datasets(500,mode="xy",max_number_steps=max_number_steps)