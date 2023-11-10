#Importing required libraries
import gym
from gym import spaces
import numpy as np
#Creating the custom environment
#Custom environment needs to inherit from the abstract class gym.Env
class Find_Dot(gym.Env):
    #add the metadata attribute to your class
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self,render_mode=None,max_number_steps=20, start_x=0, start_y=1):
        # define the environment's action_space and observation space
        
        '''Box-The argument low specifies the lower bound of each dimension and high specifies the upper bounds
        '''
        self.observation_space= gym.spaces.Box(low=-100, high=100, shape=(4,), dtype=np.float32)
         
        self.action_space= gym.spaces.Box(low=-1,high=1, shape=(2,), dtype=np.float32)
        self.goal_x_rand = 10
        self.goal_y_rand = 20
        self.state= np.array([np.random.uniform(0,5),np.random.uniform(0,5),np.random.uniform(self.goal_x_rand,self.goal_y_rand),np.random.uniform(self.goal_x_rand,self.goal_y_rand)])
        self.goal_distance = .5
        self.current_step =0
        self.reset_x = start_x
        self.reset_y = start_y
        self.max_episode_steps =max_number_steps
        self.reward=0
        self.name = 'Find_Dot'
        self.info={}
        self.info["goal_reached"]=False
        self.lower_bound = np.array([-10,-10])
        self.upper_bound = np.array([self.goal_y_rand+10, self.goal_y_rand+10])
    
    
    def step(self, action):
        '''defines the logic of your environment when the agent takes an actio
        Accepts an action, computes the state of the environment after applying that action
        '''
        done=False
      
        #setting the state of the environment based on agent's action
        # rewarding the agent for the action
        action = np.clip(action, -1.5, 1.5)
        self.state[0] +=action[0]
        self.state[1] +=action[1]
        distance_to_goal =np.linalg.norm(self.state[0:2]-self.state[2:4])
        self.reward = -np.linalg.norm(self.state[0:2]-self.state[2:4])
        self.current_step +=1
        # define the completion of the episode
        #if agent is out of bounds
        if self.state[0]<self.lower_bound[0] or self.state[0]>self.upper_bound[0] or self.state[1]<self.lower_bound[1] or self.state[1]>self.upper_bound[1]:
            self.reward -= 100
            done= True

        elif self.current_step>=self.max_episode_steps:
            done= True
            

        elif distance_to_goal<=self.goal_distance:
            self.reward += 100
            done= True
            info={}
            info["goal_reached"]=True
            return self.state, self.reward, done, info
        
        return self.state, self.reward, done, self.info
    def render(self,action):
        # Visualize your environment
        print(f"\n Current position:{self.state[0:2]}\n Goal position: {self.state[2:4]} Reward Received:{self.reward} ")
        print(f" Action taken: delta_x: {action[0]}, delta_y: {action[1]}")
        print("==================================================")
    def reset(self):
        self.state= np.array([np.random.uniform(self.reset_x,self.reset_y),np.random.uniform(self.reset_x,self.reset_y),np.random.uniform(self.goal_x_rand,self.goal_y_rand),np.random.uniform(self.goal_x_rand,self.goal_y_rand)])
        #print(f"Initial position:{self.state[0:2]} Goal position: {self.state[2:4]}")
        self.reward=0
        self.current_step=0
        self.info['goal_reached']=False
        return self.state
    def close(self):
        # close the environment
        self.state=0
        self.reward=0