import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class Safe_Grid(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self,render_mode=None,max_number_steps=30, start_x=0, start_y=1):

        self.observation_space= gym.spaces.Box(low=-5, high=25, shape=(4,), dtype=np.float32)
        self.square_obstacles = [((4, 6), (4, 6)), ((14, 16), (4, 6)), ((4, 6), (14, 16)), ((14, 16), (14, 16)), ((9, 11), (9, 11))]
        self.circle_obstacles = [(5,5), (15,5), (5,15), (15,15), (10,10)]
        self.obstacle_state_type = "absolute"
        self.obstacle_type = "circle"
        self.circle_width = 1.5
        if self.obstacle_type == "square":
            self.obstacles = self.square_obstacles
        elif self.obstacle_type == "circle":
            self.obstacles = self.circle_obstacles

        self.action_space= gym.spaces.Box(low=-1,high=1, shape=(2,), dtype=np.float32)
        self.goal_x_rand = 2
        self.goal_y_rand = 20
        self.reset_x = start_x
        self.reset_y = start_y
        self.state= np.array([np.random.uniform(self.reset_x,self.reset_y),np.random.uniform(self.reset_x,self.reset_y),
            np.random.uniform(self.goal_x_rand,self.goal_y_rand),np.random.uniform(self.goal_x_rand,self.goal_y_rand)])
        #add circle obstacles to state
        if self.obstacle_state_type == "absolute":
            for obstacle in self.circle_obstacles:
                self.state = np.append(self.state, obstacle)
        elif self.obstacle_state_type == "relative":
            for obstacle in self.circle_obstacles:
                self.state = np.append(self.state, np.linalg.norm(self.state[0:2] - obstacle))
        self.goal_distance = 1
        self.current_step =0
        self.max_episode_steps =max_number_steps
        self.reward=0
        self.cost = 0
        self.name = 'Safe_Grid'
        self.info={}
        self.info["goal_reached"]=False
        self.offset = 5
        self.lower_bound = np.array([-self.offset,-self.offset])
        self.upper_bound = np.array([self.goal_y_rand+self.offset, self.goal_y_rand+self.offset])
        self.last_dist_goal = None
        self.reward_goal = 1
        self.reward_distance = 1
        self.obstacle_cost = 3
        self.history = []
        self.acc_reward = 0
        self.acc_cost = 0
    
    
    def step(self, action):
        '''defines the logic of your environment when the agent takes an action
        Accepts an action, computes the state of the environment after applying that action
        '''
        done=False
      
        #setting the state of the environment based on agent's action
        # rewarding the agent for the action
        action = np.clip(action, -1.5, 1.5)
        self.state[0] += action[0]
        self.state[1] += action[1]
        distance_to_obs = self.distance_to_obs()
        index = 4
        for obs in distance_to_obs:
            self.state[index] = obs
            index += 1
        self.history.append(self.state.copy())
        distance_to_goal =np.linalg.norm(self.state[0:2]-self.state[2:4])
        self.reward = self._calculate_reward()
        self.current_step +=1
        #check for collision
        # if self._is_collision(self.state[0:2], self.obstacles):
        #     self.cost += 1

        self.cost = self.cal_cost()


        #if agent is out of bounds
        if self.state[0]<self.lower_bound[0] or self.state[0]>self.upper_bound[0] or self.state[1]<self.lower_bound[1] or self.state[1]>self.upper_bound[1]:
            self.reward -= 10
            done= True

        elif self.current_step>=self.max_episode_steps:
            done= True
            
        elif distance_to_goal<=self.goal_distance:
            self.reward += 10
            done= True
            info={}
            info["goal_reached"]=True
            self.acc_reward += self.reward
            self.acc_cost += self.cost
            return self.state, self.reward, self.cost, done, info

        self.acc_reward += self.reward
        self.acc_cost += self.cost
        return self.state, self.reward, self.cost, done, self.info
    # def render(self,action):
    #     # Visualize your environment
    #     print(f"\n Current position:{self.state[0:2]}\n Goal position: {self.state[2:4]} Reward Received:{self.reward} ")
    #     print(f" Action taken: delta_x: {action[0]}, delta_y: {action[1]}")
    #     print("==================================================")

    def get_new_goal(self):
        generate_new_goal = True
        offset = 1
        while generate_new_goal:
            goal_in_obstacle = False
            goal =np.array([np.random.uniform(self.goal_x_rand,self.goal_y_rand),np.random.uniform(self.goal_x_rand,self.goal_y_rand)])
            for h_pos in self.obstacles:
                h_dist = np.linalg.norm(goal - h_pos)
                if h_dist <= self.circle_width + offset:
                    goal_in_obstacle = True
                    
                
            generate_new_goal = True if goal_in_obstacle else False
            

        return goal

    def angle_between(self, p1, p2):
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        return np.rad2deg((ang1 - ang2) % (2 * np.pi))

    def distance_to_obs(self):
        """Distance to the obstacles."""
        distaces = []
        for obstacle in self.obstacles:
            distaces.append(np.linalg.norm(self.state[0:2] - obstacle))
            distaces.append(self.angle_between(self.state[0:2], obstacle))

        return distaces

    def reset(self):
        self.history = [] 
        self.acc_reward = 0
        self.acc_cost = 0
        goal = self.get_new_goal()
        self.state= np.array([np.random.uniform(self.reset_x,self.reset_y),np.random.uniform(self.reset_x,self.reset_y),goal[0],goal[1]])
        #print(f"Initial position:{self.state[0:2]} Goal position: {self.state[2:4]}")
        self.reward=0
        self.cost=0
        self.current_step=0
        self.info['goal_reached']=False
        self.last_dist_goal = self._dist_goal()
        distance_to_obs = self.distance_to_obs()
        for obs in distance_to_obs:
            self.state = np.append(self.state, obs)
    
        self.history.append(self.state.copy())

        return self.state

    def _is_collision(self, point, obstacles):
        for obstacle in obstacles:
            x_range, y_range = obstacle
            x, y = point
            if x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]:
                return True
        return False

    def _dist_goal(self):
        """Distance to the goal."""
        return np.linalg.norm(self.state[0:2] - self.state[2:4])

    def _calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
            
        # pylint: disable=no-member
        reward = 0.0
        dist_goal = self._dist_goal()
        #print(f"Last Distance to goal: {self.last_dist_goal} Current Distance to goal: {dist_goal}")
        reward += (self.last_dist_goal - dist_goal) * self.reward_distance
        self.last_dist_goal = dist_goal
        # if self.goal_achieved:
        #     reward += self.reward_goal

        return reward

    def cal_cost(self):
        """Contacts Processing."""
        cost = 0

        for h_pos in self.obstacles:
            h_dist = np.linalg.norm(self.state[0:2] - h_pos)
            if h_dist <= self.circle_width:
                cost += self.obstacle_cost * (self.circle_width - h_dist)

        return cost

    def render(self,extra=None):
        fig, ax = plt.subplots()
        history = np.array(self.history)
        # Plot the goal
        ax.plot(history[0,2], history[0,3], 'go', markersize=10, label='Goal')
        
        # Plot square obstacles
        # for obstacle in self.obstacles:
        #     x_range, y_range = obstacle
        #     ax.fill_between(x_range, y_range[0], y_range[1], facecolor='red', alpha=0.5, label='Obstacle')
        
        #plot circle obstacles
        for obstacle in self.circle_obstacles:
            x,y = obstacle
            circle = plt.Circle((x,y), self.circle_width, color='red', alpha=0.5, label='Obstacle')
            ax.add_artist(circle)
        
        # Plot the agent trajectory
        x_trajectory = history[:,0]
        y_trajectory = history[:,1]
        ax.plot(x_trajectory, y_trajectory, 'b-', label='Agent Trajectory')
        if extra is not None:
            ax.plot(extra, label='generated trajectory')

        # Set plot limits
        ax.set_xlim(self.lower_bound[0], self.upper_bound[0])
        ax.set_ylim(self.lower_bound[1], self.upper_bound[1])
        
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        #only print Obstacle label once
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        #print reawrd and cost on top
        reward = round(self.acc_reward,2)
        cost = round(self.acc_cost,2)
        plt.text(.5,.1, f"Reward: {reward}, Cost: {cost}", horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Safe Grid Rollout')
        plt.grid(True)
        plt.show()
        plt.savefig('/home/fernandi/projects/decision-diffuser/code/images/safe_grid.png')

    def close(self):
        # close the environment
        self.state=0
        self.reward=0
        self.cost=0
        self.current_step=0
        self.last_dist_goal = 0
        self.info['goal_reached']=False
        self.history = []
        self.acc_reward = 0
        self.acc_cost = 0

def main():
    env = Safe_Grid()
    print(env.reset())
    action = np.array([1,1])
    print(env.step(action))
    done = False
    while not done:
        # get random action
        action = [1,1]
        state, reward, cost, done, info = env.step(action)
        print(f"Next state: {state} Reward: {reward} Cost: {cost} Done: {done} Info: {info}")
    env.render()

if __name__ == "__main__":
    main()
