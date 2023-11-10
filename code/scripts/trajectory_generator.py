from collections import defaultdict
import matplotlib.pyplot as plt 
from pathlib import Path
from tqdm import tqdm
import numpy as np 
import pickle

def ornstein_uhlenbeck(sigma=1,tmax=20.,stepslist=[40]):
  gamma = .5
  for steps in stepslist:
      h = tmax/steps
      std = np.sqrt(h)
      k = 1
      x0 = np.linspace(0.,5.,k)
      noise = np.random.randn(steps)*std
      sde = np.zeros((steps,k))
      sde[0] = x0

      for n in range(steps-1):
          sde[n+1] = sde[n]-gamma*h*sde[n]+sigma*noise[n]

      t = np.arange(0,steps,1)*h
  sde = np.array([(index, value.item()) for (index, value) in zip(t,sde)])
  return sde

def rotate(p, origin=(0, 0), degrees=0):
  angle = np.deg2rad(degrees)
  R = np.array([[np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]])
  o = np.atleast_2d(origin)
  p = np.atleast_2d(p)
  return np.squeeze((R @ (p.T-o.T) + o.T).T)

def plot(sde, sde_rotated):
  plt.plot(sde[:,0],sde[:,1],linewidth=0.3,color="red",label="original")
  plt.plot(sde_rotated[:,0],sde_rotated[:,1],linewidth=0.3,color="green", label="rotated")
  plt.legend()
  plt.show()

def invalid_trajectory(sde_rotated):
    sde_ = sde_rotated[1:]
    diff= sde_ - sde_rotated[:-1]

    if diff.min()<-1.5 or diff.max()>1.5:
      return True
    else:
      return False

def generate_trajectory(num_trajectories=1, angle=45):
    steps =81
    num_valid_trajectories = 0
    while num_valid_trajectories < num_trajectories:
        sigma = np.random.uniform(.5,1)
        steps = np.random.randint(20,40)
        #generate trajectory
        sde = ornstein_uhlenbeck(sigma=sigma,stepslist=[steps], tmax=40)
        #rotate between 5 - 90 degrees
        sde_rotated = rotate(sde,degrees=angle)
        
        if invalid_trajectory(sde_rotated):
            continue
        num_valid_trajectories+=1
        
        #assing action, state, reward
        goal = sde_rotated[-1]
        goal_array = np.repeat([goal],steps-1,axis=0)

        observations = (np.hstack((sde_rotated[:-1],goal_array)))
        actions = (np.diff(sde_rotated, axis=0))
    
    return observations, actions

def generate_trajectories(num_trajectories=1000):
    pbar = tqdm(total=num_trajectories)
    dataset = defaultdict(list)
    steps =41
    num_valid_trajectories = 0
    while num_valid_trajectories < num_trajectories:
        sigma = np.random.uniform(0.5,1.5)
        steps = np.random.randint(20,40)
        #generate trajectory
        sde = ornstein_uhlenbeck(sigma=sigma,stepslist=[steps], tmax=20)
        #rotate between 5 - 90 degrees
        angle = np.random.randint(20,60)
        sde_rotated = rotate(sde,degrees=angle)
        
        if invalid_trajectory(sde_rotated):
            continue
        num_valid_trajectories+=1
        
        pbar.update(1)
        #assing action, state, reward
        goal = sde_rotated[-1]
        goal_array = np.repeat([goal],steps-1,axis=0)

        dataset["observations"] = (np.hstack((sde_rotated[:-1],goal_array)))
        dataset["actions"] = (np.diff(sde_rotated, axis=0))
        dataset["rewards"] = (-np.linalg.norm(sde_rotated[:-1] - goal, axis=1))
        terminals = np.repeat(False,steps-1)
        terminals[-1] = True
        dataset["terminals"] = (terminals)
        

    dataset["observations"] = np.array(dataset["observations"]).reshape(-1,4)
    dataset["actions"] = np.array(dataset["actions"]).reshape(-1,2)
    dataset["rewards"] = np.array(dataset["rewards"]).reshape(-1)
    dataset["terminals"] = np.array(dataset["terminals"]).reshape(-1)
    pbar.close()
    return dataset, num_valid_trajectories

def main():
    trajectories = 20000
    dataset, num = generate_trajectories(trajectories)
    print("Number of valid trajectories: ", num)
    path = Path(f"/home/fernandi/projects/decision-diffuser/code/trajectories/dataset_{num}.pickle")
    with open(path, "wb") as fout:
        pickle.dump(dataset, fout)

if __name__ == "__main__":
    main()