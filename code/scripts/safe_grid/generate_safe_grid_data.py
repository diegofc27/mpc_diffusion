from diffuser.environments.safe_grid import Safe_Grid
from scripts.safe_grid.trajectory_generator import generate_trajectory
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import pickle

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def main(num_trajectories):
    dataset = defaultdict(list)
    pbar = tqdm(total=num_trajectories)
    for _ in range(num_trajectories):
        env = Safe_Grid()
        state = env.reset()
        goal = state[2:4]
        angle = angle_between(goal, [0,0])
        observations, actions = generate_trajectory(angle=angle)
        done = False
        step = 0
        while not done:
            # get random action
            #random noise 
            #noise = np.random.normal(-.0, .0, 2)
            noise = np.array([0,0])
            action = actions[step] + noise

            state, reward, cost, done, info = env.step(action)
            dataset["observations"].append(state.copy())
            dataset["actions"].append(action)
            dataset["rewards"].append(reward)
            dataset["costs"].append(cost)
            dataset["terminals"].append(done)
            step+=1
        pbar.update(1)
        env.close()
    dataset["observations"] = np.array(dataset["observations"]).reshape(-1,14)
    dataset["actions"] = np.array(dataset["actions"]).reshape(-1,2)
    dataset["rewards"] = np.array(dataset["rewards"]).reshape(-1)
    dataset["costs"] = np.array(dataset["costs"]).reshape(-1)
    dataset["terminals"] = np.array(dataset["terminals"]).reshape(-1)  
    pbar.close()
    path = Path(f"/home/fernandi/projects/decision-diffuser/code/trajectories/safe_grid_{num_trajectories}_0_noise.pickle")
    with open(path, "wb") as fout:
        pickle.dump(dataset, fout)


if __name__ == "__main__":
    main(2000)