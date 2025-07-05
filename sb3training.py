import warnings
import numpy as np
import myosuite
import gym
from stable_baselines3 import PPO
from multiprocessing import Pool, Manager
from tqdm import tqdm
import time
import os

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# Initialize the environment and model outside the multiprocessing pool
env = gym.make('CenterReachOut1-v0')
env.set_perturbation_force(0)
output_dir = "evaluation_results"
os.makedirs(output_dir, exist_ok=True)

model = PPO.load('CenterReachOut_test.pickle')

# Define a function to run a single evaluation with progress bar
def evaluate_policy(iteration_start, iteration_end, total_iterations, manager_dict):
    counter = 0
    # Access shared data from the manager
    action_arr = manager_dict['action_arr']
    vel_arr = manager_dict['vel_arr']
    pos_arr = manager_dict['pos_arr']
    hpos_arr = manager_dict['hpos_arr']
    ppos_arr = manager_dict['ppos_arr']
    rew1_arr = manager_dict['rew1_arr']
    rew2_arr = manager_dict['rew2_arr']
    
    all_rewards = []
    lengths = []
    
    # Create a progress bar for each worker (with position to prevent overlap)
    progress_bar = tqdm(total=(iteration_end - iteration_start), desc=f"Core {int(iteration_start/100) + 1}", position=iteration_start, leave=True)
    
    for i in range(iteration_start, iteration_end):
        done = False
        obs = env.reset()
        
        # Reset arrays for this iteration
        action_arr[i] = np.array([0, 0, 0, 0, 0, 0])
        vel_arr[i] = np.array([0])
        pos_arr[i] = np.array([0])
        hpos_arr[i] = np.array([0, 0])
        ppos_arr[i] = np.array([0, 0])
        rew1_arr[i] = np.array([0])
        rew2_arr[i] = np.array([0])
        ep_rewards = []
        
        while not done:
            action, _ = model.predict(obs)
            action_arr[i] = np.vstack((action_arr[i], action))
            vel_arr[i] = np.vstack((vel_arr[i], (np.sqrt((env.get_obs_dict(env.sim)['hand_qvel'][0])**2 + (env.get_obs_dict(env.sim)['hand_qvel'][1])**2))))
            hpos_arr[i] = np.vstack((hpos_arr[i], [env.get_obs_dict(env.sim)['hand_qpos'][0], env.get_obs_dict(env.sim)['hand_qpos'][1]]))
            ppos_arr[i] = np.vstack((ppos_arr[i], [env.get_obs_dict(env.sim)['palm_pos'][0], env.get_obs_dict(env.sim)['palm_pos'][1]]))
            pos_arr[i] = np.vstack((pos_arr[i], np.linalg.norm(env.get_obs_dict(env.sim)['reach_err'], axis=-1)))
            obs, reward, done, info = env.step(action)
            reward_dict = env.rwd_dict
            rew1_arr[i] = np.vstack((rew1_arr[i], reward_dict['reach']))
            rew2_arr[i] = np.vstack((rew2_arr[i], reward_dict['act_reg']))
            ep_rewards.append(reward)
        
        # Trim the arrays and calculate the rewards
        hpos_arr[i] = hpos_arr[i][1:len(hpos_arr[i])]
        vel_arr[i] = vel_arr[i][1:len(vel_arr[i])]
        action_arr[i] = action_arr[i][1:len(action_arr[i])]
        pos_arr[i] = pos_arr[i][1:len(pos_arr[i])]
        
        all_rewards.append(np.sum(ep_rewards))
        lengths.append(len(action_arr[i]))
        
        # Update progress bar after each iteration
        progress_bar.update(1)
    
    # Close the progress bar once done
    progress_bar.close()
    
    # Save results to disk incrementally to reduce memory usage
    worker_id = f"{iteration_start}-{iteration_end}"
    np.savez_compressed(
        os.path.join(output_dir, f"results_{worker_id}.npz"),
        action_arr=action_arr,
        vel_arr=vel_arr,
        pos_arr=pos_arr,
        hpos_arr=hpos_arr,
        ppos_arr=ppos_arr,
        rew1_arr=rew1_arr,
        rew2_arr=rew2_arr,
        all_rewards=all_rewards,
        lengths=lengths
    )

    
    return all_rewards, lengths

def run_parallel_evaluation(total_iterations, num_cores):
    # Split iterations for each core
    iterations_per_core = total_iterations // num_cores
    ranges = [(i * iterations_per_core, (i + 1) * iterations_per_core) for i in range(num_cores)]
    
    # Handle last core to pick up the remainder of iterations
    ranges[-1] = (ranges[-1][0], total_iterations)
    
    # Create a Manager to store shared data structures
    with Manager() as manager:
        # Create shared memory data structures using Manager
        manager_dict = {
            'action_arr': manager.list([np.array([0, 0, 0, 0, 0, 0]) for _ in range(total_iterations)]),
            'vel_arr': manager.list([np.array([0]) for _ in range(total_iterations)]),
            'pos_arr': manager.list([np.array([0]) for _ in range(total_iterations)]),
            'hpos_arr': manager.list([np.array([0, 0]) for _ in range(total_iterations)]),
            'ppos_arr': manager.list([np.array([0, 0]) for _ in range(total_iterations)]),
            'rew1_arr': manager.list([np.array([0]) for _ in range(total_iterations)]),
            'rew2_arr': manager.list([np.array([0]) for _ in range(total_iterations)]),
        }
        
        # Create a multiprocessing pool
        with Pool(processes=num_cores) as pool:
            # Pass the iteration ranges to the workers
            results = pool.starmap(evaluate_policy, [(start, end, total_iterations, manager_dict) for start, end in ranges])
    
    # Combine the results from each process
    all_rewards = []
    lengths = []
    for result in results:
        all_rewards.extend(result[0])
        lengths.extend(result[1])
    
    return all_rewards, lengths

if __name__ == '__main__':
    # Number of iterations and cores to use
    total_iterations = 1000
    num_cores = 2
    
    # Run parallel evaluation with progress bars
    all_rewards, lengths = run_parallel_evaluation(total_iterations, num_cores)
    
    # Optionally, display progress or any final results
    print("Evaluation complete.")
