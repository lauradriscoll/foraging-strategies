import numpy as np
import scipy
import random


def gen_bern(probability):
    """
    Generate a random value of 1 or 0 based on the given probability.
    
    Args:
        probability (float): The probability of generating 1.
        
    Returns:
        int: Either 1 or 0, with the specified probability.
    """
    if probability < 0 or probability > 1:
        raise ValueError("Probability must be between 0 and 1.")
    
    return 1 if random.random() < p else 0

# Depletion functions for each patch
def depletion_func(a, b, c, d, t):
    return a * (b ** (-c*t)+d)
    
def calc_total_reward_rate(patch_list, travel_time, max_time, reward_value, a, b, c, d, prob = False):
    """
    Calculate the total reward rate for a discrete-time foraging system.

    Args:
        patch_list (int): List of patch types.
        travel_time (int): Time required to travel between patches.
        max_time (list): Maximum time to forage in a patch.
        reward_value (array): value for each patch
        prob (boolean): whether rewards are delivered probabilistically

    Returns:
        float: Total reward rate for the entire environment.
    """
    total_reward = 0
    total_time = 0

    for patch_id in patch_list:
        patch_reward = 0
        patch_time = max_time[patch_id]

        for t in range(patch_time):
            prob_reward = depletion_func(a[patch_id],b[patch_id],c[patch_id],d[patch_id],t)
            if prob:
                instantaneous_rate =  gen_bern(prob_reward) * reward_value[patch_id]
            else:
                instantaneous_rate =  prob_reward * reward_value[patch_id]
            patch_reward += instantaneous_rate

        total_reward += patch_reward
        total_time += max_time[patch_id] + travel_time

    total_reward_rate = total_reward / total_time
    return total_reward_rate

def run_simulation(forager, strategy, patch_list, **strategy_params):
    data = []
    total_time = 0
    total_reward = 0
    patch_entry_time = 0
    
    for patch_id in patch_list:
        t_in_patch = 0
        patch_reward = 0 #value
        rewards_in_patch = 0 #instances
        failures_in_patch = 0 #instances
        
        while True:
            prob_reward = forager.depletion_func(patch_id, t_in_patch)
            if forager.prob:
                reward = forager.gen_bern(prob_reward) * forager.reward_value[patch_id]
            else:
                reward = prob_reward * forager.reward_value[patch_id]
            
            patch_reward += reward
            total_time += 1
            t_in_patch += 1
            
            if reward > 0:
                rewards_in_patch += 1
            else:
                failures_in_patch += 1
            
            data.append({
                'time': total_time,
                'patch_id': patch_id,
                'time_in_patch': t_in_patch,
                'reward': reward,
                'cumulative_patch_reward': patch_reward,
                'prob_reward': prob_reward,
                'rewards_in_patch': rewards_in_patch,
                'failures_in_patch': failures_in_patch,
                'patch_entry_time': patch_entry_time
            })


            # print(strategy_params)
            
            # Check exit condition based on strategy
            if strategy == 'target_stops':
                if t_in_patch >= strategy_params['target_stops'][patch_id]:
                    break
            if strategy == 'mvt_rate':
                current_rate = patch_reward / (t_in_patch+1 + forager.travel_time) #adding +1 for future patch threshold
                if current_rate < strategy_params['target_reward_rate']:     
                    break
            elif strategy == 'fixed_rewards':
                if rewards_in_patch >= strategy_params['target_rewards']:
                    break
            elif strategy == 'fixed_failures':
                if failures_in_patch > strategy_params['max_failures']:
                    break
        
        # Add travel time
        total_time += forager.travel_time
        data.append({
            'time': total_time,
            'patch_id': -1,  # -1 indicates traveling
            'time_in_patch': forager.travel_time,
            'reward': 0,
            'cumulative_patch_reward': 0,
            'prob_reward': 0,
            'rewards_in_patch': 0,
            'failures_in_patch': 0,
            'patch_entry_time': None
        })
        
        patch_entry_time = total_time
        total_reward += patch_reward

    return pd.DataFrame(data), total_reward/total_time

def calculate_optimal_stops(patch_list, travel_time, reward_value, a, b, c, d, max_stops=20):
    grid = np.zeros((max_stops, max_stops))
    for x in range(max_stops):
        for y in range(max_stops):
            total_reward_rate = calc_total_reward_rate(patch_list, travel_time, [x, y], reward_value, a, b, c, d)
            grid[x, y] = total_reward_rate

    best_time = np.unravel_index(grid.argmax(), grid.shape)
    max_reward_rate = grid[best_time]

    return {
        'optimal_stops': best_time,
        'max_reward_rate': max_reward_rate,
        'reward_rate_grid': grid
    }

def moving_window_avg(data, window_size):
    """
    Calculates the moving window average of a list of ones and zeros.
    
    Args:
        data (list): A list of ones and zeros.
        window_size (int): The size of the window for the moving average.
        
    Returns:
        list: A list containing the moving window averages.
    """
    window_avgs = []
    
    for i in range(0,len(data)+ 1):
        if i < window_size:
            window_avgs.append(np.nan)
        else:
            window = data[i-window_size:i]
            window_avg = sum(window) / window_size
            window_avgs.append(window_avg)
        
    return window_avgs