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
    
    return random.randint(0, 1) if random.random() < probability else 0

# Depletion functions for each patch
def depletion_func(a, b, c, d, t):
    return a * (b ** (-c*t)+d)
    
def calc_total_reward_rate(patch_list, travel_time, max_time, reward_value, a, b, c, d):
    """
    Calculate the total reward rate for a discrete-time foraging system.

    Args:
        num_patches (int): Number of patches in the environment.
        travel_time (int): Time required to travel between patches.
        max_time (list): Maximum time to forage in a patch.
        reward_value (array): value for each patch

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
            instantaneous_rate =  prob_reward * reward_value[patch_id] # probabilistic version: gen_bern(prob_reward) * reward_value
            patch_reward += instantaneous_rate

        total_reward += patch_reward
        total_time += max_time[patch_id] + travel_time

    total_reward_rate = total_reward / total_time
    return total_reward_rate

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