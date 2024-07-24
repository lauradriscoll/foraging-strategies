import numpy as np
import scipy
import random
import pandas as pd
import matplotlib.pyplot as plt

class PatchForager:
    def __init__(self, travel_time, reward_value, a, b, c, d, prob=False):
        self.travel_time = travel_time #travel_time (int): Time required to travel between patches.
        self.reward_value = reward_value #reward_value (array): value for each patch
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.prob = prob #prob (boolean): whether rewards are delivered probabilistically

    def depletion_func(self, patch_id, t):
        return self.a[patch_id] * (self.b[patch_id] ** (-self.c[patch_id]*t)+self.d[patch_id])

    def gen_bern(self, p):
        return 1 if random.random() < p else 0
        
    def calc_total_reward_rate(self, patch_list, max_time):
        """
        Calculate the total reward rate for a discrete-time foraging system.
    
        Args:
            patch_list (int): List of patch types.
            max_time (list): Maximum time to forage in a patch.
            
        Returns:
            float: Total reward rate for the entire environment.
        """
        
        total_reward = 0
        total_time = 0
    
        for patch_id in patch_list:
            patch_reward = 0
            patch_time = max_time[patch_id]
    
            for t in range(patch_time):
                prob_reward = self.depletion_func(patch_id, patch_reward)
                
                if self.prob:
                    instantaneous_rate =  self.gen_bern(prob_reward) * self.reward_value[patch_id]
                else:
                    instantaneous_rate =  prob_reward * self.reward_value[patch_id]
                patch_reward += instantaneous_rate
    
            total_reward += patch_reward
            total_time += max_time[patch_id] + self.travel_time
    
        total_reward_rate = total_reward / total_time
        return total_reward_rate


    def calculate_optimal_stops(self, patch_list, max_stops=20):
        
        grid = np.zeros((max_stops, max_stops))
        
        for x in range(max_stops):
            for y in range(max_stops):
                
                _, total_reward_rate = self.run_simulation('target_stops', patch_list, target_stops = [x,y])
                grid[x, y] = total_reward_rate
    
        best_time = np.unravel_index(grid.argmax(), grid.shape)
        max_reward_rate = grid[best_time]
    
        return {
            'optimal_stops': best_time,
            'max_reward_rate': max_reward_rate,
            'reward_rate_grid': grid
        }

    def run_simulation(self, strategy, patch_list, **strategy_params):
        data = []
        total_time = 0
        patch_entry_time = 0
        
        for patch_id in patch_list:
            t_in_patch = 0
            patch_reward = 0 #value
            rewards_in_patch = 0 #instances
            failures_in_patch = 0 #instances
            consec_failures = 0
            
            while True:
                prob_reward = self.depletion_func(patch_id, rewards_in_patch)
                if self.prob:
                    reward = self.gen_bern(prob_reward) * self.reward_value[patch_id]
                else:
                    reward = prob_reward * self.reward_value[patch_id]
                
                patch_reward += reward
                total_time += 1
                t_in_patch += 1
                
                if reward > 0:
                    rewards_in_patch += 1
                    consec_failures = 0
                else:
                    failures_in_patch += 1
                    consec_failures += 1
                
                data.append({
                    'time': total_time,
                    'patch_id': patch_id,
                    'time_in_patch': t_in_patch,
                    'reward': reward,
                    'cumulative_patch_reward': patch_reward,
                    'prob_reward': prob_reward,
                    'rewards_in_patch': rewards_in_patch,
                    'failures_in_patch': failures_in_patch,
                    'consecutive_failures': consec_failures,
                    'patch_entry_time': patch_entry_time
                })
    
                # Check exit condition based on strategy
                if strategy == 'target_stops':
                    if t_in_patch >= strategy_params['target_stops'][patch_id]:
                        break
                if strategy == 'mvt_rate':
                    current_rate = patch_reward / (t_in_patch)
                    # print(current_rate)
                    if current_rate <= strategy_params['target_reward_rate']:     
                        break
                elif strategy == 'fixed_rewards':
                    if rewards_in_patch >= strategy_params['target_rewards'][patch_id]:
                        break
                elif strategy == 'fixed_consec_failures':
                    if consec_failures >= strategy_params['consec_failures']:
                        break
                elif strategy == 'fixed_failures':
                    if failures_in_patch > strategy_params['max_failures']:
                        break
            
            # Add travel time
            total_time += self.travel_time
            data.append({
                'time': total_time,
                'patch_id': -1,  # -1 indicates traveling
                'time_in_patch': self.travel_time,
                'reward': 0,
                'cumulative_patch_reward': 0,
                'prob_reward': 0,
                'rewards_in_patch': 0,
                'failures_in_patch': 0,
                'consecutive_failures': 0,
                'patch_entry_time': None
            })
            
            patch_entry_time = total_time

        # Convert data to a DataFrame
        data_df = pd.DataFrame(data)
        total_reward_rate = data_df['reward'].cumsum().iloc[-1] / data_df['time'].iloc[-1]
    
        return data_df, total_reward_rate

    # Create a range of time steps
    def make_pr_plot(self, patch_list, best_time, max_stops=20):
        time_steps = np.arange(max_stops)
        
        # Plot the depletion rate over time
        plt.figure(figsize=(3, 3))
        
        for patch_id in patch_list: 
            
            # Compute the depletion rate for each time step
            p_R = [self.depletion_func(patch_id,t) for t in time_steps]
        
            plt.plot(time_steps+1, p_R,label = str(p_R[best_time[patch_id]-1]))
            plt.plot(best_time[patch_id], p_R[best_time[patch_id]-1],'ok')
        
        plt.xlabel('# Rewards')
        plt.ylabel('P(R)')
        plt.legend()
        plt.show()

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