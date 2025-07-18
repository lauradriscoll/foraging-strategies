import numpy as np
import scipy
import random
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

class PatchForager:
    def __init__(self, travel_time, reward_value, a, b, c, d, prob=False, depl_fxn = 'exp', indep_var = 'rewards'):
        self.travel_time = travel_time #travel_time (int): Time required to travel between patches.
        self.reward_value = reward_value #reward_value (array): value for each patch
        self.depl_fxn = depl_fxn
        if self.depl_fxn=='exp':
            self.a = a
            self.b = b
            self.c = c
            self.d = d
        elif self.depl_fxn=='fixed':
            self.a = a #depletion fxn for a fixed number of rewards
            self.b = b #index of reward start
            self.c = c #index of reward end
        else:
            raise("depl_fxn not defined")
        self.prob = prob #prob (boolean): whether rewards are delivered probabilistically
        self.indep_var = indep_var

    def depletion_func(self, patch_id, stops, rewards):
        if self.indep_var == 'stops':
            t = stops
        elif self.indep_var == 'rewards':
            t = rewards 
        else:
            if self.indep_var[patch_id] == 'rewards':
                t = rewards
            elif self.indep_var[patch_id] == 'stops':
                t = stops
        
        if self.depl_fxn == 'exp': 
            rate = self.a[patch_id] * (self.b[patch_id] ** (-self.c[patch_id]*t)+self.d[patch_id])
            
        elif self.depl_fxn == 'fixed':
            if stops < self.b[patch_id]: #This should always be based on stops or else you'd never get reward.
                rate = 0
            elif t > self.c[patch_id]: #This can be based on stops or rewards
                rate = 0
            else:
                rate = self.a[patch_id,t] #This can be based on stops or rewards
                
        return rate

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
                prob_reward = self.depletion_func(patch_id, t)
                
                if self.prob:
                    instantaneous_rate =  self.gen_bern(prob_reward) * self.reward_value[patch_id]
                else:
                    instantaneous_rate =  prob_reward * self.reward_value[patch_id]
                patch_reward += instantaneous_rate
    
            total_reward += patch_reward
            total_time += max_time[patch_id] + self.travel_time
    
        total_reward_rate = total_reward / total_time
        return total_reward_rate

    def calculate_optimal(self, patch_list, max_stops=20, indep_var = 'rewards'):
        # Get the number of unique patches
        num_patches = len(set(patch_list))
        
        # Create a list of all possible combinations of stops
        stop_combinations = list(product(range(max_stops), repeat=num_patches))
        
        # Initialize the results dictionary
        results = {combo: 0 for combo in stop_combinations}
        
        # Calculate reward rate for each combination
        for combo in stop_combinations:
            _, total_reward_rate = self.run_simulation(indep_var, patch_list, target_rewards=list(combo))
            results[combo] = total_reward_rate
        
        # Find the best combination
        best_combo = max(results, key=results.get)
        max_reward_rate = results[best_combo]
        
        # Create a multi-dimensional grid for visualization
        grid_shape = (max_stops,) * num_patches
        grid = np.zeros(grid_shape)
        for combo, rate in results.items():
            grid[combo] = rate
        
        return {
            'optimal': best_combo,
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
            ddm_state = np.nan  # Initialize for all strategies

            # Initialize DDM state if using DDM strategy
            if strategy == 'ddm':
                ddm_state = strategy_params.get('initial_state', 0.0)
                threshold = strategy_params.get('threshold', 1.0)
                drift_rate = strategy_params.get('drift_rate', .29)  # drift toward leaving
                reward_push = strategy_params.get('reward_push', -.17)  # how much reward pushes relative to threshold
                noise_std = strategy_params.get('noise_std', 0)  # optional noise
            
            while True:

                prob_reward = self.depletion_func(patch_id, t_in_patch, rewards_in_patch)
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

                # Update DDM state if using DDM strategy
                if strategy == 'ddm':
                    # Drift toward leaving threshold
                    ddm_state += drift_rate
                    
                    # Movement relative to threshold when reward is obtained
                    if reward > 0:
                        ddm_state += reward_push
                    
                    # Optional: add noise
                    if noise_std > 0:
                        ddm_state += np.random.normal(0, noise_std)
                
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
                    'patch_entry_time': patch_entry_time,
                    'ddm_state': ddm_state
                })
    
                # Check exit condition based on strategy
                if strategy == 'stops':
                    if t_in_patch >= strategy_params['target_stops'][patch_id]:
                        break
                if strategy == 'rate':
                    current_rate = patch_reward / (t_in_patch)
                    # print(current_rate)
                    if current_rate <= strategy_params['target_reward_rate'][patch_id]:     
                        break
                elif strategy == 'rewards':
                    if rewards_in_patch >= strategy_params['target_rewards'][patch_id]:
                        break
                elif strategy == 'consec_failures':
                    if consec_failures >= strategy_params['consec_failures'][patch_id]:
                        break
                elif strategy == 'failures':
                    if failures_in_patch > strategy_params['max_failures'][patch_id]:
                        break
                elif strategy == 'ddm':
                    if ddm_state >= threshold:
                        break
                if strategy_params.get('max_time_per_patch') and t_in_patch >= strategy_params['max_time_per_patch']:
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
                'patch_entry_time': np.nan,
                'ddm_state': np.nan
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
        ax = plt.subplot(111)

        # Get unique patches only
        unique_patches = list(set(patch_list))
        color_map = plt.get_cmap('tab10')
        
        for i, patch_id in enumerate(unique_patches): 
            color = color_map(i)
            
            # Compute the depletion rate for each time step
            p_R = []
            for t in time_steps:
                try:
                    # Fix: pass stops and rewards correctly
                    prob = self.depletion_func(patch_id, t, t)  # assuming stops=rewards=t
                    p_R.append(prob)
                except:
                    p_R.append(0)
        
            plt.plot(time_steps, p_R, color=color, label=f'Patch {patch_id}', linewidth=2)
            
            # Mark optimal point if provided
            if best_time and patch_id < len(best_time):
                opt_point = best_time[patch_id]
                if opt_point < len(p_R):
                    plt.plot(opt_point, p_R[opt_point], 'o', color=color, 
                            markersize=8, markerfacecolor='white', markeredgewidth=2)
        
        plt.xlabel(f'# {self.indep_var.title()}')
        plt.ylabel('Probability of Reward')
        plt.title('Patch Depletion Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        ax.spines[['right', 'top']].set_visible(False)
        plt.tight_layout()
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