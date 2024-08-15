import numpy as np
import scipy
import random
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

class PatchForager:
    def __init__(self, travel_time, reward_value, a, b, c, d, prob=False, depl_fxn = 'fixed', indep_var = 'rewards'):
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
            raise('depl_fxn not defined')
        self.prob = prob #prob (boolean): whether rewards are delivered probabilistically
        self.indep_var = indep_var

    def depletion_func(self, patch_id, stops, rewards):

        if self.indep_var == 'stops':
            t = stops
        elif self.indep_var == 'rewards':
            t = rewards 
        else:
            raise Exception('Independent variable not specified.')
        
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
        
    def calculate_optimal_stops(self, patch_list, max_stops=20):
        # Get the number of unique patches
        num_patches = len(set(patch_list))
        
        # Create a list of all possible combinations of stops
        stop_combinations = list(product(range(max_stops), repeat=num_patches))

        # Initialize the results dictionary
        results = {combo: 0 for combo in stop_combinations}
        
        # Calculate reward rate for each combination
        for combo in stop_combinations:
            _, total_reward_rate = self.run_simulation('stops', patch_list, target_stops=list(combo))
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
            'optimal_stops': best_combo,
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

                # Check exit condition based on strategy
                if strategy == 'patch_type':
                    strategy_patch = list(strategy_params['target_patches'][patch_id].keys())[0]
                    if strategy_patch == 'target_stops':
                        if t_in_patch >= strategy_params['target_patches'][patch_id]['target_stops']:
                            break
                    elif strategy_patch == 'target_rewards':
                        if rewards_in_patch >= strategy_params['target_patches'][patch_id]['target_rewards']:
                            break
                        if t_in_patch >150:
                            break
                    elif strategy_patch == 'consec_failures':
                        if consec_failures >= strategy_params['target_patches'][patch_id]['consec_failures']:
                            break
                
                if strategy == 'stops':
                    if t_in_patch >= strategy_params['target_stops'][patch_id]:
                        break
                # elif strategy == 'rate':
                #     if t_in_patch==0:
                #         current_rate = 0
                #     else:
                #         current_rate = patch_reward / (t_in_patch)
                #     if current_rate <= strategy_params['target_reward_rate']:     
                #         break
                elif strategy == 'rewards':
                    if rewards_in_patch >= strategy_params['target_rewards'][patch_id]:
                        break
                    if t_in_patch >150:
                        break
                elif strategy == 'consec_failures':
                    if consec_failures >= strategy_params['consec_failures'][patch_id]:
                        break
                elif strategy == 'failures':
                    if failures_in_patch >= strategy_params['max_failures'][patch_id]:
                        break

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
                if strategy == 'stops':
                    if t_in_patch >= strategy_params['target_stops'][patch_id]:
                        break
                if strategy == 'rate':
                    current_rate = patch_reward / (t_in_patch)
                    # print(current_rate)
                    if current_rate <= strategy_params['target_reward_rate']:     
                        break
                elif strategy == 'rewards':
                    if rewards_in_patch >= strategy_params['target_rewards'][patch_id]:
                        break
                    if t_in_patch >150:
                        break
                elif strategy == 'consec_failures':
                    if consec_failures >= strategy_params['consec_failures'][patch_id]:
                        break
                elif strategy == 'failures':
                    if failures_in_patch >= strategy_params['max_failures'][patch_id]:
                        break
                    
                elif strategy == 'patch_type':
                    strategy_patch = list(strategy_params['target_patches'][patch_id].keys())[0]
                    if strategy_patch == 'target_stops':
                        if t_in_patch >= strategy_params['target_patches'][patch_id]['target_stops']:
                            break
                    elif strategy_patch == 'rate':
                        current_rate = patch_reward / (t_in_patch)
                        if current_rate >= strategy_params['target_patches'][patch_id]['target_reward_rate']:
                            break
                    elif strategy_patch == 'target_rewards':
                        if rewards_in_patch >= strategy_params['target_patches'][patch_id]['target_rewards']:
                            break
                        if t_in_patch >150:
                            break
                    elif strategy_patch == 'consec_failures':
                        if consec_failures >= strategy_params['target_patches'][patch_id]['consec_failures']:
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
        plt.figure(figsize=(2, 2))
        ax = plt.subplot(111)

        # Generate a color map with unique colors for each patch
        color_map = plt.get_cmap('tab20')  # You can choose a different colormap if needed
        colors = color_map(np.linspace(0, 1, len(patch_list)))

        for patch_id in patch_list: 

            # Get the color for this patch
            color = colors[patch_id]
            
            # Compute the depletion rate for each time step
            p_R = [self.depletion_func(patch_id,t) for t in time_steps]
        
            plt.plot(time_steps+1, p_R,color=color)#label = str(p_R[best_time[patch_id]-1]), 
            # plt.plot(best_time[patch_id], p_R[best_time[patch_id]-1],'o', color=color,
            #  label=f'Patch {patch_id}: {best_time[patch_id]} rewards')
        
        plt.xlabel('# Rewards')
        plt.ylabel('Probability of Reward')
        plt.legend(loc='center right', bbox_to_anchor=(1, 1.15))
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(f'figs/mvt_curves'+str(self.travel_time)+'.png', bbox_inches='tight', dpi=300)
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