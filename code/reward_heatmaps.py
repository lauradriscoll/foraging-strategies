##WARNING! This was written mostly by an LLM :)

import numpy as np
import matplotlib.pyplot as plt
from tools import PatchForager

def create_combined_heatmaps(forager, drift_range=(0, 1.5), push_range=(-1.5, 0), 
                            n_points=10, n_patches=40, n_runs=5, max_time_per_patch=50):
    """
    Create heatmaps for reward rate AND rewards per patch type.
    
    Returns:
        drift_values, push_values, reward_rate_matrix, rewards_patch0_matrix, rewards_patch1_matrix
    """
    
    # Create parameter grids
    drift_values = np.linspace(drift_range[0], drift_range[1], n_points)
    push_values = np.linspace(push_range[0], push_range[1], n_points)
    
    # Initialize matrices
    reward_rate_matrix = np.zeros((n_points, n_points))
    rewards_patch0_matrix = np.zeros((n_points, n_points))
    rewards_patch1_matrix = np.zeros((n_points, n_points))
    
    # Fixed parameters
    threshold = 1.0
    initial_state = 0.0
    noise_std = 0.0
    
    print(f"Computing combined heatmaps for {n_points}x{n_points} = {n_points**2} parameter combinations...")
    
    total_combinations = n_points * n_points
    completed = 0
    
    for i, drift_rate in enumerate(drift_values):
        for j, reward_push in enumerate(push_values):
            
            # Progress update
            completed += 1
            if completed % 25 == 0 or completed == total_combinations:
                print(f"Progress: {completed}/{total_combinations} ({100*completed/total_combinations:.1f}%)")
            
            # Track metrics across runs
            reward_rates = []
            patch0_rewards = []
            patch1_rewards = []
            
            for _ in range(n_runs):
                try:
                    # Generate random patch sequence
                    patch_sequence = np.random.choice([0, 1], size=n_patches).tolist()
                    
                    data, reward_rate = forager.run_simulation(
                        'ddm', patch_sequence,
                        threshold=threshold,
                        drift_rate=drift_rate,
                        reward_push=reward_push,
                        initial_state=initial_state,
                        noise_std=noise_std,
                        max_time_per_patch=max_time_per_patch
                    )
                    
                    reward_rates.append(reward_rate)
                    
                    # Extract rewards per patch type
                    patch_data = data[data['patch_id'] != -1]  # Remove travel rows
                    
                    # Count total rewards for each patch type
                    patch0_data = patch_data[patch_data['patch_id'] == 0]
                    patch1_data = patch_data[patch_data['patch_id'] == 1]
                    
                    # Sum rewards for each patch type
                    patch0_total = patch0_data['reward'].sum() if len(patch0_data) > 0 else 0
                    patch1_total = patch1_data['reward'].sum() if len(patch1_data) > 0 else 0
                    
                    # Count number of patches of each type visited
                    if len(patch0_data) > 0:
                        n_patch0_visits = len(patch0_data['patch_entry_time'].unique())
                    else:
                        n_patch0_visits = 1
                        
                    if len(patch1_data) > 0:
                        n_patch1_visits = len(patch1_data['patch_entry_time'].unique())
                    else:
                        n_patch1_visits = 1
                    
                    # Average rewards per patch visit
                    patch0_rewards.append(patch0_total / max(n_patch0_visits, 1))
                    patch1_rewards.append(patch1_total / max(n_patch1_visits, 1))
                    
                except Exception as e:
                    # Handle failed simulations
                    reward_rates.append(0.0)
                    patch0_rewards.append(0.0)
                    patch1_rewards.append(0.0)
            
            # Store averages
            reward_rate_matrix[i, j] = np.mean(reward_rates)
            rewards_patch0_matrix[i, j] = np.mean(patch0_rewards)
            rewards_patch1_matrix[i, j] = np.mean(patch1_rewards)
    
    return drift_values, push_values, reward_rate_matrix, rewards_patch0_matrix, rewards_patch1_matrix

def plot_combined_heatmaps(drift_values, push_values, reward_rate_matrix, 
                          rewards_patch0_matrix, rewards_patch1_matrix, optimal_params=None):
    """
    Plot combined heatmaps: reward rate + rewards per patch type side by side.
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Overall reward rate heatmap
    im1 = axes[0].imshow(reward_rate_matrix, 
                        extent=[push_values[0], push_values[-1], 
                               drift_values[0], drift_values[-1]],
                        aspect='auto', 
                        origin='lower',
                        cmap='viridis',
                        interpolation='bilinear')
    
    axes[0].set_xlabel('Reward Push')
    axes[0].set_ylabel('Drift Rate')
    axes[0].set_title('Overall Reward Rate')
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Reward Rate')
    
    # Add contours
    X, Y = np.meshgrid(push_values, drift_values)
    contours1 = axes[0].contour(X, Y, reward_rate_matrix, levels=10, colors='white', alpha=0.3, linewidths=0.5)
    
    # 2. Patch 0 rewards heatmap
    im2 = axes[1].imshow(rewards_patch0_matrix, 
                        extent=[push_values[0], push_values[-1], 
                               drift_values[0], drift_values[-1]],
                        aspect='auto', 
                        origin='lower',
                        cmap='viridis',
                        interpolation='bilinear')
    
    axes[1].set_xlabel('Reward Push')
    axes[1].set_ylabel('Drift Rate')
    axes[1].set_title('Rewards per Patch 0 Visit')
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Rewards per Visit')
    
    # Add contours
    contours2 = axes[1].contour(X, Y, rewards_patch0_matrix, levels=8, colors='white', alpha=0.4, linewidths=0.5)
    
    # 3. Patch 1 rewards heatmap
    im3 = axes[2].imshow(rewards_patch1_matrix, 
                        extent=[push_values[0], push_values[-1], 
                               drift_values[0], drift_values[-1]],
                        aspect='auto', 
                        origin='lower',
                        cmap='viridis',
                        interpolation='bilinear')
    
    axes[2].set_xlabel('Reward Push')
    axes[2].set_ylabel('Drift Rate')
    axes[2].set_title('Rewards per Patch 1 Visit')
    cbar3 = plt.colorbar(im3, ax=axes[2])
    cbar3.set_label('Rewards per Visit')
    
    # Add contours
    contours3 = axes[2].contour(X, Y, rewards_patch1_matrix, levels=8, colors='white', alpha=0.4, linewidths=0.5)
    
    # Mark optimal point on all plots if provided
    if optimal_params is not None:
        opt_drift = optimal_params['drift_rate']
        opt_push = optimal_params['reward_push']
        for ax in axes:
            ax.plot(opt_push, opt_drift, 'r*', markersize=15, label=f'Optimal')
        axes[0].legend()
    
    # Add grids
    for ax in axes:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def analyze_combined_results(drift_values, push_values, reward_rate_matrix, 
                           rewards_patch0_matrix, rewards_patch1_matrix):
    """
    Analyze the combined results to understand relationships.
    """
    
    # Find global maximum for reward rate
    max_idx = np.unravel_index(np.argmax(reward_rate_matrix), reward_rate_matrix.shape)
    max_drift = drift_values[max_idx[0]]
    max_push = push_values[max_idx[1]]
    max_reward_rate = reward_rate_matrix[max_idx]
    
    # At the optimal point, what are the patch rewards?
    opt_patch0_rewards = rewards_patch0_matrix[max_idx]
    opt_patch1_rewards = rewards_patch1_matrix[max_idx]
    
    print("COMBINED ANALYSIS:")
    print("-" * 50)
    print(f"Optimal parameters: drift={max_drift:.4f}, push={max_push:.4f}")
    print(f"Optimal reward rate: {max_reward_rate:.4f}")
    print(f"At optimal point:")
    print(f"  Patch 0 rewards per visit: {opt_patch0_rewards:.4f}")
    print(f"  Patch 1 rewards per visit: {opt_patch1_rewards:.4f}")
    print(f"  Ratio (P0/P1): {opt_patch0_rewards/max(opt_patch1_rewards, 0.001):.4f}")
    
    # Find maxima for each patch type separately
    max_idx_0 = np.unravel_index(np.argmax(rewards_patch0_matrix), rewards_patch0_matrix.shape)
    max_idx_1 = np.unravel_index(np.argmax(rewards_patch1_matrix), rewards_patch1_matrix.shape)
    
    print(f"\nPatch-specific optima:")
    print(f"Patch 0 max: drift={drift_values[max_idx_0[0]]:.4f}, push={push_values[max_idx_0[1]]:.4f}, rewards={rewards_patch0_matrix[max_idx_0]:.4f}")
    print(f"Patch 1 max: drift={drift_values[max_idx_1[0]]:.4f}, push={push_values[max_idx_1[1]]:.4f}, rewards={rewards_patch1_matrix[max_idx_1]:.4f}")
    
    return {
        'optimal_drift': max_drift,
        'optimal_push': max_push,
        'optimal_reward_rate': max_reward_rate,
        'optimal_patch0_rewards': opt_patch0_rewards,
        'optimal_patch1_rewards': opt_patch1_rewards
    }

# Main execution
if __name__ == "__main__":
    # Create forager
    forager = PatchForager(
        travel_time = 3,
        reward_value = [5, 5],
        a = [0.9, 0.6],
        b = [2.76, 2.76],
        c = [0.1278, 0.1278],
        d = [0, 0],
        prob=True,
        depl_fxn='exp'
    )
    
    print("="*60)
    print("COMBINED HEATMAPS: REWARD RATE + REWARDS PER PATCH")
    print("="*60)
    
    # Generate combined heatmap data
    drift_vals, push_vals, reward_rate_matrix, patch0_rewards, patch1_rewards = create_combined_heatmaps(
        forager,
        drift_range=(.05, 1.5),
        push_range=(-1.5, -.05), 
        n_points=25,
        n_patches=100,
        n_runs=510
    )
    
    # Analyze the results
    analysis = analyze_combined_results(drift_vals, push_vals, reward_rate_matrix, 
                                      patch0_rewards, patch1_rewards)
    
    # Create optimal parameters dict for plotting
    optimal_params = {
        'drift_rate': analysis['optimal_drift'],
        'reward_push': analysis['optimal_push']
    }
    
    # Plot combined heatmaps
    fig = plot_combined_heatmaps(drift_vals, push_vals, reward_rate_matrix, 
                                patch0_rewards, patch1_rewards, optimal_params)
    
    # Save the plot
    plt.savefig('combined_ddm_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save data
    np.savez('combined_heatmap_data.npz', 
             drift_values=drift_vals,
             push_values=push_vals, 
             reward_rate_matrix=reward_rate_matrix,
             patch0_rewards=patch0_rewards,
             patch1_rewards=patch1_rewards,
             analysis=analysis)
    
    print(f"\nCombined heatmaps saved as 'combined_ddm_heatmaps.png'")
    print(f"Data saved as 'combined_heatmap_data.npz'")