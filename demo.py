#!/usr/bin/env python3
"""
Quick demonstration script for TRPO implementation.

This script demonstrates the key features of the TRPO project:
1. Environment setup and testing
2. Quick training demonstration
3. Model evaluation
4. Performance comparison

Usage:
    python demo.py [--env ENV_NAME] [--timesteps N] [--episodes N]
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import numpy as np
import matplotlib.pyplot as plt
from agents.trpo import TRPOAgent, TRPOConfig
from envs.environments import make_env, get_env_info
from utils.logging import ExperimentLogger


def demo_environment(env_name: str):
    """Demonstrate environment functionality."""
    print(f"\n🌍 Testing Environment: {env_name}")
    print("=" * 50)
    
    try:
        env = make_env(env_name)
        env_info = get_env_info(env)
        
        print(f"✅ Environment loaded successfully!")
        print(f"   Observation Space: {env_info['observation_space']}")
        print(f"   Action Space: {env_info['action_space']}")
        print(f"   Discrete Actions: {env_info['is_discrete']}")
        print(f"   Observation Dim: {env_info['obs_dim']}")
        print(f"   Action Dim: {env_info['action_dim']}")
        
        # Test a few steps
        obs, _ = env.reset()
        print(f"   Initial observation: {obs}")
        
        for step in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   Step {step+1}: action={action}, reward={reward:.3f}, done={terminated or truncated}")
            
            if terminated or truncated:
                break
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Error with {env_name}: {e}")
        return False


def demo_training(env_name: str, total_timesteps: int = 2000):
    """Demonstrate TRPO training."""
    print(f"\n🚀 Training TRPO Agent on {env_name}")
    print("=" * 50)
    
    # Create configuration
    config = TRPOConfig(
        env_name=env_name,
        total_timesteps=total_timesteps,
        batch_size=min(500, total_timesteps // 4),
        log_interval=5,
        log_dir=f"demo_logs_{env_name.lower().replace('-', '_')}"
    )
    
    print(f"Configuration:")
    print(f"   Total timesteps: {config.total_timesteps}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Max KL divergence: {config.max_kl_divergence}")
    print(f"   Log directory: {config.log_dir}")
    
    # Create agent and logger
    agent = TRPOAgent(config)
    logger = ExperimentLogger(config.log_dir, f"demo_{env_name}")
    
    # Convert config to dict and handle numpy types
    config_dict = {}
    for key, value in config.__dict__.items():
        if isinstance(value, np.integer):
            config_dict[key] = int(value)
        elif isinstance(value, np.floating):
            config_dict[key] = float(value)
        elif isinstance(value, np.ndarray):
            config_dict[key] = value.tolist()
        else:
            config_dict[key] = value
    
    logger.log_hyperparameters(config_dict)
    logger.log_environment_info(get_env_info(agent.env))
    
    # Training loop
    print(f"\n🎯 Starting training...")
    episode_rewards = []
    episode_lengths = []
    
    total_steps = 0
    episode_count = 0
    
    while total_steps < config.total_timesteps:
        # Collect trajectories
        trajectories = agent.collect_trajectories(config.batch_size)
        total_steps += len(trajectories['obs'])
        
        # Compute advantages and returns
        advantages, returns = agent.compute_advantages(
            trajectories['rewards'],
            trajectories['dones'],
            trajectories['obs']
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy
        policy_loss = agent.update_policy(
            trajectories['obs'],
            trajectories['actions'],
            advantages,
            trajectories['old_log_probs']
        )
        
        # Update value function
        value_loss = agent.update_value_function(trajectories['obs'], returns)
        
        # Track statistics
        episode_count += len(agent.episode_rewards)
        
        if agent.episode_rewards:
            episode_rewards.append(np.mean(agent.episode_rewards))
            episode_lengths.append(np.mean(agent.episode_lengths))
        
        # Log progress
        if episode_count % config.log_interval == 0:
            avg_reward = np.mean(agent.episode_rewards) if agent.episode_rewards else 0
            avg_length = np.mean(agent.episode_lengths) if agent.episode_lengths else 0
            
            metrics = {
                'episode_reward': avg_reward,
                'episode_length': avg_length,
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'total_steps': total_steps
            }
            
            logger.log_metrics(metrics, episode_count)
            
            print(f"Episode {episode_count}, Avg Reward: {avg_reward:.2f}, "
                  f"Avg Length: {avg_length:.2f}, Policy Loss: {policy_loss:.4f}, "
                  f"Value Loss: {value_loss:.4f}, Total Steps: {total_steps}")
    
    print(f"\n✅ Training completed!")
    
    # Save model
    model_path = f"{config.log_dir}/trained_model.pth"
    agent.save_model(model_path)
    print(f"💾 Model saved to: {model_path}")
    
    return agent, episode_rewards, episode_lengths


def demo_evaluation(agent, n_episodes: int = 10):
    """Demonstrate model evaluation."""
    print(f"\n🔍 Evaluating trained model ({n_episodes} episodes)")
    print("=" * 50)
    
    eval_stats = agent.evaluate(n_episodes)
    
    print(f"📊 Evaluation Results:")
    print(f"   Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
    print(f"   Mean Length: {eval_stats['mean_length']:.2f} ± {eval_stats['std_length']:.2f}")
    
    return eval_stats


def demo_comparison(env_name: str, trained_reward: float, n_episodes: int = 10):
    """Compare trained agent with random policy."""
    print(f"\n🎲 Comparing with random policy ({n_episodes} episodes)")
    print("=" * 50)
    
    env = make_env(env_name)
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
    
    env.close()
    
    random_mean = np.mean(episode_rewards)
    random_std = np.std(episode_rewards)
    
    print(f"📊 Random Policy Results:")
    print(f"   Mean Reward: {random_mean:.2f} ± {random_std:.2f}")
    
    improvement = trained_reward - random_mean
    print(f"\n🏆 TRPO vs Random Policy:")
    print(f"   Reward Improvement: {improvement:.2f}")
    print(f"   Improvement Factor: {trained_reward / random_mean:.2f}x" if random_mean > 0 else "   N/A (random policy got 0 reward)")


def plot_training_progress(episode_rewards, episode_lengths, save_path=None):
    """Plot training progress."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Episode rewards
    ax1.plot(episode_rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Update')
    ax1.set_ylabel('Average Reward')
    ax1.grid(True)
    
    # Episode lengths
    ax2.plot(episode_lengths)
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Update')
    ax2.set_ylabel('Average Length')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training plots saved to: {save_path}")
    
    plt.show()


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description='TRPO Demonstration')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                       help='Environment name (default: CartPole-v1)')
    parser.add_argument('--timesteps', type=int, default=2000,
                       help='Total training timesteps (default: 2000)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--plot', action='store_true',
                       help='Show training plots')
    
    args = parser.parse_args()
    
    print("🎉 TRPO Implementation Demo")
    print("=" * 50)
    print(f"Environment: {args.env}")
    print(f"Training timesteps: {args.timesteps}")
    print(f"Evaluation episodes: {args.episodes}")
    
    # 1. Test environment
    if not demo_environment(args.env):
        print("❌ Environment test failed. Exiting.")
        return
    
    # 2. Train agent
    try:
        agent, episode_rewards, episode_lengths = demo_training(args.env, args.timesteps)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # 3. Evaluate agent
    try:
        eval_stats = demo_evaluation(agent, args.episodes)
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return
    
    # 4. Compare with random policy
    try:
        demo_comparison(args.env, eval_stats['mean_reward'], args.episodes)
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return
    
    # 5. Plot training progress
    if args.plot and episode_rewards:
        plot_training_progress(
            episode_rewards, 
            episode_lengths,
            save_path=f"demo_logs_{args.env.lower().replace('-', '_')}/training_progress.png"
        )
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"📁 Check the 'demo_logs_{args.env.lower().replace('-', '_')}' directory for results.")


if __name__ == "__main__":
    main()
