"""
Command-line interface and visualization tools for TRPO training.

This module provides CLI tools for training, evaluating, and visualizing
TRPO agents with various environments.
"""

import argparse
import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agents.trpo import TRPOAgent, TRPOConfig
from envs.environments import make_env, get_env_info


class TrainingVisualizer:
    """Visualization tools for training progress."""
    
    def __init__(self, save_dir: str = "logs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_curves(self, stats: Dict[str, List[float]], 
                           save_path: Optional[str] = None) -> None:
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('TRPO Training Progress', fontsize=16)
        
        # Episode rewards
        axes[0, 0].plot(stats['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].grid(True)
        
        # Episode lengths
        axes[0, 1].plot(stats['episode_lengths'])
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Length')
        axes[0, 1].grid(True)
        
        # Policy loss
        axes[1, 0].plot(stats['policy_loss'])
        axes[1, 0].set_title('Policy Loss')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        
        # Value loss
        axes[1, 1].plot(stats['value_loss'])
        axes[1, 1].set_title('Value Loss')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_reward_distribution(self, rewards: List[float], 
                               save_path: Optional[str] = None) -> None:
        """Plot reward distribution."""
        plt.figure(figsize=(10, 6))
        plt.hist(rewards, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Reward Distribution')
        plt.xlabel('Episode Reward')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        plt.axvline(mean_reward, color='red', linestyle='--', 
                   label=f'Mean: {mean_reward:.2f}')
        plt.axvline(mean_reward + std_reward, color='orange', linestyle='--', 
                   alpha=0.7, label=f'+1 Std: {mean_reward + std_reward:.2f}')
        plt.axvline(mean_reward - std_reward, color='orange', linestyle='--', 
                   alpha=0.7, label=f'-1 Std: {mean_reward - std_reward:.2f}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curve(self, rewards: List[float], window: int = 100,
                          save_path: Optional[str] = None) -> None:
        """Plot smoothed learning curve."""
        if len(rewards) < window:
            window = len(rewards)
        
        # Calculate moving average
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        plt.figure(figsize=(12, 6))
        plt.plot(rewards, alpha=0.3, color='blue', label='Raw Rewards')
        plt.plot(range(window-1, len(rewards)), moving_avg, 
                color='red', linewidth=2, label=f'Moving Average ({window})')
        plt.title('Learning Curve')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class TRPOTrainer:
    """Main trainer class for TRPO."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.agent = None
        self.visualizer = TrainingVisualizer(self.config.log_dir)
    
    def _load_config(self, config_path: Optional[str]) -> TRPOConfig:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)
            
            return TRPOConfig(**config_dict)
        else:
            return TRPOConfig()
    
    def train(self) -> Dict[str, Any]:
        """Train the TRPO agent."""
        print("🚀 Starting TRPO Training")
        print(f"Environment: {self.config.env_name}")
        print(f"Total timesteps: {self.config.total_timesteps}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Max KL divergence: {self.config.max_kl_divergence}")
        print("-" * 50)
        
        # Create agent
        self.agent = TRPOAgent(self.config)
        
        # Print environment info
        env_info = get_env_info(self.agent.env)
        print(f"Environment Info:")
        for key, value in env_info.items():
            print(f"  {key}: {value}")
        print("-" * 50)
        
        # Train
        start_time = datetime.now()
        stats = self.agent.train()
        end_time = datetime.now()
        
        training_time = (end_time - start_time).total_seconds()
        print(f"\n✅ Training completed in {training_time:.2f} seconds")
        
        # Save model
        model_path = self.config.log_dir + "/trpo_model.pth"
        self.agent.save_model(model_path)
        
        # Save training stats
        stats_path = self.config.log_dir + "/training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Create visualizations
        self.visualizer.plot_training_curves(
            stats, 
            save_path=self.config.log_dir + "/training_curves.png"
        )
        
        if stats['episode_rewards']:
            self.visualizer.plot_learning_curve(
                stats['episode_rewards'],
                save_path=self.config.log_dir + "/learning_curve.png"
            )
        
        return stats
    
    def evaluate(self, model_path: str, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate a trained model."""
        if not self.agent:
            self.agent = TRPOAgent(self.config)
        
        print(f"🔍 Evaluating model: {model_path}")
        self.agent.load_model(model_path)
        
        eval_stats = self.agent.evaluate(n_episodes)
        
        print(f"Evaluation Results ({n_episodes} episodes):")
        print(f"  Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
        print(f"  Mean Length: {eval_stats['mean_length']:.2f} ± {eval_stats['std_length']:.2f}")
        
        return eval_stats
    
    def demo(self, model_path: str, n_episodes: int = 5, render: bool = True) -> None:
        """Demonstrate trained agent."""
        if not self.agent:
            self.agent = TRPOAgent(self.config)
        
        print(f"🎬 Demonstrating model: {model_path}")
        self.agent.load_model(model_path)
        
        for episode in range(n_episodes):
            obs, _ = self.agent.env.reset()
            episode_reward = 0
            episode_length = 0
            
            print(f"\nEpisode {episode + 1}:")
            
            while True:
                if render:
                    frame = self.agent.env.render()
                    if frame is not None:
                        # You could display the frame here if needed
                        pass
                
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.agent.device)
                with torch.no_grad():
                    action, _ = self.agent.policy.get_action(obs_tensor, deterministic=True)
                    action_np = action.cpu().numpy()[0]
                
                obs, reward, terminated, truncated, info = self.agent.env.step(action_np)
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            print(f"  Reward: {episode_reward:.2f}, Length: {episode_length}")


def create_config_file(config_path: str, env_name: str = "CartPole-v1") -> None:
    """Create a sample configuration file."""
    config = TRPOConfig(env_name=env_name)
    
    config_dict = {
        'env_name': config.env_name,
        'max_episode_steps': config.max_episode_steps,
        'total_timesteps': config.total_timesteps,
        'batch_size': config.batch_size,
        'n_epochs': config.n_epochs,
        'max_kl_divergence': config.max_kl_divergence,
        'damping': config.damping,
        'cg_iters': config.cg_iters,
        'backtrack_iters': config.backtrack_iters,
        'backtrack_coeff': config.backtrack_coeff,
        'max_backtrack': config.max_backtrack,
        'vf_lr': config.vf_lr,
        'vf_iters': config.vf_iters,
        'hidden_sizes': config.hidden_sizes,
        'activation': config.activation,
        'log_interval': config.log_interval,
        'save_interval': config.save_interval,
        'log_dir': config.log_dir
    }
    
    with open(config_path, 'w') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            yaml.dump(config_dict, f, default_flow_style=False)
        else:
            json.dump(config_dict, f, indent=2)
    
    print(f"✅ Configuration file created: {config_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='TRPO Training and Evaluation')
    parser.add_argument('--mode', choices=['train', 'eval', 'demo', 'config'], 
                       default='train', help='Mode to run')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--model', type=str, help='Model file path for eval/demo')
    parser.add_argument('--env', type=str, default='CartPole-v1', 
                       help='Environment name')
    parser.add_argument('--episodes', type=int, default=10, 
                       help='Number of episodes for eval/demo')
    parser.add_argument('--render', action='store_true', 
                       help='Render environment during demo')
    parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    if args.mode == 'config':
        config_path = args.config or f'config_{args.env}.yaml'
        create_config_file(config_path, args.env)
        return
    
    # Update config if output directory specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    trainer = TRPOTrainer(args.config)
    
    if args.output:
        trainer.config.log_dir = args.output
    
    if args.mode == 'train':
        trainer.train()
    
    elif args.mode == 'eval':
        if not args.model:
            print("❌ Error: Model path required for evaluation")
            return
        trainer.evaluate(args.model, args.episodes)
    
    elif args.mode == 'demo':
        if not args.model:
            print("❌ Error: Model path required for demo")
            return
        trainer.demo(args.model, args.episodes, args.render)


if __name__ == "__main__":
    main()
