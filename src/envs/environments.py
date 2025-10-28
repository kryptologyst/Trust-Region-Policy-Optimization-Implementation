"""
Environment wrappers and custom environments for TRPO training.

This module provides environment wrappers and custom environments
for testing and training TRPO agents.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations to [-1, 1] range."""
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.obs_low = self.observation_space.low
        self.obs_high = self.observation_space.high
        self.obs_range = self.obs_high - self.obs_low
        
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation to [-1, 1]."""
        # Handle infinite bounds
        obs_range = self.obs_range.copy()
        obs_range[obs_range == 0] = 1.0  # Avoid division by zero
        obs_range[np.isinf(obs_range)] = 1.0  # Handle infinite ranges
        
        normalized = 2.0 * (obs - self.obs_low) / obs_range - 1.0
        return np.clip(normalized, -1.0, 1.0)


class RewardShaping(gym.RewardWrapper):
    """Add reward shaping to improve learning."""
    
    def __init__(self, env: gym.Env, reward_scale: float = 1.0):
        super().__init__(env)
        self.reward_scale = reward_scale
        
    def reward(self, reward: float) -> float:
        """Scale reward."""
        return reward * self.reward_scale


class EpisodeLogger(gym.Wrapper):
    """Log episode statistics."""
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_count = 0
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        self.episode_length += 1
        
        if terminated or truncated:
            self.episode_count += 1
            info['episode_reward'] = self.episode_reward
            info['episode_length'] = self.episode_length
            self.episode_reward = 0
            self.episode_length = 0
            
        return obs, reward, terminated, truncated, info


class GridWorld(gym.Env):
    """
    Custom GridWorld environment for testing TRPO.
    
    A simple grid world where the agent must navigate from start to goal
    while avoiding obstacles.
    """
    
    def __init__(self, size: int = 8, max_steps: int = 100):
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.current_step = 0
        
        # Define action space (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        
        # Define observation space (agent position + goal position)
        self.observation_space = spaces.Box(
            low=0, high=size-1, shape=(4,), dtype=np.float32
        )
        
        # Initialize grid
        self.grid = np.zeros((size, size))
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([size-1, size-1])
        
        # Add some obstacles
        self._add_obstacles()
        
        # Action mappings
        self.actions = {
            0: np.array([-1, 0]),  # up
            1: np.array([1, 0]),   # down
            2: np.array([0, -1]),  # left
            3: np.array([0, 1])    # right
        }
        
    def _add_obstacles(self):
        """Add random obstacles to the grid."""
        num_obstacles = self.size // 2
        for _ in range(num_obstacles):
            x, y = random.randint(1, self.size-2), random.randint(1, self.size-2)
            if (x, y) != tuple(self.goal_pos) and (x, y) != tuple(self.agent_pos):
                self.grid[x, y] = 1
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        return np.concatenate([self.agent_pos, self.goal_pos]).astype(np.float32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([self.size-1, self.size-1])
        self.current_step = 0
        
        # Regenerate obstacles
        self.grid = np.zeros((self.size, self.size))
        self._add_obstacles()
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        self.current_step += 1
        
        # Move agent
        new_pos = self.agent_pos + self.actions[action]
        
        # Check bounds
        if (0 <= new_pos[0] < self.size and 
            0 <= new_pos[1] < self.size and 
            self.grid[new_pos[0], new_pos[1]] == 0):
            self.agent_pos = new_pos
        
        # Calculate reward
        distance_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        reward = -0.01  # Small negative reward for each step
        
        # Check if goal reached
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 1.0
            terminated = True
        else:
            terminated = False
        
        # Check if max steps reached
        truncated = self.current_step >= self.max_steps
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def render(self, mode: str = 'rgb_array') -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == 'rgb_array':
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Draw grid
            for i in range(self.size):
                for j in range(self.size):
                    if self.grid[i, j] == 1:  # Obstacle
                        rect = Rectangle((j, i), 1, 1, facecolor='black')
                        ax.add_patch(rect)
                    elif i == self.goal_pos[0] and j == self.goal_pos[1]:  # Goal
                        rect = Rectangle((j, i), 1, 1, facecolor='green')
                        ax.add_patch(rect)
                    elif i == self.agent_pos[0] and j == self.agent_pos[1]:  # Agent
                        rect = Rectangle((j, i), 1, 1, facecolor='blue')
                        ax.add_patch(rect)
            
            ax.set_xlim(0, self.size)
            ax.set_ylim(0, self.size)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_title('GridWorld')
            
            # Convert to numpy array
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            return image
        
        return None


class MountainCarContinuous(gym.Env):
    """
    Continuous version of MountainCar environment.
    
    This is a simplified continuous version for testing TRPO
    on continuous control tasks.
    """
    
    def __init__(self):
        super().__init__()
        
        # Continuous action space (force)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space (position, velocity)
        self.observation_space = spaces.Box(
            low=np.array([-1.2, -0.07]), 
            high=np.array([0.6, 0.07]), 
            dtype=np.float32
        )
        
        # Environment parameters
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = 0.0
        
        # Physics parameters
        self.force = 0.001
        self.gravity = 0.0025
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Random starting position
        self.state = np.array([
            np.random.uniform(low=-0.6, high=-0.4),
            np.random.uniform(low=-0.007, high=0.007)
        ], dtype=np.float32)
        
        return self.state.copy(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        position, velocity = self.state
        
        # Apply force
        force = action[0] * self.force
        
        # Update velocity
        velocity += force - self.gravity * np.cos(3 * position)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        
        # Update position
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        
        # Reset velocity if hit the left wall
        if position == self.min_position and velocity < 0:
            velocity = 0
        
        self.state = np.array([position, velocity], dtype=np.float32)
        
        # Calculate reward
        reward = 0
        if position >= self.goal_position:
            reward = 100
            terminated = True
        else:
            # Small reward for getting closer to goal
            reward = -1 + (position - self.min_position) / (self.max_position - self.min_position)
            terminated = False
        
        truncated = False
        
        return self.state.copy(), reward, terminated, truncated, {}
    
    def render(self, mode: str = 'rgb_array') -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == 'rgb_array':
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Draw mountain
            x = np.linspace(self.min_position, self.max_position, 100)
            y = np.sin(3 * x)
            ax.plot(x, y, 'b-', linewidth=2)
            
            # Draw goal
            ax.plot(self.goal_position, np.sin(3 * self.goal_position), 'go', markersize=10)
            
            # Draw car
            car_x = self.state[0]
            car_y = np.sin(3 * car_x)
            ax.plot(car_x, car_y, 'ro', markersize=8)
            
            ax.set_xlim(self.min_position, self.max_position)
            ax.set_ylim(-1.1, 1.1)
            ax.set_title('MountainCar Continuous')
            ax.grid(True)
            
            # Convert to numpy array
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            return image
        
        return None


def make_env(env_name: str, **kwargs) -> gym.Env:
    """Create and wrap environment."""
    if env_name == "GridWorld":
        env = GridWorld(**kwargs)
    elif env_name == "MountainCarContinuous":
        env = MountainCarContinuous()
    else:
        env = gym.make(env_name, **kwargs)
    
    # Apply wrappers
    env = EpisodeLogger(env)
    env = RewardShaping(env, reward_scale=1.0)
    
    # Only normalize if not already normalized
    if not isinstance(env, NormalizeObservation):
        env = NormalizeObservation(env)
    
    return env


def get_env_info(env: gym.Env) -> Dict[str, Any]:
    """Get environment information."""
    obs_dim = None
    action_dim = None
    
    if hasattr(env.observation_space, 'shape'):
        obs_dim = int(env.observation_space.shape[0])
    
    if isinstance(env.action_space, spaces.Discrete):
        action_dim = int(env.action_space.n)
    elif hasattr(env.action_space, 'shape'):
        action_dim = int(env.action_space.shape[0])
    
    return {
        'name': env.spec.id if hasattr(env.spec, 'id') else 'Custom',
        'observation_space': str(env.observation_space),
        'action_space': str(env.action_space),
        'is_discrete': isinstance(env.action_space, spaces.Discrete),
        'obs_dim': obs_dim,
        'action_dim': action_dim
    }
