"""
Unit tests for TRPO implementation.

This module contains comprehensive unit tests for the TRPO agent,
networks, and environment components.
"""

import unittest
import torch
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agents.trpo import TRPOAgent, TRPOConfig, PolicyNetwork, ValueNetwork
from envs.environments import GridWorld, MountainCarContinuous, make_env
from utils.logging import ConfigManager, ExperimentLogger


class TestPolicyNetwork(unittest.TestCase):
    """Test cases for PolicyNetwork."""
    
    def setUp(self):
        self.obs_dim = 4
        self.action_dim = 2
        self.hidden_sizes = [64, 32]
        self.policy = PolicyNetwork(self.obs_dim, self.action_dim, self.hidden_sizes)
    
    def test_forward_pass(self):
        """Test forward pass through policy network."""
        obs = torch.randn(10, self.obs_dim)
        output = self.policy(obs)
        
        self.assertEqual(output.shape, (10, self.action_dim))
        self.assertTrue(torch.isfinite(output).all())
    
    def test_get_action_deterministic(self):
        """Test deterministic action selection."""
        obs = torch.randn(1, self.obs_dim)
        action, log_prob = self.policy.get_action(obs, deterministic=True)
        
        self.assertEqual(action.shape, (1,))
        self.assertEqual(log_prob.shape, (1,))
        self.assertTrue(torch.isfinite(log_prob).all())
    
    def test_get_action_stochastic(self):
        """Test stochastic action selection."""
        obs = torch.randn(1, self.obs_dim)
        action, log_prob = self.policy.get_action(obs, deterministic=False)
        
        self.assertEqual(action.shape, (1,))
        self.assertEqual(log_prob.shape, (1,))
        self.assertTrue(torch.isfinite(log_prob).all())
        self.assertTrue(0 <= action.item() < self.action_dim)


class TestValueNetwork(unittest.TestCase):
    """Test cases for ValueNetwork."""
    
    def setUp(self):
        self.obs_dim = 4
        self.hidden_sizes = [64, 32]
        self.value_net = ValueNetwork(self.obs_dim, self.hidden_sizes)
    
    def test_forward_pass(self):
        """Test forward pass through value network."""
        obs = torch.randn(10, self.obs_dim)
        output = self.value_net(obs)
        
        self.assertEqual(output.shape, (10,))
        self.assertTrue(torch.isfinite(output).all())


class TestTRPOConfig(unittest.TestCase):
    """Test cases for TRPOConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TRPOConfig()
        
        self.assertEqual(config.env_name, "CartPole-v1")
        self.assertEqual(config.total_timesteps, 100000)
        self.assertEqual(config.batch_size, 4000)
        self.assertEqual(config.max_kl_divergence, 0.01)
        self.assertIsInstance(config.hidden_sizes, list)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TRPOConfig(
            env_name="MountainCar-v0",
            total_timesteps=50000,
            batch_size=2000,
            hidden_sizes=[128, 64]
        )
        
        self.assertEqual(config.env_name, "MountainCar-v0")
        self.assertEqual(config.total_timesteps, 50000)
        self.assertEqual(config.batch_size, 2000)
        self.assertEqual(config.hidden_sizes, [128, 64])


class TestTRPOAgent(unittest.TestCase):
    """Test cases for TRPOAgent."""
    
    def setUp(self):
        self.config = TRPOConfig(
            env_name="CartPole-v1",
            total_timesteps=1000,  # Small for testing
            batch_size=100,
            log_interval=5
        )
        self.agent = TRPOAgent(self.config)
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertIsNotNone(self.agent.policy)
        self.assertIsNotNone(self.agent.value_function)
        self.assertIsNotNone(self.agent.env)
        self.assertEqual(self.agent.obs_dim, 4)  # CartPole observation space
        self.assertEqual(self.agent.action_dim, 2)  # CartPole action space
    
    def test_collect_trajectories(self):
        """Test trajectory collection."""
        trajectories = self.agent.collect_trajectories(50)
        
        self.assertIn('obs', trajectories)
        self.assertIn('actions', trajectories)
        self.assertIn('rewards', trajectories)
        self.assertIn('dones', trajectories)
        self.assertIn('old_log_probs', trajectories)
        
        self.assertEqual(len(trajectories['obs']), 50)
        self.assertEqual(len(trajectories['actions']), 50)
        self.assertEqual(len(trajectories['rewards']), 50)
        self.assertEqual(len(trajectories['dones']), 50)
        self.assertEqual(len(trajectories['old_log_probs']), 50)
    
    def test_compute_advantages(self):
        """Test advantage computation."""
        batch_size = 20
        rewards = torch.randn(batch_size)
        dones = torch.zeros(batch_size, dtype=torch.bool)
        dones[-1] = True  # End episode
        obs = torch.randn(batch_size, self.agent.obs_dim)
        
        advantages, returns = self.agent.compute_advantages(rewards, dones, obs)
        
        self.assertEqual(advantages.shape, (batch_size,))
        self.assertEqual(returns.shape, (batch_size,))
        self.assertTrue(torch.isfinite(advantages).all())
        self.assertTrue(torch.isfinite(returns).all())
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pth")
            
            # Save model
            self.agent.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Load model
            self.agent.load_model(model_path)
            # If no exception is raised, loading was successful
    
    def test_evaluate(self):
        """Test model evaluation."""
        eval_stats = self.agent.evaluate(n_episodes=3)
        
        self.assertIn('mean_reward', eval_stats)
        self.assertIn('std_reward', eval_stats)
        self.assertIn('mean_length', eval_stats)
        self.assertIn('std_length', eval_stats)
        
        self.assertIsInstance(eval_stats['mean_reward'], float)
        self.assertIsInstance(eval_stats['std_reward'], float)
        self.assertIsInstance(eval_stats['mean_length'], float)
        self.assertIsInstance(eval_stats['std_length'], float)


class TestGridWorld(unittest.TestCase):
    """Test cases for GridWorld environment."""
    
    def setUp(self):
        self.env = GridWorld(size=5)
    
    def test_environment_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.size, 5)
        self.assertEqual(self.env.action_space.n, 4)
        self.assertEqual(self.env.observation_space.shape, (4,))
    
    def test_reset(self):
        """Test environment reset."""
        obs, info = self.env.reset()
        
        self.assertEqual(obs.shape, (4,))
        self.assertEqual(obs[0], 0)  # Agent x position
        self.assertEqual(obs[1], 0)  # Agent y position
        self.assertEqual(obs[2], 4)  # Goal x position
        self.assertEqual(obs[3], 4)  # Goal y position
    
    def test_step(self):
        """Test environment step."""
        obs, _ = self.env.reset()
        
        # Take a step
        obs, reward, terminated, truncated, info = self.env.step(0)  # Move up
        
        self.assertEqual(obs.shape, (4,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
    
    def test_goal_reaching(self):
        """Test reaching the goal."""
        obs, _ = self.env.reset()
        
        # Move to goal (assuming goal is at bottom-right)
        for _ in range(10):
            obs, reward, terminated, truncated, info = self.env.step(3)  # Move right
            if terminated:
                break
        
        # Should eventually reach goal or timeout
        self.assertTrue(terminated or truncated)


class TestMountainCarContinuous(unittest.TestCase):
    """Test cases for MountainCarContinuous environment."""
    
    def setUp(self):
        self.env = MountainCarContinuous()
    
    def test_environment_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.action_space.shape, (1,))
        self.assertEqual(self.env.observation_space.shape, (2,))
    
    def test_reset(self):
        """Test environment reset."""
        obs, info = self.env.reset()
        
        self.assertEqual(obs.shape, (2,))
        self.assertTrue(self.env.min_position <= obs[0] <= self.env.max_position)
        self.assertTrue(-self.env.max_speed <= obs[1] <= self.env.max_speed)
    
    def test_step(self):
        """Test environment step."""
        obs, _ = self.env.reset()
        
        # Take a step
        action = np.array([0.5])  # Apply positive force
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.assertEqual(obs.shape, (2,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_load_config(self):
        """Test configuration saving and loading."""
        config = {"test_param": 42, "nested": {"value": "test"}}
        
        # Save as JSON
        json_path = self.config_manager.save_config(config, "test.json")
        self.assertTrue(os.path.exists(json_path))
        
        # Load JSON
        loaded_config = self.config_manager.load_config("test.json")
        self.assertEqual(loaded_config, config)
        
        # Save as YAML
        yaml_path = self.config_manager.save_config(config, "test.yaml")
        self.assertTrue(os.path.exists(yaml_path))
        
        # Load YAML
        loaded_config = self.config_manager.load_config("test.yaml")
        self.assertEqual(loaded_config, config)
    
    def test_get_experiment_config(self):
        """Test getting experiment configuration."""
        config = self.config_manager.get_experiment_config("CartPole-v1")
        
        self.assertIn("env_name", config)
        self.assertIn("total_timesteps", config)
        self.assertIn("batch_size", config)
        self.assertEqual(config["env_name"], "CartPole-v1")


class TestExperimentLogger(unittest.TestCase):
    """Test cases for ExperimentLogger."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logger = ExperimentLogger(self.temp_dir, "test_experiment")
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_log_hyperparameters(self):
        """Test hyperparameter logging."""
        config = {"learning_rate": 0.001, "batch_size": 32}
        self.logger.log_hyperparameters(config)
        
        config_file = Path(self.temp_dir) / "config.json"
        self.assertTrue(config_file.exists())
    
    def test_log_metrics(self):
        """Test metrics logging."""
        metrics = {"reward": 10.5, "loss": 0.1}
        self.logger.log_metrics(metrics, step=100)
        
        metrics_file = Path(self.temp_dir) / "metrics.jsonl"
        self.assertTrue(metrics_file.exists())
    
    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        # Create a simple agent for testing
        config = TRPOConfig(total_timesteps=100)
        agent = TRPOAgent(config)
        
        self.logger.save_checkpoint(agent, step=50)
        
        checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        self.assertTrue(checkpoint_dir.exists())
        
        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        self.assertEqual(len(checkpoint_files), 1)


class TestEnvironmentIntegration(unittest.TestCase):
    """Integration tests for environments."""
    
    def test_make_env_cartpole(self):
        """Test making CartPole environment."""
        env = make_env("CartPole-v1")
        
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (4,))
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        self.assertEqual(obs.shape, (4,))
        self.assertIsInstance(reward, float)
    
    def test_make_env_gridworld(self):
        """Test making GridWorld environment."""
        env = make_env("GridWorld", size=6)
        
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (4,))
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        self.assertEqual(obs.shape, (4,))
        self.assertIsInstance(reward, float)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestPolicyNetwork,
        TestValueNetwork,
        TestTRPOConfig,
        TestTRPOAgent,
        TestGridWorld,
        TestMountainCarContinuous,
        TestConfigManager,
        TestExperimentLogger,
        TestEnvironmentIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
