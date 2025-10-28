"""
Configuration management and logging utilities.

This module provides configuration loading/saving and logging setup
for TRPO training experiments.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import torch
import numpy as np


class ConfigManager:
    """Manages configuration files and experiment settings."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def save_config(self, config: Dict[str, Any], filename: str) -> str:
        """Save configuration to file."""
        config_path = self.config_dir / filename
        
        with open(config_path, 'w') as f:
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                yaml.dump(config, f, default_flow_style=False)
            else:
                json.dump(config, f, indent=2)
        
        return str(config_path)
    
    def load_config(self, filename: str) -> Dict[str, Any]:
        """Load configuration from file."""
        config_path = self.config_dir / filename
        
        with open(config_path, 'r') as f:
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                return json.load(f)
    
    def get_experiment_config(self, env_name: str, 
                            total_timesteps: int = 100000) -> Dict[str, Any]:
        """Get configuration for specific environment."""
        configs = {
            "CartPole-v1": {
                "env_name": "CartPole-v1",
                "max_episode_steps": 500,
                "total_timesteps": total_timesteps,
                "batch_size": 4000,
                "n_epochs": 10,
                "max_kl_divergence": 0.01,
                "damping": 0.1,
                "cg_iters": 10,
                "backtrack_iters": 10,
                "backtrack_coeff": 0.8,
                "max_backtrack": 10,
                "vf_lr": 1e-3,
                "vf_iters": 5,
                "hidden_sizes": [64, 64],
                "activation": "tanh",
                "log_interval": 10,
                "save_interval": 100,
                "log_dir": f"logs/cartpole_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            },
            "MountainCar-v0": {
                "env_name": "MountainCar-v0",
                "max_episode_steps": 200,
                "total_timesteps": total_timesteps,
                "batch_size": 2000,
                "n_epochs": 10,
                "max_kl_divergence": 0.01,
                "damping": 0.1,
                "cg_iters": 10,
                "backtrack_iters": 10,
                "backtrack_coeff": 0.8,
                "max_backtrack": 10,
                "vf_lr": 1e-3,
                "vf_iters": 5,
                "hidden_sizes": [64, 64],
                "activation": "tanh",
                "log_interval": 10,
                "save_interval": 100,
                "log_dir": f"logs/mountaincar_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            },
            "GridWorld": {
                "env_name": "GridWorld",
                "max_episode_steps": 100,
                "total_timesteps": total_timesteps,
                "batch_size": 2000,
                "n_epochs": 10,
                "max_kl_divergence": 0.01,
                "damping": 0.1,
                "cg_iters": 10,
                "backtrack_iters": 10,
                "backtrack_coeff": 0.8,
                "max_backtrack": 10,
                "vf_lr": 1e-3,
                "vf_iters": 5,
                "hidden_sizes": [64, 64],
                "activation": "tanh",
                "log_interval": 10,
                "save_interval": 100,
                "log_dir": f"logs/gridworld_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
        }
        
        return configs.get(env_name, configs["CartPole-v1"])


class ExperimentLogger:
    """Logging utilities for experiments."""
    
    def __init__(self, log_dir: str, experiment_name: str = "trpo_experiment"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # Setup logging
        self.setup_logging()
        
        # Experiment metadata
        self.start_time = datetime.now()
        self.metadata = {
            "experiment_name": experiment_name,
            "start_time": self.start_time.isoformat(),
            "log_dir": str(self.log_dir)
        }
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.log_dir / "experiment.log"
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        self.logger = logger
    
    def log_hyperparameters(self, config: Dict[str, Any]):
        """Log hyperparameters."""
        # Convert numpy types to Python types for JSON serialization
        serializable_config = {}
        for key, value in config.items():
            if isinstance(value, np.integer):
                serializable_config[key] = int(value)
            elif isinstance(value, np.floating):
                serializable_config[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable_config[key] = value.tolist()
            else:
                serializable_config[key] = value
        
        self.metadata["hyperparameters"] = serializable_config
        
        # Save to file
        config_file = self.log_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(serializable_config, f, indent=2)
        
        self.logger.info("Hyperparameters logged")
    
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log training metrics."""
        log_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        # Append to metrics log
        metrics_file = self.log_dir / "metrics.jsonl"
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Log to console
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} - {metrics_str}")
    
    def log_model_info(self, model: torch.nn.Module):
        """Log model architecture and parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }
        
        self.metadata["model_info"] = model_info
        
        # Save model info
        model_info_file = self.log_dir / "model_info.json"
        with open(model_info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        self.logger.info(f"Model info logged: {total_params:,} total parameters")
    
    def log_environment_info(self, env_info: Dict[str, Any]):
        """Log environment information."""
        # Convert non-serializable objects to strings
        serializable_env_info = {}
        for key, value in env_info.items():
            if hasattr(value, '__dict__'):
                serializable_env_info[key] = str(value)
            else:
                serializable_env_info[key] = value
        
        self.metadata["environment"] = serializable_env_info
        
        env_info_file = self.log_dir / "environment.json"
        with open(env_info_file, 'w') as f:
            json.dump(serializable_env_info, f, indent=2)
        
        self.logger.info("Environment info logged")
    
    def save_checkpoint(self, agent, step: int):
        """Save training checkpoint."""
        checkpoint_dir = self.log_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pth"
        
        checkpoint = {
            "step": step,
            "policy_state_dict": agent.policy.state_dict(),
            "value_function_state_dict": agent.value_function.state_dict(),
            "training_stats": agent.training_stats,
            "config": agent.config.__dict__,
            "timestamp": datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    
    def finalize_experiment(self, final_stats: Dict[str, Any]):
        """Finalize experiment logging."""
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["duration_seconds"] = (
            datetime.now() - self.start_time
        ).total_seconds()
        self.metadata["final_stats"] = final_stats
        
        # Save experiment summary
        summary_file = self.log_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        self.logger.info("Experiment finalized")
        self.logger.info(f"Total duration: {self.metadata['duration_seconds']:.2f} seconds")


class TensorBoardLogger:
    """TensorBoard logging integration."""
    
    def __init__(self, log_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.available = True
        except ImportError:
            print("Warning: TensorBoard not available. Install with: pip install tensorboard")
            self.available = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value to TensorBoard."""
        if self.available:
            self.writer.add_scalar(tag, value, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        """Log histogram to TensorBoard."""
        if self.available:
            self.writer.add_histogram(tag, values, step)
    
    def log_model_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor):
        """Log model graph to TensorBoard."""
        if self.available:
            self.writer.add_graph(model, input_tensor)
    
    def close(self):
        """Close TensorBoard writer."""
        if self.available:
            self.writer.close()


def setup_experiment_logging(log_dir: str, experiment_name: str = "trpo_experiment"):
    """Setup complete experiment logging."""
    logger = ExperimentLogger(log_dir, experiment_name)
    tb_logger = TensorBoardLogger(log_dir)
    
    return logger, tb_logger


def create_experiment_config(env_name: str, **kwargs) -> Dict[str, Any]:
    """Create experiment configuration with sensible defaults."""
    config_manager = ConfigManager()
    base_config = config_manager.get_experiment_config(env_name)
    
    # Update with provided kwargs
    base_config.update(kwargs)
    
    return base_config
