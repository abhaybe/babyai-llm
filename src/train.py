"""
Main training script for BabyAI with LLM integration.
Uses Stable-Baselines3 for PPO training.
"""

import argparse
import yaml
import os
from pathlib import Path
from datetime import datetime
import json

import gymnasium as gym
import minigrid  # Required for MiniGrid environments to register
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from minigrid.wrappers import ImgObsWrapper  # Flatten observation to just image

from env_wrapper import make_babyai_env, LLMMode
from llm_wrapper import LLMWrapper
from minigrid_extractor import MinigridFeaturesExtractor
from llm_help_mode import HelpWhenStuckWrapper
from llm_code_gen import CodeGenRewardShaper


class LLMStatsCallback:
    """Callback to log LLM usage statistics."""
    
    def __init__(self, env, log_freq=100):
        self.env = env
        self.log_freq = log_freq
        self.n_calls = 0
    
    def __call__(self, locals_dict, globals_dict):
        self.n_calls += 1
        if self.n_calls % self.log_freq == 0:
            # Try to get stats from environment
            if hasattr(self.env, 'get_llm_stats'):
                stats = self.env.get_llm_stats()
                print(f"\nLLM Stats at step {self.n_calls * self.log_freq}:")
                print(json.dumps(stats, indent=2))
        return True


def make_env(config: dict, rank: int = 0):
    """
    Create a single environment instance.
    
    Args:
        config: Configuration dictionary
        rank: Rank for multi-processing
        
    Returns:
        A function that creates the environment
    """
    def _init():
        env = make_babyai_env(
            env_name=config['env_name'],
            llm_mode=config['llm_mode'],
            llm_config=config.get('llm_config'),
            help_threshold=config.get('help_threshold', 50),
            max_help_requests=config.get('max_help_requests', 3),
            verbose=config.get('verbose', True) and rank == 0  # Only verbose for first env
        )
        
        # Add Help-When-Stuck mode
        if config.get('use_help_mode', False):
            from llm_wrapper import LLMWrapper
            
            llm_config = config.get('llm_config', {})
            llm = LLMWrapper(**llm_config)
            
            env = HelpWhenStuckWrapper(
                env,
                llm,
                stuck_threshold=config.get('stuck_threshold', 50),
                max_help_per_episode=config.get('max_help_per_episode', 3),
                verbose=(rank == 0)
            )
            
            if rank == 0:
                print("\n" + "="*60)
                print("✓ HELP-WHEN-STUCK MODE ENABLED")
                print(f"  Stuck threshold: {config.get('stuck_threshold', 50)} steps")
                print(f"  Max help per episode: {config.get('max_help_per_episode', 3)}")
                print("="*60 + "\n")
        
        # Add Code Generation mode
        if config.get('use_code_gen', False):
            from llm_wrapper import LLMWrapper
            
            llm_config = config.get('llm_config', {})
            llm = LLMWrapper(**llm_config)
            
            env = CodeGenRewardShaper(
                env,
                llm,
                shaping_weight=config.get('code_gen_weight', 1.0),
                verbose=(rank == 0)
            )
            
            if rank == 0:
                print("\n" + "="*60)
                print("✓ CODE GENERATION MODE ENABLED")
                print(f"  Shaping weight: {config.get('code_gen_weight', 1.0)}")
                print("="*60 + "\n")
        
        # Wrap with ImgObsWrapper to convert dict obs to just image
        # This makes it compatible with CNN policies
        env = ImgObsWrapper(env)
        env = Monitor(env)
        return env
    
    return _init


def train(config_or_path):
    """
    Train a PPO agent on BabyAI.
    
    Args:
        config_or_path: Either a dict (config) or str (path to config YAML)
    """
    # Load configuration
    if isinstance(config_or_path, str):
        with open(config_or_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = config_or_path
    
    print("Configuration:")
    print(json.dumps(config, indent=2))
    
    # Create output directory
    exp_name = config.get('exp_name', 'experiment')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.get('output_dir', './output')) / f"{exp_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Create environments
    n_envs = config.get('n_envs', 4)
    
    if n_envs == 1:
        env = DummyVecEnv([make_env(config, 0)])
    else:
        # For multiple envs, only first one should use LLM to avoid parallel API calls
        # (This is a design choice - you could also implement parallel LLM calls)
        env_fns = [make_env(config, i) for i in range(n_envs)]
        env = DummyVecEnv(env_fns)  # Use DummyVecEnv to avoid pickling issues with LLM
    
    # Create evaluation environment (no LLM for faster eval)
    eval_config = config.copy()
    eval_config['llm_mode'] = config.get('eval_llm_mode', config['llm_mode'])
    eval_config['verbose'] = False
    eval_env = DummyVecEnv([make_env(eval_config, 0)])
    
    # Create PPO model
    ppo_config = config.get('ppo_config', {})
    
    # Use custom feature extractor for MiniGrid's small observations
    # Larger network for better performance on harder tasks
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,  # Increased from 128
            cnn_channels=[32, 64, 64]  # Larger CNN
        ),
        net_arch=dict(
            pi=[256, 256],  # Policy network: 2 layers of 256 units each
            vf=[256, 256]   # Value network: 2 layers of 256 units each
        )
    )
    
    model = PPO(
        policy="CnnPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=ppo_config.get('learning_rate', 3e-4),
        n_steps=ppo_config.get('n_steps', 128),
        batch_size=ppo_config.get('batch_size', 64),
        n_epochs=ppo_config.get('n_epochs', 4),
        gamma=ppo_config.get('gamma', 0.99),
        gae_lambda=ppo_config.get('gae_lambda', 0.95),
        clip_range=ppo_config.get('clip_range', 0.2),
        ent_coef=ppo_config.get('ent_coef', 0.01),
        vf_coef=ppo_config.get('vf_coef', 0.5),
        max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
        verbose=1,
        tensorboard_log=str(output_dir / "tensorboard")
    )
    
    print("\nModel architecture:")
    print(model.policy)
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config.get('checkpoint_freq', 10000),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_babyai"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval"),
        eval_freq=config.get('eval_freq', 5000),
        n_eval_episodes=config.get('n_eval_episodes', 10),
        deterministic=True
    )
    
    callbacks = [checkpoint_callback, eval_callback]
    
    # Train
    total_timesteps = config.get('total_timesteps', 100000)
    print(f"\nStarting training for {total_timesteps} timesteps...")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    model.save(output_dir / "final_model")
    print(f"\nTraining complete! Model saved to {output_dir}")
    
    # Print LLM statistics if applicable
    if config['llm_mode'] != 'none':
        try:
            unwrapped_env = env.envs[0].env  # Get first unwrapped environment
            if hasattr(unwrapped_env, 'get_llm_stats'):
                print("\nFinal LLM Statistics:")
                print(json.dumps(unwrapped_env.get_llm_stats(), indent=2))
        except Exception as e:
            print(f"Could not get LLM stats: {e}")
    
    # Cleanup
    env.close()
    eval_env.close()


def evaluate(model_path: str, config_path: str, n_episodes: int = 20):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to saved model
        config_path: Path to configuration file
        n_episodes: Number of episodes to evaluate
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    eval_config = config.copy()
    eval_config['verbose'] = True
    env = DummyVecEnv([make_env(eval_config, 0)])
    
    # Load model
    model = PPO.load(model_path)
    
    # Evaluate
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"\nEvaluating for {n_episodes} episodes...")
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            ep_length += 1
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        if ep_reward > 0:
            success_count += 1
        
        print(f"Episode {ep+1}: reward={ep_reward:.2f}, length={ep_length}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Evaluation Results:")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Success rate: {success_count}/{n_episodes} ({100*success_count/n_episodes:.1f}%)")
    print("="*60)
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent on BabyAI with LLM integration")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--eval", action="store_true", help="Evaluate instead of train")
    parser.add_argument("--model", type=str, help="Path to model for evaluation")
    parser.add_argument("--n_eval_episodes", type=int, default=20, help="Number of evaluation episodes")
    
    # Config overrides
    parser.add_argument("--total_timesteps", type=int, help="Override total_timesteps from config")
    parser.add_argument("--n_envs", type=int, help="Override n_envs from config")
    parser.add_argument("--learning_rate", type=float, help="Override learning_rate from config")
    
    args = parser.parse_args()
    
    if args.eval:
        if not args.model:
            raise ValueError("Must provide --model path for evaluation")
        evaluate(args.model, args.config, args.n_eval_episodes)
    else:
        # Check if we need to apply overrides
        has_overrides = args.total_timesteps or args.n_envs or args.learning_rate
        
        if has_overrides:
            # Load config and apply overrides
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            
            if args.total_timesteps:
                config['total_timesteps'] = args.total_timesteps
                print(f"Override: total_timesteps = {args.total_timesteps}")
            
            if args.n_envs:
                config['n_envs'] = args.n_envs
                print(f"Override: n_envs = {args.n_envs}")
            
            if args.learning_rate:
                config['ppo_config']['learning_rate'] = args.learning_rate
                print(f"Override: learning_rate = {args.learning_rate}")
            
            # Pass modified config dict
            train(config)
        else:
            # No overrides, pass config path
            train(args.config)
