#!/usr/bin/env python3
"""
Simple evaluation script for trained models.
Shows clear success rates and statistics.
"""

import argparse
import yaml
import numpy as np
from pathlib import Path

import gymnasium as gym
import minigrid
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.monitor import Monitor

from env_wrapper import make_babyai_env


def make_eval_env(config):
    """Create environment for evaluation."""
    def _init():
        env = make_babyai_env(
            env_name=config['env_name'],
            llm_mode='none',  # No LLM during evaluation
            llm_config=None,
            verbose=False
        )
        env = ImgObsWrapper(env)
        env = Monitor(env)
        return env
    return _init


def evaluate_model(model_path, config_path, n_episodes=50):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to saved model (e.g., output/exp_name_timestamp/best_model)
        config_path: Path to config file
        n_episodes: Number of episodes to evaluate
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*70)
    print(f"EVALUATING MODEL: {model_path}")
    print(f"Config: {config_path}")
    print(f"Environment: {config['env_name']}")
    print(f"Episodes: {n_episodes}")
    print("="*70)
    
    # Create environment
    env = DummyVecEnv([make_eval_env(config)])
    
    # Get actual env to check
    test_env = make_eval_env(config)()
    print(f"Environment created: {test_env.spec.id if hasattr(test_env, 'spec') else 'Unknown'}")
    print(f"Max episode steps: {test_env.max_steps if hasattr(test_env, 'max_steps') else 'Unknown'}")
    test_env.close()
    print()
    
    # Load model
    try:
        # Check if it's a directory (best_model) or a file (final_model.zip)
        model_path_obj = Path(model_path)
        
        if model_path_obj.is_dir():
            # It's a directory, look for .zip file inside
            zip_files = list(model_path_obj.glob("*.zip"))
            if zip_files:
                actual_model_path = str(zip_files[0])
                print(f"Loading from: {actual_model_path}")
            else:
                # Try loading the directory directly (some saves are like this)
                actual_model_path = str(model_path_obj)
                print(f"Loading directory: {actual_model_path}")
        else:
            actual_model_path = model_path
            print(f"Loading file: {actual_model_path}")
        
        model = PPO.load(actual_model_path)
        print(f"✓ Model loaded successfully")
        
        # Show model info
        print(f"  Policy type: {type(model.policy).__name__}")
        print(f"  Observation space: {model.observation_space}")
        print()
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Evaluate
    episode_rewards = []
    episode_lengths = []
    successes = []
    
    print(f"Running {n_episodes} episodes...\n")
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0
        
        # Print first episode in detail for debugging
        show_detail = (ep == 0 and n_episodes > 1)
        
        if show_detail:
            print(f"  [Episode 1 Detail - for debugging]")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            ep_length += 1
            
            if show_detail and ep_length <= 10:
                print(f"    Step {ep_length}: action={action}, reward={reward[0]:.3f}")
        
        if show_detail:
            print(f"    ... (episode completed in {ep_length} steps, reward={ep_reward:.3f})\n")
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        success = ep_reward > 0
        successes.append(success)
        
        # Print progress every 10 episodes
        if (ep + 1) % 10 == 0:
            current_success_rate = sum(successes) / len(successes)
            print(f"  Episodes {ep+1}/{n_episodes}: {current_success_rate*100:.1f}% success so far")
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    success_count = sum(successes)
    success_rate = success_count / n_episodes
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Success Rate:     {success_count}/{n_episodes} ({success_rate*100:.1f}%)")
    print(f"Mean Reward:      {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"Mean Episode Len: {mean_length:.1f} ± {std_length:.1f} steps")
    print("="*70)
    
    # Success distribution
    if success_rate > 0:
        successful_lengths = [episode_lengths[i] for i in range(n_episodes) if successes[i]]
        print(f"\nSuccessful episodes took: {np.mean(successful_lengths):.1f} ± {np.std(successful_lengths):.1f} steps")
    
    if success_rate < 1:
        failed_lengths = [episode_lengths[i] for i in range(n_episodes) if not successes[i]]
        print(f"Failed episodes took:     {np.mean(failed_lengths):.1f} ± {np.std(failed_lengths):.1f} steps")
    
    print("\n")
    
    env.close()
    
    return {
        'success_rate': success_rate,
        'mean_reward': mean_reward,
        'mean_length': mean_length,
        'success_count': success_count,
        'n_episodes': n_episodes
    }


def find_models_in_output(output_dir='./output'):
    """Find all trained models in output directory."""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Output directory not found: {output_dir}")
        return []
    
    models = []
    
    # Look for experiment directories
    for exp_dir in output_path.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Check for best_model
        best_model = exp_dir / 'best_model'
        if best_model.exists():
            models.append(('best', exp_dir.name, best_model))
        
        # Check for final_model
        final_model = exp_dir / 'final_model.zip'
        if final_model.exists():
            models.append(('final', exp_dir.name, final_model))
    
    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained PPO models")
    parser.add_argument("--model", type=str, help="Path to model (e.g., output/exp_name/best_model)")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--n_episodes", type=int, default=50, help="Number of evaluation episodes")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--all", action="store_true", help="Evaluate all models in output/")
    
    args = parser.parse_args()
    
    if args.list or args.all:
        # Find all models
        print("\nSearching for trained models in ./output/...\n")
        models = find_models_in_output()
        
        if not models:
            print("No trained models found in ./output/")
            print("\nRun training first:")
            print("  python train.py --config ../configs/baseline_doorkey.yaml")
            exit(0)
        
        print("Found models:")
        print("-" * 70)
        for model_type, exp_name, model_path in models:
            print(f"  [{model_type:5s}] {exp_name}")
            print(f"           {model_path}")
        print("-" * 70)
        print()
        
        if args.list:
            print("To evaluate a model:")
            print("  python eval.py --model output/exp_name/best_model --config ../configs/exp_config.yaml")
            exit(0)
        
        # Evaluate all models
        if args.all:
            print(f"Evaluating all models ({len(models)} total)...\n")
            
            results = []
            
            for model_type, exp_name, model_path in models:
                # Guess config from exp_name
                if 'baseline' in exp_name.lower():
                    config = '../configs/baseline_doorkey.yaml'
                elif 'help' in exp_name.lower():
                    config = '../configs/help_mode.yaml'
                elif 'code' in exp_name.lower():
                    config = '../configs/code_gen.yaml'
                elif 'combined' in exp_name.lower():
                    config = '../configs/combined.yaml'
                else:
                    print(f"Skipping {exp_name} (unknown config)")
                    continue
                
                print(f"\n{'='*70}")
                print(f"Evaluating: {exp_name} ({model_type})")
                print(f"{'='*70}")
                
                result = evaluate_model(str(model_path), config, args.n_episodes)
                if result:
                    result['name'] = exp_name
                    result['type'] = model_type
                    results.append(result)
            
            # Summary
            print("\n" + "="*70)
            print("SUMMARY OF ALL MODELS")
            print("="*70)
            for r in results:
                print(f"{r['name']:40s} {r['type']:5s}  {r['success_rate']*100:5.1f}%  ({r['mean_length']:5.1f} steps)")
            print("="*70)
            
    elif args.model and args.config:
        # Evaluate specific model
        evaluate_model(args.model, args.config, args.n_episodes)
    
    else:
        print("Usage:")
        print("  List models:      python eval.py --list")
        print("  Evaluate all:     python eval.py --all")
        print("  Evaluate one:     python eval.py --model output/exp/best_model --config ../configs/exp.yaml")
        print("\nExample:")
        print("  python eval.py --model output/baseline_doorkey_20241204_120000/best_model --config ../configs/baseline_doorkey.yaml")
