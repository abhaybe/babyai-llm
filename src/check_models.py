#!/usr/bin/env python3
"""
Diagnostic script to check if models were saved correctly.
Run this to debug 0% evaluation issues.
"""

import os
from pathlib import Path
import numpy as np

def check_experiment(exp_dir):
    """Check a single experiment directory."""
    print("="*70)
    print(f"Checking: {exp_dir.name}")
    print("="*70)
    
    # Check config
    config_file = exp_dir / "config.yaml"
    if config_file.exists():
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)
        print(f"Environment: {config.get('env_name', 'Unknown')}")
        print(f"Total timesteps: {config.get('total_timesteps', 'Unknown')}")
    else:
        print("⚠️  No config.yaml found")
    
    # Check best_model
    best_model_dir = exp_dir / "best_model"
    if best_model_dir.exists():
        files = list(best_model_dir.iterdir())
        print(f"\n✓ best_model/ exists with {len(files)} files:")
        for f in files:
            size = f.stat().st_size / (1024*1024)
            print(f"  - {f.name} ({size:.2f} MB)")
    else:
        print("\n❌ best_model/ NOT FOUND")
        print("   This means EvalCallback didn't save anything!")
        print("   Possible reasons:")
        print("   - Training didn't run long enough")
        print("   - Agent never achieved good performance")
        print("   - eval_freq too high (never evaluated)")
    
    # Check eval logs
    eval_dir = exp_dir / "eval"
    if eval_dir.exists():
        eval_files = list(eval_dir.glob("evaluations.npz"))
        if eval_files:
            print(f"\n✓ Evaluation logs exist")
            
            # Load and show best performance
            try:
                data = np.load(eval_files[0])
                results = data['results']
                timesteps = data['timesteps']
                
                # Results shape: (n_evals, n_eval_episodes)
                mean_rewards = results.mean(axis=1)
                best_idx = mean_rewards.argmax()
                best_reward = mean_rewards[best_idx]
                best_timestep = timesteps[best_idx]
                
                print(f"  Best performance during training:")
                print(f"    Mean reward: {best_reward:.3f}")
                print(f"    At timestep: {best_timestep}")
                print(f"  Full reward history: {mean_rewards}")
                
                if best_reward < 0.1:
                    print("\n  ⚠️  WARNING: Best reward is very low!")
                    print("     Agent didn't learn much during training")
                    print("     Try:")
                    print("     - Training longer (increase total_timesteps)")
                    print("     - Checking if environment is too hard")
                    print("     - Verifying hyperparameters")
                
            except Exception as e:
                print(f"  Error reading eval data: {e}")
        else:
            print(f"\n⚠️  eval/ exists but no evaluations.npz")
    else:
        print("\n❌ No eval/ directory")
        print("   EvalCallback didn't run at all!")
    
    # Check final model
    final_model = exp_dir / "final_model.zip"
    if final_model.exists():
        size = final_model.stat().st_size / (1024*1024)
        print(f"\n✓ final_model.zip exists ({size:.2f} MB)")
    else:
        print("\n⚠️  No final_model.zip (training didn't complete?)")
    
    # Check tensorboard logs
    tb_dir = exp_dir / "tensorboard"
    if tb_dir.exists():
        print(f"\n✓ TensorBoard logs exist")
        print(f"  View with: tensorboard --logdir={tb_dir}")
    
    print()


def main():
    """Check all experiments."""
    output_dir = Path('./output')
    
    if not output_dir.exists():
        print("❌ No ./output directory found")
        print("   You haven't trained any models yet!")
        return
    
    experiments = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    
    if not experiments:
        print("❌ No experiments found in ./output/")
        return
    
    print(f"\nFound {len(experiments)} experiment(s)\n")
    
    for exp_dir in experiments:
        check_experiment(exp_dir)
        print()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for exp_dir in experiments:
        best_model = exp_dir / "best_model"
        eval_dir = exp_dir / "eval"
        
        status = "✓" if best_model.exists() else "❌"
        
        # Try to get best reward
        best_reward_str = "N/A"
        if eval_dir.exists():
            eval_files = list(eval_dir.glob("evaluations.npz"))
            if eval_files:
                try:
                    data = np.load(eval_files[0])
                    results = data['results']
                    mean_rewards = results.mean(axis=1)
                    best_reward = mean_rewards.max()
                    best_reward_str = f"{best_reward:.3f}"
                except:
                    pass
        
        print(f"{status} {exp_dir.name:45s} Best reward: {best_reward_str}")
    
    print("="*70)
    print("\nIf best_model is missing (❌), the model wasn't saved!")
    print("If best reward is < 0.1, the agent didn't learn anything.")
    print("\nFor 0% evaluation, most likely:")
    print("  1. Model trained on different environment than evaluation")
    print("  2. Training didn't run long enough")
    print("  3. Task is too hard for the hyperparameters")


if __name__ == "__main__":
    import sys
    # os.chdir('src')
    main()
