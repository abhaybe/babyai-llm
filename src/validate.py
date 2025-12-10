#!/usr/bin/env python3
"""
Quick validation script to test all components before running full experiments.
"""

import sys
import os

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from llm_help_mode import HelpWhenStuckWrapper
        print("  ✓ llm_help_mode imported")
    except ImportError as e:
        print(f"  ✗ Failed to import llm_help_mode: {e}")
        return False
    
    try:
        from llm_code_gen import CodeGenRewardShaper
        print("  ✓ llm_code_gen imported")
    except ImportError as e:
        print(f"  ✗ Failed to import llm_code_gen: {e}")
        return False
    
    try:
        from llm_wrapper import LLMWrapper
        print("  ✓ llm_wrapper imported")
    except ImportError as e:
        print(f"  ✗ Failed to import llm_wrapper: {e}")
        return False
    
    try:
        import minigrid
        print("  ✓ minigrid imported")
    except ImportError as e:
        print(f"  ✗ Failed to import minigrid: {e}")
        return False
    
    return True


def test_environment():
    """Test that MiniGrid environment can be created."""
    print("\nTesting environment creation...")
    
    try:
        import gymnasium as gym
        import minigrid
        
        env = gym.make("MiniGrid-DoorKey-8x8-v0")
        print("  ✓ DoorKey environment created")
        
        obs, info = env.reset()
        print(f"  ✓ Environment reset successful")
        print(f"    Observation shape: {obs['image'].shape}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"  ✗ Environment test failed: {e}")
        return False


def test_api_key():
    """Test that OpenAI API key is set."""
    print("\nTesting API key...")
    
    api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        print("  ✗ OPENAI_API_KEY not set")
        print("    Run: export OPENAI_API_KEY='sk-...'")
        return False
    
    if not api_key.startswith('sk-'):
        print("  ⚠  OPENAI_API_KEY doesn't look valid (should start with sk-)")
        return False
    
    print(f"  ✓ API key is set (starts with: {api_key[:10]}...)")
    return True


def test_llm_connection():
    """Test that we can connect to OpenAI API."""
    print("\nTesting LLM connection...")
    
    if not os.environ.get('OPENAI_API_KEY'):
        print("  ⚠  Skipping (no API key)")
        return True
    
    try:
        from llm_wrapper import LLMWrapper
        
        llm = LLMWrapper(provider="openai", model="gpt-4o-mini")
        print("  ✓ LLM wrapper created")
        
        response = llm.query("Say 'test successful'")
        print(f"  ✓ LLM query successful")
        print(f"    Response: {response[:50]}...")
        return True
        
    except Exception as e:
        print(f"  ✗ LLM test failed: {e}")
        return False


def test_configs():
    """Test that all config files exist and are valid."""
    print("\nTesting config files...")
    
    configs = [
        '../configs/baseline_doorkey.yaml',
        '../configs/help_mode.yaml',
        '../configs/code_gen.yaml',
        '../configs/combined.yaml',
    ]
    
    import yaml
    all_valid = True
    
    for config_path in configs:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required fields
            required = ['exp_name', 'env_name', 'total_timesteps']
            missing = [f for f in required if f not in config]
            
            if missing:
                print(f"  ✗ {config_path}: missing {missing}")
                all_valid = False
            else:
                print(f"  ✓ {config_path}")
                
        except Exception as e:
            print(f"  ✗ {config_path}: {e}")
            all_valid = False
    
    return all_valid


def main():
    """Run all validation tests."""
    print("="*60)
    print("VALIDATION TESTS")
    print("="*60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Environment", test_environment()))
    results.append(("API Key", test_api_key()))
    results.append(("Configs", test_configs()))
    
    # LLM connection test is optional
    if os.environ.get('OPENAI_API_KEY'):
        results.append(("LLM Connection", test_llm_connection()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s} {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("="*60)
    
    if all_passed:
        print("\n✅ All tests passed! Ready to run experiments.")
        print("\nNext steps:")
        print("  1. Set API key: export OPENAI_API_KEY='sk-...'")
        print("  2. Run baseline: python train.py --config ../configs/baseline_doorkey.yaml")
        return 0
    else:
        print("\n⚠️  Some tests failed. Fix issues before running experiments.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
