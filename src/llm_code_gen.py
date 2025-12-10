"""
Code Generation: LLM writes reward shaping functions.

Key features:
- LLM writes executable Python code
- Safe exec() with sandboxing
- Provides examples and templates
- Validates generated code
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, Callable
import ast
import re
from llm_wrapper import LLMWrapper


class CodeGenRewardShaper(gym.Wrapper):
    """
    LLM generates reward shaping code that runs during training.
    
    At initialization, LLM writes a reward shaping function.
    This function is safely executed every step to add auxiliary rewards.
    """
    
    def __init__(
        self,
        env: gym.Env,
        llm: LLMWrapper,
        shaping_weight: float = 1.0,
        verbose: bool = True
    ):
        super().__init__(env)
        self.llm = llm
        self.shaping_weight = shaping_weight
        self.verbose = verbose
        
        # Generate reward function
        self.reward_function = self._generate_reward_function()
        
        # Tracking
        self.achieved_milestones = set()
        self.prev_state = None
    
    def _get_code_generation_prompt(self) -> str:
        """
        Create prompt for LLM to generate reward shaping code.
        Includes examples and constraints.
        """
        mission = getattr(self.env.unwrapped, 'mission', 'reach the goal')
        
        prompt = f"""You are writing a reward shaping function for reinforcement learning.

TASK: {mission}

Your function will be called every step to provide auxiliary rewards for progress.

AVAILABLE STATE INFORMATION:
- env: The environment object
- env.unwrapped.agent_pos: tuple (x, y) of agent position
- env.unwrapped.carrying: Object agent is holding (None if nothing)
- env.unwrapped.grid.get(x, y): Get object at position
- prev_carrying: What agent carried in previous step (None if nothing)

AVAILABLE OBJECT ATTRIBUTES:
- obj.type: string like 'key', 'door', 'goal', 'wall'
- obj.color: string like 'yellow', 'red', 'blue'
- obj.is_open: boolean (for doors)
- obj.is_locked: boolean (for doors)

EXAMPLE 1 - Simple progress tracking:
```python
def shaped_reward(env, prev_carrying):
    reward = 0.0
    
    # Reward for picking up an object
    if env.unwrapped.carrying and not prev_carrying:
        reward += 0.1
    
    return reward
```

EXAMPLE 2 - Task-specific rewards:
```python
def shaped_reward(env, prev_carrying):
    reward = 0.0
    
    # Reward for picking up key specifically
    carrying = env.unwrapped.carrying
    if carrying and not prev_carrying:
        if hasattr(carrying, 'type') and carrying.type == 'key':
            reward += 0.15
    
    # Reward for opening door
    grid = env.unwrapped.grid
    for i in range(grid.width):
        for j in range(grid.height):
            cell = grid.get(i, j)
            if cell and hasattr(cell, 'type') and cell.type == 'door':
                if hasattr(cell, 'is_open') and cell.is_open:
                    reward += 0.15
    
    return reward
```

CONSTRAINTS:
1. Function MUST be named 'shaped_reward'
2. Function MUST take parameters: (env, prev_carrying)
3. Function MUST return a float
4. Do NOT import any libraries
5. Do NOT use file I/O or network operations
6. Keep rewards between 0.0 and 1.0
7. Use try/except for safety

YOUR TASK:
Write a shaped_reward function for the task "{mission}".
Identify 2-4 sub-goals and reward progress toward them.

RESPOND WITH ONLY THE PYTHON FUNCTION, NO EXPLANATIONS:

```python
def shaped_reward(env, prev_carrying):
    # Your code here
    pass
```

Your code:"""

        return prompt
    
    def _generate_reward_function(self) -> Optional[Callable]:
        """
        Ask LLM to generate reward shaping code, then safely execute it.
        """
        try:
            prompt = self._get_code_generation_prompt()
            response = self.llm.query(prompt)
            
            if self.verbose:
                print(f"\n{'='*60}")
                print("LLM CODE GENERATION")
                print(f"{'='*60}")
                # Extract just the function, not the full prompt/response
            
            # Extract code from response
            code = self._extract_code_from_response(response)
            
            if code is None:
                if self.verbose:
                    print("⚠️ Could not extract code from LLM response")
                    print(f"{'='*60}\n")
                return self._default_reward_function()
            
            if self.verbose:
                print("Generated function:")
                print(code[:500] + ("..." if len(code) > 500 else ""))  # Truncate if too long
                print(f"{'='*60}\n")
            
            if code is None:
                if self.verbose:
                    print("⚠️ Could not extract code, using default reward function")
                return self._default_reward_function()
            
            # Validate and compile code
            reward_func = self._safe_exec_code(code)
            
            if reward_func is None:
                if self.verbose:
                    print("⚠️ Code validation failed, using default reward function")
                return self._default_reward_function()
            
            if self.verbose:
                print("✓ LLM-generated reward function loaded successfully!\n")
            
            return reward_func
            
        except Exception as e:
            if self.verbose:
                print(f"Error generating reward function: {e}")
                print("Using default reward function")
            return self._default_reward_function()
    
    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        
        # Try to find code in markdown blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0]
        
        # Try to find code without markdown
        code_blocks = re.findall(r'```\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0]
        
        # Try to find function definition directly
        if 'def shaped_reward' in response:
            # Extract from 'def' to end
            start_idx = response.find('def shaped_reward')
            if start_idx != -1:
                code = response[start_idx:]
                # Try to find reasonable end point
                return code
        
        return None
    
    def _safe_exec_code(self, code: str) -> Optional[Callable]:
        """
        Safely execute LLM-generated code with validation.
        """
        try:
            # Parse to AST for validation
            tree = ast.parse(code)
            
            # Security checks
            for node in ast.walk(tree):
                # Disallow imports
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if self.verbose:
                        print("⚠️ Generated code contains imports - rejected")
                    return None
                
                # Disallow file operations
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['open', 'exec', 'eval', '__import__']:
                            if self.verbose:
                                print(f"⚠️ Generated code contains {node.func.id} - rejected")
                            return None
            
            # Create restricted namespace
            safe_globals = {
                '__builtins__': {
                    'hasattr': hasattr,
                    'getattr': getattr,
                    'isinstance': isinstance,
                    'range': range,
                    'len': len,
                    'float': float,
                    'int': int,
                    'str': str,
                    'True': True,
                    'False': False,
                    'None': None,
                },
            }
            
            safe_locals = {}
            
            # Execute code
            exec(code, safe_globals, safe_locals)
            
            # Get the function
            if 'shaped_reward' not in safe_locals:
                if self.verbose:
                    print("⚠️ Generated code doesn't define 'shaped_reward' function")
                return None
            
            reward_func = safe_locals['shaped_reward']
            
            # Test the function with dummy inputs
            try:
                test_result = reward_func(self.env, None)
                if not isinstance(test_result, (int, float)):
                    if self.verbose:
                        print(f"⚠️ Function returned {type(test_result)}, expected number")
                    return None
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Function test failed: {e}")
                return None
            
            return reward_func
            
        except SyntaxError as e:
            if self.verbose:
                print(f"⚠️ Syntax error in generated code: {e}")
            return None
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Error executing generated code: {e}")
            return None
    
    def _default_reward_function(self) -> Callable:
        """Fallback reward function if code generation fails."""
        def default_shaped_reward(env, prev_carrying):
            reward = 0.0
            
            # Simple: reward for picking up anything
            try:
                if env.unwrapped.carrying and not prev_carrying:
                    reward += 0.1
            except:
                pass
            
            return reward
        
        return default_shaped_reward
    
    def reset(self, **kwargs):
        """Reset state tracking."""
        self.prev_state = None
        self.achieved_milestones = set()
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Step with LLM-generated reward shaping."""
        # Store previous carrying state
        prev_carrying = None
        try:
            prev_carrying = self.env.unwrapped.carrying
        except:
            pass
        
        # Take step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate shaped reward using LLM-generated function
        shaped_reward = 0.0
        try:
            shaped_reward = self.reward_function(self.env, prev_carrying)
            shaped_reward = float(shaped_reward)
            
            # Clamp to reasonable range
            shaped_reward = max(0.0, min(1.0, shaped_reward))
            
            # Apply weight
            shaped_reward *= self.shaping_weight
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Error in reward function: {e}")
            shaped_reward = 0.0
        
        # Add to total reward
        total_reward = reward + shaped_reward
        
        if shaped_reward > 0 and self.verbose:
            print(f"  → Shaped reward: +{shaped_reward:.3f}")
        
        # Add info
        info['shaped_reward'] = shaped_reward
        info['original_reward'] = reward
        
        return obs, total_reward, terminated, truncated, info


def test_code_generation():
    """Test the code generation with a sample environment."""
    import minigrid
    from llm_wrapper import LLMWrapper
    
    print("Testing Code Generation Reward Shaping")
    print("="*60)
    
    # Create environment
    env = gym.make("MiniGrid-DoorKey-8x8-v0")
    
    # Create LLM
    llm = LLMWrapper(provider="openai", model="gpt-3.5-turbo")
    
    # Wrap with code gen
    env = CodeGenRewardShaper(env, llm, verbose=True)
    
    print("\nRunning test episode...")
    obs, info = env.reset()
    
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        if reward > 0:
            print(f"Step {step}: reward = {reward:.3f}")
        
        if done or truncated:
            break
    
    print("\n✓ Test complete!")


if __name__ == "__main__":
    test_code_generation()
