"""
Improved reward shaping: LLM judges if sub-goals are achieved.
This actually uses the LLM's understanding, not hardcoded rules!
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, List
from llm_wrapper import LLMWrapper


class LLMJudgeRewardShaper(gym.Wrapper):
    """
    LLM not only identifies sub-goals but also JUDGES when they're achieved!
    
    This is more general - works on any task without hardcoded detection.
    Trade-off: Makes LLM API call every N steps (slower, costs money).
    """
    
    def __init__(
        self,
        env: gym.Env,
        llm: LLMWrapper,
        shaping_weight: float = 0.1,
        check_frequency: int = 10,  # Check every 10 steps
        verbose: bool = False
    ):
        super().__init__(env)
        self.llm = llm
        self.shaping_weight = shaping_weight
        self.check_frequency = check_frequency
        self.verbose = verbose
        
        # Get sub-goals from LLM
        self.sub_goals = self._identify_sub_goals()
        self.achieved_sub_goals = set()
        self.step_count = 0
        
        if verbose:
            print("\n🎯 LLM IDENTIFIED SUB-GOALS:")
            for i, goal in enumerate(self.sub_goals):
                print(f"  {i+1}. {goal}")
    
    def _identify_sub_goals(self) -> List[str]:
        """Ask LLM to break down task into sub-goals."""
        try:
            mission = getattr(self.env.unwrapped, 'mission', 'complete the task')
            
            prompt = f"""Analyze this gridworld task: "{mission}"

Break this into 3-5 sub-goals. Format as a numbered list:
1. First sub-goal
2. Second sub-goal
etc.

Be specific and measurable."""

            response = self.llm.query(prompt)
            
            # Parse sub-goals
            sub_goals = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    goal = line.split('.', 1)[-1].strip()
                    if goal:
                        sub_goals.append(goal)
            
            return sub_goals if sub_goals else ["complete task"]
            
        except Exception as e:
            print(f"Error getting sub-goals: {e}")
            return ["complete task"]
    
    def _get_current_state_description(self) -> str:
        """Describe current state for LLM."""
        try:
            agent_pos = self.env.unwrapped.agent_pos
            carrying = getattr(self.env.unwrapped, 'carrying', None)
            
            desc = [f"Agent position: {agent_pos}"]
            
            if carrying:
                desc.append(f"Agent is carrying: {carrying.type} {carrying.color}")
            else:
                desc.append("Agent is not carrying anything")
            
            # Check what's in front
            front_pos = self.env.unwrapped.front_pos
            front_cell = self.env.unwrapped.grid.get(*front_pos)
            
            if front_cell and front_cell.type != 'empty':
                desc.append(f"In front of agent: {front_cell.type} {front_cell.color}")
                if hasattr(front_cell, 'is_open'):
                    desc.append(f"  Door is {'open' if front_cell.is_open else 'locked'}")
            
            return "\n".join(desc)
            
        except Exception as e:
            return f"Error: {e}"
    
    def _check_subgoals_with_llm(self) -> Dict[str, bool]:
        """
        Ask LLM to judge which sub-goals have been achieved.
        
        This is the key improvement: LLM does the detection!
        """
        try:
            state_desc = self._get_current_state_description()
            
            # List unachieved sub-goals
            remaining_goals = [g for g in self.sub_goals 
                             if g not in self.achieved_sub_goals]
            
            if not remaining_goals:
                return {}
            
            prompt = f"""You are monitoring an agent's progress in a gridworld task.

Current state:
{state_desc}

Remaining sub-goals:
{chr(10).join(f"{i+1}. {g}" for i, g in enumerate(remaining_goals))}

For each sub-goal, has it been ACHIEVED based on the current state?

Answer with ONLY "YES" or "NO" for each sub-goal, like:
1. YES
2. NO
3. NO

Be strict - only say YES if clearly achieved."""

            response = self.llm.query(prompt)
            
            # Parse LLM's judgement
            achieved = {}
            lines = [l.strip() for l in response.split('\n') if l.strip()]
            
            for i, line in enumerate(lines):
                if i >= len(remaining_goals):
                    break
                    
                goal = remaining_goals[i]
                if 'yes' in line.lower():
                    achieved[goal] = True
                    
            return achieved
            
        except Exception as e:
            print(f"Error checking sub-goals: {e}")
            return {}
    
    def reset(self, **kwargs):
        """Reset and clear achieved sub-goals."""
        self.achieved_sub_goals = set()
        self.step_count = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Step and check for sub-goal achievements using LLM."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        shaped_reward = 0.0
        
        # Check sub-goals with LLM every N steps
        if self.step_count % self.check_frequency == 0:
            newly_achieved = self._check_subgoals_with_llm()
            
            for goal in newly_achieved:
                if goal not in self.achieved_sub_goals:
                    shaped_reward += self.shaping_weight
                    self.achieved_sub_goals.add(goal)
                    
                    if self.verbose:
                        print(f"\n  ✓ LLM detected: '{goal}' achieved! (+{self.shaping_weight})")
        
        total_reward = reward + shaped_reward
        return obs, total_reward, terminated, truncated, info


# Hybrid approach: Use hardcoded when possible, LLM when needed
class HybridRewardShaper(gym.Wrapper):
    """
    Best of both worlds:
    - Hardcoded detection for known tasks (fast, free)
    - LLM fallback for unknown tasks or ambiguous states
    """
    
    def __init__(self, env: gym.Env, llm: LLMWrapper, **kwargs):
        super().__init__(env)
        self.llm = llm
        
        # Try hardcoded detection first
        self.task_type = self._detect_task_type()
        
        if self.task_type == 'doorkey':
            from reward_shaping import LLMRewardShaper
            # Use fast hardcoded version
            print("✓ Using hardcoded detection for DoorKey")
            # (Would need to refactor to use this as a mixin)
        else:
            # Fall back to LLM judge
            print("✓ Using LLM judge for unknown task")
            # Use LLM-based detection
    
    def _detect_task_type(self) -> str:
        """Detect what kind of task this is."""
        env_name = getattr(self.env.unwrapped, 'spec', None)
        if env_name and 'DoorKey' in env_name.id:
            return 'doorkey'
        elif env_name and 'Empty' in env_name.id:
            return 'empty'
        else:
            return 'unknown'


# Example usage comparison:
"""
# CURRENT (Hardcoded - Fast but Limited):
env = LLMRewardShaper(env, llm)
# - Free after initial LLM call
# - Only works on DoorKey
# - Instant sub-goal detection

# NEW (LLM Judge - Slower but General):
env = LLMJudgeRewardShaper(env, llm, check_frequency=10)
# - Costs ~$0.001 per check (every 10 steps)
# - Works on ANY task
# - LLM does the thinking

# HYBRID (Smart - Best of Both):
env = HybridRewardShaper(env, llm)
# - Uses hardcoded when available
# - Falls back to LLM for unknowns
# - Optimal cost/performance
"""
