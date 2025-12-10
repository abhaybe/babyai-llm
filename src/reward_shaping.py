"""
LLM-based reward shaping - LLM provides auxiliary rewards for sub-goals.
This actually changes what the agent learns!
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any
from llm_wrapper import LLMWrapper


class LLMRewardShaper(gym.Wrapper):
    """
    Use LLM to provide shaped rewards for intermediate progress.
    
    Example for DoorKey:
    - LLM identifies sub-goals: "find key", "pick up key", "find door", "unlock door"
    - Provides +0.1 bonus when agent makes progress toward these
    - Agent learns faster by getting intermediate feedback!
    """
    
    def __init__(
        self,
        env: gym.Env,
        llm: LLMWrapper,
        shaping_weight: float = 0.1,
        verbose: bool = False
    ):
        super().__init__(env)
        self.llm = llm
        self.shaping_weight = shaping_weight
        self.verbose = verbose
        
        # Get sub-goals from LLM at initialization
        self.sub_goals = self._identify_sub_goals()
        self.achieved_sub_goals = set()
        
        if verbose:
            print("\n🎯 LLM IDENTIFIED SUB-GOALS:")
            for i, goal in enumerate(self.sub_goals):
                print(f"  {i+1}. {goal}")
    
    def _identify_sub_goals(self) -> list:
        """Ask LLM to break down task into sub-goals."""
        try:
            # Access unwrapped environment to get mission (bypasses wrappers)
            mission = getattr(self.env.unwrapped, 'mission', 'complete the task')
            
            prompt = f"""Analyze this gridworld task: "{mission}"

Break this task into 3-5 measurable sub-goals that an agent should achieve in order.
For each sub-goal, describe what state change indicates it's achieved.

Format your response as a numbered list of sub-goals only, like:
1. Find the key
2. Pick up the key
3. Locate the door
4. Unlock the door
5. Reach the goal

Be specific and focus on observable state changes."""

            response = self.llm.query(prompt)
            
            # Parse sub-goals (simple line-based parsing)
            sub_goals = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering
                    goal = line.split('.', 1)[-1].strip()
                    if goal:
                        sub_goals.append(goal.lower())
            
            return sub_goals if sub_goals else ["complete task"]
            
        except Exception as e:
            print(f"Error getting sub-goals: {e}")
            return ["complete task"]
    
    def reset(self, **kwargs):
        """Reset and clear achieved sub-goals."""
        self.achieved_sub_goals = set()
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Step and add shaped reward for sub-goal progress."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check if any new sub-goals achieved
        shaped_reward = 0.0
        
        # For DoorKey specifically (can be generalized)
        if hasattr(self.env.unwrapped, 'carrying') and hasattr(self.env.unwrapped, 'grid'):
            carrying = self.env.unwrapped.carrying
            agent_pos = self.env.unwrapped.agent_pos
            
            # Sub-goal 1: Found/picked up key
            if carrying is not None and 'key' in str(type(carrying)).lower():
                if 'picked_key' not in self.achieved_sub_goals:
                    shaped_reward += self.shaping_weight
                    self.achieved_sub_goals.add('picked_key')
                    if self.verbose:
                        print(f"  ✓ Sub-goal achieved: Picked up key (+{self.shaping_weight})")
            
            # Sub-goal 2: At door location
            front_pos = self.env.unwrapped.front_pos
            front_cell = self.env.unwrapped.grid.get(*front_pos)
            if front_cell is not None and 'door' in str(type(front_cell)).lower():
                if 'found_door' not in self.achieved_sub_goals:
                    shaped_reward += self.shaping_weight * 0.5
                    self.achieved_sub_goals.add('found_door')
                    if self.verbose:
                        print(f"  ✓ Sub-goal achieved: Found door (+{self.shaping_weight*0.5})")
            
            # Sub-goal 3: Door is open
            if front_cell is not None and hasattr(front_cell, 'is_open') and front_cell.is_open:
                if 'opened_door' not in self.achieved_sub_goals:
                    shaped_reward += self.shaping_weight
                    self.achieved_sub_goals.add('opened_door')
                    if self.verbose:
                        print(f"  ✓ Sub-goal achieved: Opened door (+{self.shaping_weight})")
        
        # Add shaped reward to actual reward
        total_reward = reward + shaped_reward
        
        return obs, total_reward, terminated, truncated, info


class LLMCurriculumWrapper(gym.Wrapper):
    """
    LLM suggests which difficulty level to train on based on recent performance.
    Automatically adjusts task difficulty!
    """
    
    def __init__(self, base_env_name: str, llm: LLMWrapper):
        self.base_env_name = base_env_name
        self.llm = llm
        self.current_difficulty = "easy"
        self.episode_count = 0
        self.success_history = []
        
        # Start with easiest version
        env = gym.make(self._get_env_name("easy"))
        super().__init__(env)
    
    def _get_env_name(self, difficulty: str) -> str:
        """Map difficulty to actual environment name."""
        # Example for DoorKey
        mapping = {
            "easy": "MiniGrid-DoorKey-5x5-v0",
            "medium": "MiniGrid-DoorKey-8x8-v0",
            "hard": "MiniGrid-DoorKey-16x16-v0"
        }
        return mapping.get(difficulty, mapping["medium"])
    
    def _check_curriculum_progress(self):
        """Ask LLM if agent is ready for harder difficulty."""
        if len(self.success_history) < 20:
            return  # Need more episodes
        
        recent_success = np.mean(self.success_history[-20:])
        
        if recent_success > 0.8 and self.current_difficulty == "easy":
            # Ask LLM if we should progress
            prompt = f"""An RL agent training on {self._get_env_name('easy')} has:
- Success rate: {recent_success:.1%} over last 20 episodes
- Current difficulty: {self.current_difficulty}

Should we increase difficulty to 'medium'? Answer YES or NO and briefly explain."""
            
            response = self.llm.query(prompt).lower()
            
            if 'yes' in response:
                print(f"\n📈 LLM CURRICULUM: Increasing difficulty to MEDIUM")
                self.current_difficulty = "medium"
                self.env = gym.make(self._get_env_name("medium"))
                self.success_history = []  # Reset for new difficulty
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if terminated or truncated:
            self.success_history.append(1 if reward > 0 else 0)
            self.episode_count += 1
            
            if self.episode_count % 20 == 0:
                self._check_curriculum_progress()
        
        return obs, reward, terminated, truncated, info


# Example usage:
"""
from reward_shaping import LLMRewardShaper
from llm_wrapper import LLMWrapper

# Create environment with LLM reward shaping
llm = LLMWrapper(provider="openai", model="gpt-3.5-turbo")
env = gym.make("MiniGrid-DoorKey-8x8-v0")
env = LLMRewardShaper(env, llm, shaping_weight=0.1, verbose=True)

# Now when agent picks up key, it gets +0.1 bonus!
# This speeds up learning significantly.
"""
