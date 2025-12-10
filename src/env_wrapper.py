"""
Environment wrapper for BabyAI with LLM integration.
Handles different modes: baseline, planning, and help-when-stuck.
"""

import gymnasium as gym
import minigrid  # Required for MiniGrid environments to register
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

from llm_wrapper import LLMWrapper, BabyAILLMPrompts


class LLMMode(Enum):
    """Different modes of LLM integration."""
    NONE = "none"  # Baseline - no LLM
    PLANNING = "planning"  # LLM provides plan at episode start
    HELP = "help"  # LLM helps when agent is stuck


class BabyAILLMWrapper(gym.Wrapper):
    """
    Wrapper for BabyAI environment that integrates LLM guidance.
    
    This wrapper can operate in different modes:
    - NONE: Standard RL without LLM (baseline)
    - PLANNING: LLM provides high-level plan at episode start
    - HELP: LLM provides advice when agent hasn't received reward for N steps
    """
    
    def __init__(
        self,
        env: gym.Env,
        llm: Optional[LLMWrapper] = None,
        mode: LLMMode = LLMMode.NONE,
        help_threshold: int = 50,  # Steps without reward before asking for help
        max_help_requests: int = 3,  # Max help requests per episode
        verbose: bool = True
    ):
        """
        Initialize the wrapper.
        
        Args:
            env: BabyAI environment
            llm: LLM wrapper (required if mode != NONE)
            mode: LLM integration mode
            help_threshold: Steps without reward before requesting help
            max_help_requests: Maximum help requests allowed per episode
            verbose: Whether to print LLM advice
        """
        super().__init__(env)
        self.llm = llm
        self.mode = mode
        self.help_threshold = help_threshold
        self.max_help_requests = max_help_requests
        self.verbose = verbose
        
        # Episode tracking
        self.steps_since_reward = 0
        self.help_requests_this_episode = 0
        self.recent_actions = []
        self.episode_plan = None
        self.total_llm_queries = 0
        
        # Validate LLM is provided when needed
        if self.mode != LLMMode.NONE and llm is None:
            raise ValueError(f"LLM wrapper required for mode {mode}")
    
    def reset(self, **kwargs):
        """Reset environment and get initial LLM guidance if needed."""
        result = self.env.reset(**kwargs)
        
        # Handle both old gym (returns obs) and new gymnasium (returns obs, info)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        
        # Reset tracking
        self.steps_since_reward = 0
        self.help_requests_this_episode = 0
        self.recent_actions = []
        self.episode_plan = None
        
        # Get initial plan if in planning mode
        if self.mode == LLMMode.PLANNING and self.llm is not None:
            self.episode_plan = self._get_initial_plan()
            if self.verbose and self.episode_plan:
                print("\n" + "="*60)
                print("LLM PLAN:")
                print(self.episode_plan)
                print("="*60 + "\n")
        
        return (obs, info) if isinstance(result, tuple) else obs
    
    def step(self, action):
        """Step environment and provide LLM help if needed."""
        result = self.env.step(action)
        
        # Handle both old gym and new gymnasium API
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        
        # Track actions
        action_name = self._action_to_string(action)
        self.recent_actions.append(action_name)
        
        # Update reward tracking
        if reward > 0:
            self.steps_since_reward = 0
        else:
            self.steps_since_reward += 1
        
        # Check if we should ask for help
        if (self.mode == LLMMode.HELP and 
            self.llm is not None and
            self.steps_since_reward >= self.help_threshold and
            self.help_requests_this_episode < self.max_help_requests and
            not done):
            
            help_advice = self._get_help()
            if self.verbose and help_advice:
                print("\n" + "="*60)
                print("LLM HELP (stuck for {} steps):".format(self.steps_since_reward))
                print(help_advice)
                print("="*60 + "\n")
            
            self.help_requests_this_episode += 1
            # Reset counter after getting help
            self.steps_since_reward = 0
        
        # Return in format compatible with both APIs
        if len(result) == 4:
            return obs, reward, done, info
        else:
            return obs, reward, terminated, truncated, info
    
    def _get_initial_plan(self) -> Optional[str]:
        """Get high-level plan from LLM at episode start."""
        try:
            instruction = self.env.unwrapped.mission
            grid_desc = self._get_grid_description()
            
            prompt = BabyAILLMPrompts.get_planning_prompt(instruction, grid_desc)
            plan = self.llm.query(prompt)
            self.total_llm_queries += 1
            
            return plan
        except Exception as e:
            print(f"Error getting initial plan: {e}")
            return None
    
    def _get_help(self) -> Optional[str]:
        """Get help from LLM when agent is stuck."""
        try:
            instruction = self.env.unwrapped.mission
            grid_desc = self._get_grid_description()
            agent_pos = self.env.unwrapped.agent_pos
            
            prompt = BabyAILLMPrompts.get_help_prompt(
                instruction=instruction,
                grid_description=grid_desc,
                agent_position=agent_pos,
                steps_without_reward=self.steps_since_reward,
                recent_actions=self.recent_actions
            )
            
            help_text = self.llm.query(prompt)
            self.total_llm_queries += 1
            
            return help_text
        except Exception as e:
            print(f"Error getting help: {e}")
            return None
    
    def _get_grid_description(self) -> str:
        """Generate a text description of the current grid state."""
        try:
            # Get grid from unwrapped environment (bypasses wrappers)
            grid = self.env.unwrapped.grid
            agent_pos = self.env.unwrapped.agent_pos
            agent_dir = self.env.unwrapped.agent_dir
            
            # Direction mapping
            dir_map = {0: "right", 1: "down", 2: "left", 3: "up"}
            
            description = []
            description.append(f"Grid size: {grid.width}x{grid.height}")
            description.append(f"Agent position: {agent_pos}, facing {dir_map[agent_dir]}")
            
            # List visible objects
            objects = []
            for i in range(grid.width):
                for j in range(grid.height):
                    cell = grid.get(i, j)
                    if cell is not None and not cell.type == 'wall':
                        if cell.type != 'empty':
                            objects.append(f"- {cell.type} {cell.color} at ({i}, {j})")
            
            if objects:
                description.append("Objects in environment:")
                description.extend(objects[:10])  # Limit to avoid long prompts
            
            return "\n".join(description)
        except Exception as e:
            return f"Error getting grid description: {e}"
    
    def _action_to_string(self, action: int) -> str:
        """Convert action index to readable string."""
        action_map = {
            0: "left",
            1: "right", 
            2: "forward",
            3: "pickup",
            4: "drop",
            5: "toggle",
            6: "done"
        }
        return action_map.get(action, f"action_{action}")
    
    def get_llm_stats(self) -> Dict[str, Any]:
        """Get statistics about LLM usage."""
        stats = {
            "total_queries": self.total_llm_queries,
            "mode": self.mode.value
        }
        
        if self.llm:
            stats.update(self.llm.get_stats())
        
        return stats


def make_babyai_env(
    env_name: str = "MiniGrid-DoorKey-5x5-v0",
    llm_mode: str = "none",
    llm_config: Optional[Dict] = None,
    **kwargs
) -> BabyAILLMWrapper:
    """
    Create a MiniGrid environment with optional LLM integration.
    
    Args:
        env_name: Name of MiniGrid environment
                  Examples: "MiniGrid-DoorKey-5x5-v0", "MiniGrid-Empty-8x8-v0"
        llm_mode: One of "none", "planning", "help"
        llm_config: Configuration dict for LLM (provider, model, etc.)
        **kwargs: Additional arguments for BabyAILLMWrapper
        
    Returns:
        Wrapped MiniGrid environment
    """
    # Create base environment
    env = gym.make(env_name)
    
    # Parse mode
    mode = LLMMode(llm_mode.lower())
    
    # Create LLM if needed
    llm = None
    if mode != LLMMode.NONE:
        if llm_config is None:
            llm_config = {}
        llm = LLMWrapper(**llm_config)
    
    # Wrap environment
    wrapped_env = BabyAILLMWrapper(
        env=env,
        llm=llm,
        mode=mode,
        **kwargs
    )
    
    return wrapped_env
