"""
Help-When-Stuck: LLM provides single action when agent is stuck.

Key features:
- Only calls LLM when stuck (no reward for N steps)
- LLM outputs single action with strict formatting
- Easy parsing with fallbacks
- Tells LLM what actions are available
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, List
from llm_wrapper import LLMWrapper


class HelpWhenStuckWrapper(gym.Wrapper):
    """
    LLM provides help when agent is stuck.
    
    When agent hasn't received reward for `stuck_threshold` steps,
    ask LLM for a single action suggestion.
    """
    
    def __init__(
        self,
        env: gym.Env,
        llm: LLMWrapper,
        stuck_threshold: int = 50,
        max_help_per_episode: int = 3,
        verbose: bool = True
    ):
        super().__init__(env)
        self.llm = llm
        self.stuck_threshold = stuck_threshold
        self.max_help_per_episode = max_help_per_episode
        self.verbose = verbose
        
        # Tracking
        self.steps_without_reward = 0
        self.help_count_this_episode = 0
        self.recent_actions = []
        self.total_llm_calls = 0
        self.successful_helps = 0
        self.failed_helps = 0
        
        # Action mapping
        self.action_names = {
            0: "turn_left",
            1: "turn_right",
            2: "move_forward",
            3: "pickup",
            4: "drop",
            5: "toggle",
            6: "done"
        }
        self.name_to_action = {v: k for k, v in self.action_names.items()}
    
    def _get_available_actions(self) -> List[str]:
        """
        Determine which actions make sense in current state.
        This helps LLM give better suggestions.
        """
        available = ["turn_left", "turn_right", "move_forward"]
        
        # Can pickup if object is in front
        try:
            front_pos = self.env.unwrapped.front_pos
            front_cell = self.env.unwrapped.grid.get(*front_pos)
            
            if front_cell and front_cell.can_pickup():
                available.append("pickup")
            
            if front_cell and front_cell.type == 'door':
                available.append("toggle")
                
        except:
            pass
        
        return available
    
    def _get_state_description(self) -> str:
        """Get description of current state for LLM."""
        try:
            agent_pos = self.env.unwrapped.agent_pos
            agent_dir = self.env.unwrapped.agent_dir
            carrying = self.env.unwrapped.carrying
            
            # Direction mapping
            dir_names = ["right", "down", "left", "up"]
            dir_name = dir_names[agent_dir]
            
            desc = [
                f"Position: {agent_pos}",
                f"Facing: {dir_name}",
            ]
            
            # What's being carried
            if carrying:
                desc.append(f"Carrying: {carrying.type} ({carrying.color})")
            else:
                desc.append("Carrying: nothing")
            
            # What's in front
            front_pos = self.env.unwrapped.front_pos
            front_cell = self.env.unwrapped.grid.get(*front_pos)
            
            if front_cell:
                if front_cell.type == 'wall':
                    desc.append("In front: wall (blocked)")
                elif front_cell.type == 'door':
                    status = "open" if front_cell.is_open else "locked"
                    desc.append(f"In front: {front_cell.color} door ({status})")
                else:
                    desc.append(f"In front: {front_cell.type} ({front_cell.color})")
            else:
                desc.append("In front: empty space")
            
            return "\n".join(desc)
            
        except Exception as e:
            return f"Position: {agent_pos}, Error getting details: {e}"
    
    def _ask_llm_for_help(self) -> Optional[int]:
        """
        Ask LLM for help. Returns action index or None.
        """
        try:
            state_desc = self._get_state_description()
            available_actions = self._get_available_actions()
            recent_actions_str = ", ".join([self.action_names[a] for a in self.recent_actions[-10:]])
            mission = getattr(self.env.unwrapped, 'mission', 'reach the goal')
            
            # Construct prompt with strict formatting
            prompt = f"""You are helping an agent stuck in a gridworld.

TASK: {mission}

CURRENT STATE:
{state_desc}

RECENT ACTIONS (last 10): {recent_actions_str}

STUCK: Agent has taken {self.steps_without_reward} steps without reward.

AVAILABLE ACTIONS: {', '.join(available_actions)}

RESPOND WITH EXACTLY ONE ACTION FROM THE LIST ABOVE.
Format your response as:
ACTION: [action_name]
REASON: [brief reason]

Example response:
ACTION: turn_right
REASON: You're facing a wall, turn to explore

Your response:"""

            response = self.llm.query(prompt)
            self.total_llm_calls += 1
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"LLM HELP REQUEST #{self.total_llm_calls} (stuck for {self.steps_without_reward} steps)")
                print(f"{'='*60}")
                # Don't print full response - too slow!
                # Just print what action was suggested
            
            # Parse response
            action = self._parse_action_from_response(response, available_actions)
            
            return action
            
        except Exception as e:
            if self.verbose:
                print(f"Error asking LLM for help: {e}")
            return None
    
    def _parse_action_from_response(self, response: str, available_actions: List[str]) -> Optional[int]:
        """
        Parse LLM response to extract action.
        Multiple fallback strategies for robustness.
        """
        response_lower = response.lower()
        
        # Strategy 1: Look for "ACTION: [action_name]" format
        if "action:" in response_lower:
            lines = response.split('\n')
            for line in lines:
                if line.lower().startswith('action:'):
                    action_str = line.split(':', 1)[1].strip().lower()
                    # Remove punctuation
                    action_str = action_str.replace('.', '').replace(',', '')
                    
                    if action_str in self.name_to_action:
                        return self.name_to_action[action_str]
        
        # Strategy 2: Look for any available action mentioned
        for action_name in available_actions:
            if action_name.lower() in response_lower:
                return self.name_to_action[action_name]
        
        # Strategy 3: Look for common variations
        variations = {
            'left': 'turn_left',
            'right': 'turn_right',
            'forward': 'move_forward',
            'ahead': 'move_forward',
            'pick': 'pickup',
            'grab': 'pickup',
            'get': 'pickup',
            'open': 'toggle',
            'unlock': 'toggle',
        }
        
        for keyword, action_name in variations.items():
            if keyword in response_lower and action_name in available_actions:
                return self.name_to_action[action_name]
        
        # Failed to parse
        if self.verbose:
            print(f"⚠️ Could not parse action from: {response}")
        
        return None
    
    def reset(self, **kwargs):
        """Reset tracking variables."""
        self.steps_without_reward = 0
        self.help_count_this_episode = 0
        self.recent_actions = []
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """
        Step with potential LLM help when stuck.
        """
        # Check if we should ask for help BEFORE taking action
        should_ask_help = (
            self.steps_without_reward >= self.stuck_threshold and
            self.help_count_this_episode < self.max_help_per_episode
        )
        
        if should_ask_help:
            llm_action = self._ask_llm_for_help()
            
            if llm_action is not None:
                if self.verbose:
                    print(f"  → LLM suggests: {self.action_names[llm_action]}")
                    print(f"{'='*60}\n")
                action = llm_action
                self.help_count_this_episode += 1
                
                # Reset counter after getting help (cooldown period)
                self.steps_without_reward = 0
                
                # Track if help was successful (next 10 steps)
                self._tracking_help = True
                self._help_steps = 0
                self._help_initial_reward = 0
            else:
                # Failed to parse LLM response, still reset counter to avoid spam
                if self.verbose:
                    print(f"  ⚠️  Could not parse LLM response")
                    print(f"{'='*60}\n")
                self.steps_without_reward = 0
                self.help_count_this_episode += 1  # Count failed attempts too
        
        # Take step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track action
        self.recent_actions.append(action)
        if len(self.recent_actions) > 20:
            self.recent_actions.pop(0)
        
        # Update stuck counter
        if reward > 0:
            self.steps_without_reward = 0
            
            # If we were tracking help effectiveness
            if hasattr(self, '_tracking_help') and self._tracking_help:
                self.successful_helps += 1
                if self.verbose:
                    print(f"  ✓ LLM help led to reward within {self._help_steps} steps!")
                self._tracking_help = False
        else:
            self.steps_without_reward += 1
        
        # Track help effectiveness
        if hasattr(self, '_tracking_help') and self._tracking_help:
            self._help_steps += 1
            if self._help_steps > 10:
                self.failed_helps += 1
                self._tracking_help = False
        
        # Add stats to info
        info['llm_help_calls'] = self.total_llm_calls
        info['llm_help_success'] = self.successful_helps
        info['llm_help_failed'] = self.failed_helps
        
        return obs, reward, terminated, truncated, info
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about LLM help."""
        total_attempts = self.successful_helps + self.failed_helps
        success_rate = self.successful_helps / total_attempts if total_attempts > 0 else 0
        
        return {
            'total_llm_calls': self.total_llm_calls,
            'successful_helps': self.successful_helps,
            'failed_helps': self.failed_helps,
            'success_rate': success_rate
        }
