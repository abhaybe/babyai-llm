"""
LLM Wrapper for providing guidance to RL agents.
Supports multiple providers and includes caching to save API costs during development.
"""

import os
import json
import hashlib
from typing import Optional, Dict, List, Any
from pathlib import Path


class LLMWrapper:
    """Wrapper for LLM APIs with caching support."""
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4-turbo-preview",
        cache_dir: str = "./llm_cache",
        temperature: float = 0.7,
        max_tokens: int = 500,
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM wrapper.
        
        Args:
            provider: "openai", "anthropic", or "custom"
            model: Model name to use
            cache_dir: Directory to cache LLM responses
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            api_key: API key (if None, reads from environment)
        """
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Setup cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize client
        self.client = self._init_client(api_key)
        
        # Track API usage
        self.num_calls = 0
        self.cache_hits = 0
        
    def _init_client(self, api_key: Optional[str]):
        """Initialize the appropriate LLM client."""
        if self.provider == "openai":
            import openai
            if api_key:
                openai.api_key = api_key
            elif "OPENAI_API_KEY" not in os.environ:
                print("Warning: No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
            return openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
            
        elif self.provider == "anthropic":
            import anthropic
            if api_key:
                return anthropic.Anthropic(api_key=api_key)
            elif "ANTHROPIC_API_KEY" in os.environ:
                return anthropic.Anthropic()
            else:
                print("Warning: No Anthropic API key found. Set ANTHROPIC_API_KEY environment variable.")
                return anthropic.Anthropic(api_key=api_key or "")
                
        elif self.provider == "custom":
            # Placeholder for custom API (like Tinker API)
            print("Using custom provider - you'll need to implement _call_custom_api()")
            return None
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key from prompt."""
        # Include model and temperature in cache key
        cache_string = f"{self.model}_{self.temperature}_{prompt}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Try to get cached response."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                self.cache_hits += 1
                return data['response']
        return None
    
    def _save_to_cache(self, cache_key: str, prompt: str, response: str):
        """Save response to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump({
                'prompt': prompt,
                'response': response,
                'model': self.model,
                'temperature': self.temperature
            }, f, indent=2)
    
    def query(self, prompt: str, use_cache: bool = True) -> str:
        """
        Query the LLM with a prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            use_cache: Whether to use cached responses
            
        Returns:
            The LLM's response as a string
        """
        # Check cache
        cache_key = self._get_cache_key(prompt)
        if use_cache:
            cached = self._get_cached_response(cache_key)
            if cached:
                return cached
        
        # Make API call
        self.num_calls += 1
        
        if self.provider == "openai":
            response = self._call_openai(prompt)
        elif self.provider == "anthropic":
            response = self._call_anthropic(prompt)
        elif self.provider == "custom":
            response = self._call_custom_api(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        # Cache response
        if use_cache:
            self._save_to_cache(cache_key, prompt, response)
        
        return response
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def _call_custom_api(self, prompt: str) -> str:
        """
        Placeholder for custom API implementation.
        Replace this with your actual API call (e.g., Tinker API).
        """
        raise NotImplementedError(
            "Custom API not implemented. "
            "Replace this method with your Tinker API or other provider."
        )
    
    def get_stats(self) -> Dict[str, int]:
        """Get usage statistics."""
        return {
            "total_calls": self.num_calls,
            "cache_hits": self.cache_hits,
            "actual_api_calls": self.num_calls - self.cache_hits
        }


class BabyAILLMPrompts:
    """Prompt templates for BabyAI environment."""
    
    @staticmethod
    def get_planning_prompt(instruction: str, grid_description: str) -> str:
        """
        Prompt for overall episode planning.
        
        Args:
            instruction: The language instruction (e.g., "go to the red ball")
            grid_description: Description of the current environment
        """
        return f"""You are helping an agent navigate a gridworld environment to complete tasks.

Task: {instruction}

Environment:
{grid_description}

Provide a high-level plan (3-5 steps) for completing this task. Be specific about what the agent should look for and do.
Keep your response concise and actionable."""
    
    @staticmethod
    def get_help_prompt(
        instruction: str,
        grid_description: str,
        agent_position: tuple,
        steps_without_reward: int,
        recent_actions: List[str]
    ) -> str:
        """
        Prompt for when agent is stuck and needs help.
        
        Args:
            instruction: The language instruction
            grid_description: Current environment state
            agent_position: Current (x, y) position
            steps_without_reward: How many steps since last reward
            recent_actions: List of recent actions taken
        """
        recent_actions_str = ", ".join(recent_actions[-10:]) if recent_actions else "None"
        
        return f"""You are helping an agent that appears to be stuck in a gridworld navigation task.

Task: {instruction}

Current situation:
- Agent position: {agent_position}
- Steps without reward: {steps_without_reward}
- Recent actions: {recent_actions_str}

Environment:
{grid_description}

The agent seems stuck. Provide specific, actionable advice (2-3 suggestions) to help it make progress.
Focus on concrete actions the agent can take right now."""
