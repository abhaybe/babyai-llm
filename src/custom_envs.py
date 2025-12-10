"""
Custom MiniGrid Environment - Full control over everything!

This lets you create a completely custom task with:
- Custom grid layout
- Custom objects
- Custom reward function
- Custom observations
- LLM-based dynamics!
"""

import gymnasium as gym
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Key, Door, Wall
from minigrid.minigrid_env import MiniGridEnv
import numpy as np


class LLMGuidedEnv(MiniGridEnv):
    """
    Custom environment where LLM can influence:
    1. Task generation
    2. Reward structure
    3. Object placement
    4. Even the mission text!
    """
    
    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps=100,
        llm=None,
        **kwargs
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.llm = llm
        
        # Mission space
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[]
        )
        
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            **kwargs
        )
    
    def _gen_mission(self):
        """Generate mission text - can use LLM here!"""
        if self.llm:
            # LLM generates the mission!
            mission = self.llm.query(
                "Generate a short gridworld task description (5-10 words)"
            )
            return mission
        else:
            return "reach the goal"
    
    def _gen_grid(self, width, height):
        """
        Generate the grid layout.
        You have FULL CONTROL here!
        """
        # Create empty grid
        self.grid = Grid(width, height)
        
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Place objects - can use LLM to decide placement!
        if self.llm:
            # Ask LLM for layout
            layout = self._llm_generate_layout()
            self._place_objects_from_layout(layout)
        else:
            # Default layout
            self._place_default_objects()
        
        # Place agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
    
    def _place_default_objects(self):
        """Default object placement."""
        # Place a goal
        self.put_obj(Goal(), 6, 6)
        
        # Place a key
        self.put_obj(Key('yellow'), 3, 3)
        
        # Place a door
        self.put_obj(Door('yellow', is_locked=True), 5, 3)
    
    def _llm_generate_layout(self):
        """Ask LLM to design the level!"""
        try:
            prompt = f"""Design a simple gridworld level of size {self.width}x{self.height}.
            
Specify object placements:
- goal at (x, y)
- key at (x, y) if needed
- door at (x, y) if needed

Format as:
goal: (6, 6)
key: (3, 3)
door: (5, 3)

Keep it simple and solvable."""

            response = self.llm.query(prompt)
            
            # Parse LLM response
            layout = {}
            for line in response.split('\n'):
                if ':' in line:
                    obj_type, coords = line.split(':', 1)
                    obj_type = obj_type.strip().lower()
                    # Parse coordinates
                    coords = coords.strip()
                    # Simple parsing: (x, y)
                    if '(' in coords and ')' in coords:
                        x, y = coords.strip('()').split(',')
                        layout[obj_type] = (int(x.strip()), int(y.strip()))
            
            return layout
            
        except Exception as e:
            print(f"LLM layout generation failed: {e}")
            return {}
    
    def _place_objects_from_layout(self, layout):
        """Place objects according to LLM's layout."""
        if 'goal' in layout:
            self.put_obj(Goal(), *layout['goal'])
        else:
            self.put_obj(Goal(), 6, 6)  # Default
        
        if 'key' in layout:
            self.put_obj(Key('yellow'), *layout['key'])
        
        if 'door' in layout:
            door_pos = layout['door']
            self.put_obj(Door('yellow', is_locked=True), *door_pos)
    
    def step(self, action):
        """
        Custom step function - can modify rewards here!
        """
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Custom reward shaping (integrated into env!)
        shaped_reward = 0.0
        
        # Example: Reward for getting closer to goal
        if hasattr(self, '_prev_dist_to_goal'):
            current_dist = np.linalg.norm(
                np.array(self.agent_pos) - np.array(self._goal_pos)
            )
            if current_dist < self._prev_dist_to_goal:
                shaped_reward = 0.01  # Small bonus for progress
            self._prev_dist_to_goal = current_dist
        
        # Can even ask LLM for reward!
        # (Don't do this every step - too slow)
        
        total_reward = reward + shaped_reward
        
        return obs, total_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Custom reset."""
        # Store goal position for distance calculations
        obs, info = super().reset(**kwargs)
        
        # Find goal position
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                cell = self.grid.get(i, j)
                if cell and cell.type == 'goal':
                    self._goal_pos = (i, j)
                    break
        
        self._prev_dist_to_goal = np.linalg.norm(
            np.array(self.agent_pos) - np.array(self._goal_pos)
        )
        
        return obs, info


class LLMCurriculumEnv(MiniGridEnv):
    """
    Environment that gets progressively harder based on agent performance.
    LLM decides the difficulty!
    """
    
    def __init__(self, llm=None, **kwargs):
        self.llm = llm
        self.difficulty = 1  # Start easy
        self.success_rate = 0.0
        self.episode_count = 0
        self.recent_successes = []
        
        mission_space = MissionSpace(
            mission_func=lambda: f"reach the goal (difficulty {self.difficulty})"
        )
        
        super().__init__(
            mission_space=mission_space,
            grid_size=8,
            max_steps=100,
            **kwargs
        )
    
    def _gen_grid(self, width, height):
        """Generate grid with difficulty-based complexity."""
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        
        # Difficulty 1: Just a goal
        if self.difficulty == 1:
            self.put_obj(Goal(), 6, 6)
        
        # Difficulty 2: Goal + some walls
        elif self.difficulty == 2:
            self.put_obj(Goal(), 6, 6)
            self.grid.vert_wall(4, 1, 5)
        
        # Difficulty 3: Key + door + goal
        elif self.difficulty == 3:
            self.put_obj(Goal(), 6, 6)
            self.put_obj(Key('yellow'), 2, 2)
            self.put_obj(Door('yellow', is_locked=True), 5, 3)
        
        # Higher difficulty: Ask LLM!
        else:
            if self.llm:
                self._llm_generate_hard_level()
            else:
                # Default hard level
                self.put_obj(Goal(), 6, 6)
                self.put_obj(Key('yellow'), 2, 2)
                self.put_obj(Door('yellow', is_locked=True), 5, 3)
                # More obstacles
                self.grid.vert_wall(3, 1, 3)
        
        self.place_agent()
    
    def _llm_generate_hard_level(self):
        """LLM creates a challenging level."""
        try:
            prompt = f"""Create a challenging gridworld level.
Current difficulty: {self.difficulty}
Agent's recent success rate: {self.success_rate:.1%}

Design a level that's appropriately challenging but solvable.
Include: goal, key, locked door, and some wall obstacles."""

            # LLM designs level
            layout = self.llm.query(prompt)
            # Parse and place objects...
            
        except Exception as e:
            print(f"LLM level generation failed: {e}")
            # Fallback
            self.put_obj(Goal(), 6, 6)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Track performance
        if terminated or truncated:
            self.episode_count += 1
            self.recent_successes.append(1 if reward > 0 else 0)
            
            # Keep last 20 episodes
            if len(self.recent_successes) > 20:
                self.recent_successes.pop(0)
            
            # Update difficulty every 20 episodes
            if self.episode_count % 20 == 0:
                self._update_difficulty()
        
        return obs, reward, terminated, truncated, info
    
    def _update_difficulty(self):
        """Adjust difficulty based on performance."""
        if len(self.recent_successes) < 10:
            return
        
        self.success_rate = np.mean(self.recent_successes)
        
        if self.llm:
            # Ask LLM if we should increase difficulty
            prompt = f"""Agent has {self.success_rate:.1%} success rate.
Current difficulty: {self.difficulty}/5

Should we increase difficulty? Answer YES or NO."""
            
            response = self.llm.query(prompt)
            
            if 'yes' in response.lower() and self.difficulty < 5:
                self.difficulty += 1
                print(f"\n📈 LLM increased difficulty to {self.difficulty}!")
        else:
            # Automatic adjustment
            if self.success_rate > 0.8:
                self.difficulty = min(5, self.difficulty + 1)
            elif self.success_rate < 0.3:
                self.difficulty = max(1, self.difficulty - 1)


# Register custom environments
def register_custom_envs():
    """Register custom environments with gymnasium."""
    from gymnasium.envs.registration import register
    
    register(
        id='MiniGrid-LLMGuided-v0',
        entry_point='custom_envs:LLMGuidedEnv',
    )
    
    register(
        id='MiniGrid-LLMCurriculum-v0',
        entry_point='custom_envs:LLMCurriculumEnv',
    )


# Usage example:
"""
from custom_envs import LLMGuidedEnv, LLMCurriculumEnv
from llm_wrapper import LLMWrapper

# Create LLM
llm = LLMWrapper(provider="openai", model="gpt-3.5-turbo")

# Option 1: LLM-guided environment
env = LLMGuidedEnv(size=8, llm=llm)

# Option 2: Curriculum learning
env = LLMCurriculumEnv(llm=llm)

# Train as normal
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, info = env.reset()
"""
