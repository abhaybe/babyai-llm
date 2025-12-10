"""
Text-conditioned policy - PPO that can actually read LLM instructions!
This modifies the neural network to accept both image AND text.
"""

import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from transformers import AutoTokenizer, AutoModel


class TextImageFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that processes both images (CNN) and text (transformer).
    
    Inputs:
    - Image: (7, 7, 3) RGB observation
    - Text: LLM hint/instruction as string
    
    Output:
    - Combined feature vector (image features + text features)
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        text_encoder: str = "distilbert-base-uncased"  # Small, fast
    ):
        super().__init__(observation_space, features_dim)
        
        # CNN for image (same as before)
        n_input_channels = observation_space['image'].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute CNN output size
        with th.no_grad():
            sample_image = th.zeros(1, *observation_space['image'].shape)
            cnn_output_dim = self.cnn(sample_image).shape[1]
        
        # Text encoder (frozen pre-trained model for efficiency)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)
        self.text_encoder = AutoModel.from_pretrained(text_encoder)
        
        # Freeze text encoder (optional - fine-tune if needed)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        text_dim = self.text_encoder.config.hidden_size  # Usually 768
        
        # Combine image and text features
        self.combiner = nn.Sequential(
            nn.Linear(cnn_output_dim + text_dim, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: dict) -> th.Tensor:
        """
        Process both image and text observations.
        
        Args:
            observations: Dict with 'image' and 'text' keys
        
        Returns:
            Combined feature vector
        """
        # Process image
        images = observations['image']
        image_features = self.cnn(images)
        
        # Process text
        texts = observations['text']  # List of strings
        
        # Tokenize text
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Get text features
        with th.no_grad():  # Frozen encoder
            text_output = self.text_encoder(**encoded)
            text_features = text_output.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Combine
        combined = th.cat([image_features, text_features], dim=1)
        return self.combiner(combined)


class SimplifiedTextImageExtractor(BaseFeaturesExtractor):
    """
    Simplified version - uses simple text embedding instead of transformer.
    Much faster, good enough for short hints.
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        vocab_size: int = 1000,
        text_embed_dim: int = 64
    ):
        super().__init__(observation_space, features_dim)
        
        # CNN for image
        n_input_channels = observation_space['image'].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with th.no_grad():
            sample_image = th.zeros(1, *observation_space['image'].shape)
            cnn_output_dim = self.cnn(sample_image).shape[1]
        
        # Simple text embedding
        self.text_embedding = nn.Embedding(vocab_size, text_embed_dim)
        self.text_lstm = nn.LSTM(text_embed_dim, text_embed_dim, batch_first=True)
        
        # Combine
        self.combiner = nn.Sequential(
            nn.Linear(cnn_output_dim + text_embed_dim, features_dim),
            nn.ReLU()
        )
        
        # Simple vocabulary (for demo - use proper tokenizer in practice)
        self.vocab = self._build_vocab()
    
    def _build_vocab(self):
        """Build simple word->id mapping."""
        words = ['<pad>', '<unk>', 'find', 'key', 'door', 'open', 'pick', 'up',
                 'go', 'to', 'goal', 'north', 'south', 'east', 'west', 'red', 
                 'blue', 'green', 'yellow', 'the', 'a', 'and', 'then']
        return {word: i for i, word in enumerate(words)}
    
    def _text_to_ids(self, text: str):
        """Convert text to token IDs."""
        words = text.lower().split()
        ids = [self.vocab.get(word, 1) for word in words]  # 1 = <unk>
        return ids[:20]  # Max 20 words
    
    def forward(self, observations: dict) -> th.Tensor:
        # Process image
        images = observations['image']
        image_features = self.cnn(images)
        
        # Process text (simple version)
        texts = observations['text']
        
        # Convert texts to IDs and pad
        batch_ids = []
        max_len = 20
        for text in texts:
            ids = self._text_to_ids(text)
            ids += [0] * (max_len - len(ids))  # Pad
            batch_ids.append(ids[:max_len])
        
        text_tensor = th.tensor(batch_ids, device=images.device)
        text_embeds = self.text_embedding(text_tensor)
        
        # LSTM to process sequence
        _, (hidden, _) = self.text_lstm(text_embeds)
        text_features = hidden.squeeze(0)
        
        # Combine
        combined = th.cat([image_features, text_features], dim=1)
        return self.combiner(combined)


# Wrapper to provide text observations to the policy
class TextObservationWrapper(gym.Wrapper):
    """
    Wrapper that adds LLM hint as part of observation.
    Now agent can actually SEE the LLM advice!
    """
    
    def __init__(self, env, llm, mode='planning'):
        super().__init__(env)
        self.llm = llm
        self.mode = mode
        self.current_hint = ""
        
        # Modify observation space to include text
        from gymnasium.spaces import Dict, Box, Text
        
        # Keep image observation
        original_space = env.observation_space
        if isinstance(original_space, Dict):
            image_space = original_space['image']
        else:
            image_space = original_space
        
        # Add text space
        self.observation_space = Dict({
            'image': image_space,
            'text': Text(max_length=200)  # LLM hint
        })
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Get LLM hint
        if self.mode == 'planning':
            mission = getattr(self.env.unwrapped, 'mission', 'complete task')
            prompt = f"Task: {mission}\nProvide a brief strategy (1 sentence):"
            self.current_hint = self.llm.query(prompt)
        else:
            self.current_hint = "explore and find the goal"
        
        # Return observation with text
        return {
            'image': obs if not isinstance(obs, dict) else obs['image'],
            'text': self.current_hint
        }, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Return observation with text
        return {
            'image': obs if not isinstance(obs, dict) else obs['image'],
            'text': self.current_hint
        }, reward, terminated, truncated, info


# How to use:
"""
from text_policy import SimplifiedTextImageExtractor, TextObservationWrapper
from llm_wrapper import LLMWrapper

# 1. Create environment with text observation
llm = LLMWrapper(provider="openai", model="gpt-3.5-turbo")
env = gym.make("MiniGrid-DoorKey-8x8-v0")
env = TextObservationWrapper(env, llm, mode='planning')

# 2. Create PPO with text-aware policy
policy_kwargs = dict(
    features_extractor_class=SimplifiedTextImageExtractor,
    features_extractor_kwargs=dict(features_dim=256),
)

model = PPO(
    "MultiInputPolicy",  # Important: Use MultiInputPolicy for dict obs!
    env,
    policy_kwargs=policy_kwargs,
    verbose=1
)

# 3. Train - now agent can READ the LLM hints!
model.learn(total_timesteps=100000)

# The agent now actually uses the text! 🎉
"""
