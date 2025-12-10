"""
Custom CNN feature extractor for MiniGrid environments.
MiniGrid has small observation sizes (7x7) so we need smaller kernels.
"""

import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for MiniGrid.
    
    MiniGrid observations are small (typically 7x7x3), so we use:
    - Smaller kernel sizes (2x2 instead of 8x8)
    - More channels for better representation
    - Appropriate feature dimensions
    
    Args:
        observation_space: The observation space
        features_dim: Number of output features (default: 128, can increase to 256/512)
        cnn_channels: Number of CNN channels (default: [32, 64, 64], can increase)
    """
    
    def __init__(
        self, 
        observation_space: gym.Space, 
        features_dim: int = 128,
        cnn_channels: list = None
    ):
        super().__init__(observation_space, features_dim)
        
        # Default: larger network for better performance
        if cnn_channels is None:
            cnn_channels = [32, 64, 64]  # Increased from [16, 32, 64]
        
        # Get observation shape
        n_input_channels = observation_space.shape[0]
        
        # Build CNN layers
        layers = []
        in_channels = n_input_channels
        
        for out_channels in cnn_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=1, padding=0),
                nn.ReLU()
            ])
            in_channels = out_channels
        
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)
        
        # Compute shape by doing one forward pass
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        
        # Larger linear layer
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),  # Extra layer for more capacity
            nn.ReLU()
        )
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
