# Create register_env.py

import gymnasium as gym
from gymnasium.envs.registration import register

# Register the custom environment
register(
    id='TrafficLight-v0',
    entry_point='traffic_env:TrafficLightEnv',
    max_episode_steps=1000,
)
