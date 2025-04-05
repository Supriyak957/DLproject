# Create main.py

import os
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

# Import custom modules
from register_env import register
from traffic_env import TrafficLightEnv
from train_traffic_agent import train_agent, plot_rewards, test_agent

# Main function
def main():
    # Create log directory
    log_dir = './logs/'
    os.makedirs(log_dir, exist_ok=True)
    
    # Train the agent
    print("Training the AI agent...")
    model = train_agent(
        env_id='TrafficLight-v0', 
        total_timesteps=30000,  # Reduced for demonstration in Colab
        log_dir=log_dir,
        algorithm='PPO'  # You can change to 'DQN' or 'A2C'
    )
    
    # Plot training rewards
    print("Plotting training rewards...")
    plot_rewards(log_dir)
    
    # Test the trained agent
    print("Testing the trained agent...")
    model_path = os.path.join(log_dir, 'final_model.zip')
    test_agent(model_path, 'TrafficLight-v0', num_episodes=2)  # Reduced episodes for brevity
    
    print("Experiment completed!")

if __name__ == "__main__":
    main()
