# Create train_traffic_agent.py

import numpy as np
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from traffic_env import TrafficLightEnv
import gymnasium as gym

# Create a callback to save model and monitor training
class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Calculate mean reward over last 100 episodes
            x, y = self.model.logger.get_log()
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {mean_reward:.2f}")

                # New best model, save the agent
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(os.path.join(self.save_path, 'model'))
        return True

def train_agent(env_id, total_timesteps=100000, log_dir='./logs/', algorithm='PPO'):
    """Train a reinforcement learning agent on the traffic environment"""
    # Create directories for logs and models
    os.makedirs(log_dir, exist_ok=True)
    
    # Create and wrap the environment
    env = gym.make(env_id)
    env = Monitor(env, log_dir)
    
    # Choose the RL algorithm
    if algorithm == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
    elif algorithm == 'DQN':
        model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
    elif algorithm == 'A2C':
        model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
    else:
        raise ValueError(f"Algorithm {algorithm} not supported")
    
    # Train the agent
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Save the final model
    model.save(os.path.join(log_dir, 'final_model'))
    
    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
    
    return model

def plot_rewards(log_dir):
    """Plot training rewards from monitor file"""
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    
    # Load data from monitor.csv
    log_path = os.path.join(log_dir, 'monitor.csv')
    try:
        data = load_results(log_dir)
        x, y = ts2xy(data, 'timesteps')
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(x, y)
        plt.xlabel('Timesteps')
        plt.ylabel('Rewards')
        plt.title('Training Rewards')
        plt.grid()
        plt.show()
    except:
        print(f"No monitor file found at {log_path}")

def test_agent(model_path, env_id, num_episodes=5):
    """Test a trained agent and render results"""
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = gym.make(env_id, render_mode="rgb_array")
    
    # Run episodes
    all_wait_times = []
    all_queue_lengths = []
    all_throughputs = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_wait_times = []
        episode_queue_lengths = []
        cars_passed = 0
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record metrics
            episode_wait_times.append(info['waiting_time'])
            episode_queue_lengths.append(info['queue_length'])
            cars_passed = info['passed_cars']
            
            # Get frame for video
            frame = env.render()
            
            # Display the frame
            if env.current_step % 10 == 0:  # Only show every 10th step to save output space
                plt.figure(figsize=(8, 8))
                plt.imshow(frame)
                plt.title(f"Episode {episode+1}, Step {env.current_step}")
                plt.axis('off')
                plt.show()
                plt.close()
        
        # Record episode metrics
        all_wait_times.append(np.mean(episode_wait_times))
        all_queue_lengths.append(np.mean(episode_queue_lengths))
        all_throughputs.append(cars_passed)
        
        print(f"Episode {episode+1} stats:")
        print(f"  Average wait time: {np.mean(episode_wait_times):.2f}")
        print(f"  Average queue length: {np.mean(episode_queue_lengths):.2f}")
        print(f"  Throughput (cars passed): {cars_passed}")
    
    # Show overall results
    print("\nOverall test results:")
    print(f"  Average wait time: {np.mean(all_wait_times):.2f} +/- {np.std(all_wait_times):.2f}")
    print(f"  Average queue length: {np.mean(all_queue_lengths):.2f} +/- {np.std(all_queue_lengths):.2f}")
    print(f"  Average throughput: {np.mean(all_throughputs):.2f} +/- {np.std(all_throughputs):.2f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.bar(range(num_episodes), all_wait_times)
    plt.xlabel('Episode')
    plt.ylabel('Average Wait Time')
    plt.title('Wait Time by Episode')
    
    plt.subplot(1, 3, 2)
    plt.bar(range(num_episodes), all_queue_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Average Queue Length')
    plt.title('Queue Length by Episode')
    
    plt.subplot(1, 3, 3)
    plt.bar(range(num_episodes), all_throughputs)
    plt.xlabel('Episode')
    plt.ylabel('Throughput (cars)')
    plt.title('Throughput by Episode')
    
    plt.tight_layout()
    plt.show()
    
    env.close()
