import pygame
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
import os

class TrafficLightEnv(gym.Env):
    """Traffic Light Environment that follows gym interface"""
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode=None):
        super(TrafficLightEnv, self).__init__()
        
        # Define environment parameters
        self.grid_size = 5  # Size of the intersection grid
        self.max_cars = 20  # Maximum number of cars per lane
        self.max_steps = 1000  # Maximum steps per episode
        self.current_step = 0
        
        # States: Traffic density in each of the 4 directions (N, S, E, W)
        # Each direction can have 0 to max_cars
        self.observation_space = spaces.Box(
            low=0, high=self.max_cars, shape=(4,), dtype=np.int32)
        
        # Actions: 0 = N-S Green, E-W Red; 1 = N-S Red, E-W Green
        self.action_space = spaces.Discrete(2)
        
        # Traffic lights
        self.lights = {
            'ns': 'green',  # North-South
            'ew': 'red'     # East-West
        }
        
        # Car queues for each direction
        self.queues = {
            'north': [],
            'south': [],
            'east': [],
            'west': []
        }
        
        # Traffic arrival rates (cars per step)
        self.arrival_rates = {
            'north': 0.3,
            'south': 0.3,
            'east': 0.3,
            'west': 0.3
        }
        
        # Cars that have passed through the intersection
        self.passed_cars = 0
        
        # Statistics
        self.waiting_times = []
        self.queue_lengths = []
        
        # PyGame setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.width = 600
        self.height = 600
        
        # Initial reset
        self.reset()
    
    def _get_obs(self):
        # Return current queue lengths as observation
        return np.array([
            len(self.queues['north']),
            len(self.queues['south']),
            len(self.queues['east']),
            len(self.queues['west'])
        ])
    
    def _get_info(self):
        # Return additional info for debugging
        return {
            'waiting_time': np.mean(self.waiting_times) if self.waiting_times else 0,
            'queue_length': np.mean(self.queue_lengths) if self.queue_lengths else 0,
            'passed_cars': self.passed_cars
        }
    
    def reset(self, seed=None, options=None):
        self.current_step = 0
        
        # Reset traffic lights
        self.lights = {
            'ns': 'green',
            'ew': 'red'
        }
        
        # Reset car queues
        self.queues = {
            'north': [],
            'south': [],
            'east': [],
            'west': []
        }
        
        # Reset statistics
        self.passed_cars = 0
        self.waiting_times = []
        self.queue_lengths = []
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info
    
    def _generate_traffic(self):
        # Generate new cars based on arrival rates
        for direction, rate in self.arrival_rates.items():
            if random.random() < rate and len(self.queues[direction]) < self.max_cars:
                # Add a new car to the queue with arrival time
                self.queues[direction].append(self.current_step)
    
    def _process_traffic(self):
        # Process traffic based on current light status
        if self.lights['ns'] == 'green':
            # Process North-South traffic
            for direction in ['north', 'south']:
                if self.queues[direction]:
                    # Remove one car from the queue (it passed through)
                    arrival_time = self.queues[direction].pop(0)
                    wait_time = self.current_step - arrival_time
                    self.waiting_times.append(wait_time)
                    self.passed_cars += 1
        else:
            # Process East-West traffic
            for direction in ['east', 'west']:
                if self.queues[direction]:
                    # Remove one car from the queue (it passed through)
                    arrival_time = self.queues[direction].pop(0)
                    wait_time = self.current_step - arrival_time
                    self.waiting_times.append(wait_time)
                    self.passed_cars += 1
    
    def step(self, action):
        # Update traffic light based on action
        if action == 0:  # N-S Green, E-W Red
            self.lights['ns'] = 'green'
            self.lights['ew'] = 'red'
        else:  # N-S Red, E-W Green
            self.lights['ns'] = 'red'
            self.lights['ew'] = 'green'
        
        # Generate new traffic
        self._generate_traffic()
        
        # Process traffic flow
        self._process_traffic()
        
        # Update statistics
        total_queue = sum(len(queue) for queue in self.queues.values())
        self.queue_lengths.append(total_queue)
        
        # Calculate reward: negative of total waiting time
        reward = -total_queue
        
        # Check if episode is done
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.Surface((self.width, self.height))
            self.font = pygame.font.SysFont('Arial', 20)
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Fill background
        self.screen.fill((220, 220, 220))
        
        # Draw road
        pygame.draw.rect(self.screen, (80, 80, 80), (0, self.height//2 - 40, self.width, 80))
        pygame.draw.rect(self.screen, (80, 80, 80), (self.width//2 - 40, 0, 80, self.height))
        
        # Draw lane markings
        for i in range(0, self.width, 30):
            pygame.draw.rect(self.screen, (255, 255, 0), (i, self.height//2 - 2, 15, 4))
        for i in range(0, self.height, 30):
            pygame.draw.rect(self.screen, (255, 255, 0), (self.width//2 - 2, i, 4, 15))
        
        # Draw traffic lights
        # North-South lights
        ns_color = (0, 255, 0) if self.lights['ns'] == 'green' else (255, 0, 0)
        pygame.draw.circle(self.screen, ns_color, (self.width//2 + 60, self.height//2 - 60), 15)
        pygame.draw.circle(self.screen, ns_color, (self.width//2 - 60, self.height//2 + 60), 15)
        
        # East-West lights
        ew_color = (0, 255, 0) if self.lights['ew'] == 'green' else (255, 0, 0)
        pygame.draw.circle(self.screen, ew_color, (self.width//2 - 60, self.height//2 - 60), 15)
        pygame.draw.circle(self.screen, ew_color, (self.width//2 + 60, self.height//2 + 60), 15)
        
        # Draw cars in queues
        # North queue
        for i, _ in enumerate(self.queues['north'][:10]):  # Show up to 10 cars
            pygame.draw.rect(self.screen, (0, 0, 255), 
                            (self.width//2 + 10, self.height//2 - 100 - i*30, 20, 25))
        
        # South queue
        for i, _ in enumerate(self.queues['south'][:10]):
            pygame.draw.rect(self.screen, (255, 0, 255), 
                            (self.width//2 - 30, self.height//2 + 75 + i*30, 20, 25))
        
        # East queue
        for i, _ in enumerate(self.queues['east'][:10]):
            pygame.draw.rect(self.screen, (255, 165, 0), 
                            (self.width//2 + 75 + i*30, self.height//2 - 30, 25, 20))
        
        # West queue
        for i, _ in enumerate(self.queues['west'][:10]):
            pygame.draw.rect(self.screen, (0, 255, 255), 
                            (self.width//2 - 100 - i*30, self.height//2 + 10, 25, 20))
        
        # Display stats
        queue_text = self.font.render(f'Total Queue: {sum(len(q) for q in self.queues.values())}', True, (0, 0, 0))
        self.screen.blit(queue_text, (10, 10))
        
        wait_time = np.mean(self.waiting_times) if self.waiting_times else 0
        wait_text = self.font.render(f'Avg Wait: {wait_time:.2f}', True, (0, 0, 0))
        self.screen.blit(wait_text, (10, 40))
        
        passed_text = self.font.render(f'Passed: {self.passed_cars}', True, (0, 0, 0))
        self.screen.blit(passed_text, (10, 70))
        
        step_text = self.font.render(f'Step: {self.current_step}', True, (0, 0, 0))
        self.screen.blit(step_text, (10, 100))
        
        if self.render_mode == "human":
            # Not actually displayed in Colab, but needed for render to work
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )
        
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
