# simple_train.py - A basic RL implementation without external libraries
import numpy as np
import pickle
import os
from collections import deque
import random

from ray_tracer_env import RayTracerEnv
from vector import Vector
from colour import Colour
from object import Sphere
from material import Material
from light import GlobalLight, PointLight


class SimpleQNetwork:
    """A simple neural network for Q-learning"""
    def __init__(self, state_size, action_size, hidden_size=64):
        self.state_size = state_size
        self.action_size = action_size
        
        # Simple weights (could be replaced with proper NN)
        self.W1 = np.random.randn(state_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, action_size) * 0.01
        self.b2 = np.zeros((1, action_size))
        
    def forward(self, state):
        """Forward pass"""
        z1 = np.dot(state, self.W1) + self.b1
        a1 = np.tanh(z1)  # Tanh activation
        z2 = np.dot(a1, self.W2) + self.b2
        return z2  # No activation on output (Q-values)
    
    def predict(self, state):
        """Predict Q-values for a state"""
        return self.forward(state)
    
    def update(self, states, targets, learning_rate=0.001):
        """Simple gradient update"""
        # This is a very simplified version - in practice you'd use backprop
        # For simplicity, we'll just do a basic update
        pass


class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


def discretize_action(action_continuous, num_bins=8):
    """Discretize continuous action space for simple Q-learning"""
    theta_bins = np.linspace(0, np.pi/2, num_bins)
    phi_bins = np.linspace(0, 2*np.pi, num_bins)
    
    theta_discrete = np.digitize(action_continuous[0], theta_bins) - 1
    phi_discrete = np.digitize(action_continuous[1], phi_bins) - 1
    
    action_idx = theta_discrete * num_bins + phi_discrete
    return action_idx


def continuous_from_discrete(action_idx, num_bins=8):
    """Convert discrete action back to continuous"""
    theta_bins = np.linspace(0, np.pi/2, num_bins)
    phi_bins = np.linspace(0, 2*np.pi, num_bins)
    
    theta_idx = action_idx // num_bins
    phi_idx = action_idx % num_bins
    
    theta = theta_bins[theta_idx]
    phi = phi_bins[phi_idx]
    
    return np.array([theta, phi], dtype=np.float32)


def create_training_scene():
    """Create a scene optimized for training"""
    # Simple materials
    matte = Material(reflective=0, transparent=0, emitive=0.05, refractive_index=1)
    reflective = Material(reflective=1, transparent=0, emitive=0, refractive_index=1)
    light_mat = Material(reflective=0, transparent=0, emitive=1, refractive_index=1)
    
    # Simple scene with clear light path
    spheres = [
        # Ground
        Sphere(Vector(0, -101, -3), 100, matte, Colour(150, 150, 150), id=1),
        # Central target
        Sphere(Vector(0, 0, -3), 0.5, reflective, Colour(255, 255, 255), id=2),
        # Light (easy to hit)
        Sphere(Vector(0, 2, -3), 0.5, light_mat, Colour(255, 255, 200), id=99),
    ]
    
    lights = [
        PointLight(
            id=99,
            position=Vector(0, 2, -3),
            colour=Colour(255, 255, 200),
            strength=10.0,
            max_angle=np.pi,
            func=0
        )
    ]
    
    return spheres, [], lights


def simple_q_learning():
    """Simple Q-learning implementation"""
    print("Starting Simple Q-Learning")
    
    # Create environment
    spheres, global_lights, point_lights = create_training_scene()
    env = RayTracerEnv(
        spheres=spheres,
        global_light_sources=global_lights,
        point_light_sources=point_lights,
        max_bounces=5,
        image_width=200,
        image_height=150,
        fov=90
    )
    
    # Discretize action space
    num_bins = 8
    num_actions = num_bins * num_bins
    
    # Initialize Q-table
    state_size = env.observation_space.shape[0]
    
    # Simple discretization of state space (very coarse)
    state_bins = 4
    q_table = np.random.randn(state_bins, state_bins, state_bins, num_actions) * 0.01
    
    # Training parameters
    num_episodes = 1000
    max_steps = 10
    learning_rate = 0.1
    discount_factor = 0.95
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    
    rewards_history = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        # Discretize initial state (very simple - just use first 3 position values)
        state_idx = (
            min(int((obs[0] + 5) * state_bins / 10), state_bins-1),
            min(int((obs[1] + 5) * state_bins / 10), state_bins-1),
            min(int((obs[2] + 5) * state_bins / 10), state_bins-1)
        )
        
        while not done and step < max_steps:
            # Epsilon-greedy policy
            if np.random.random() < epsilon:
                action_idx = np.random.randint(0, num_actions)
            else:
                action_idx = np.argmax(q_table[state_idx])
            
            # Convert to continuous action
            action = continuous_from_discrete(action_idx, num_bins)
            
            # Take action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            # Discretize next state
            next_state_idx = (
                min(int((next_obs[0] + 5) * state_bins / 10), state_bins-1),
                min(int((next_obs[1] + 5) * state_bins / 10), state_bins-1),
                min(int((next_obs[2] + 5) * state_bins / 10), state_bins-1)
            )
            
            # Q-learning update
            best_next_action = np.argmax(q_table[next_state_idx])
            td_target = reward + discount_factor * q_table[next_state_idx][best_next_action]
            td_error = td_target - q_table[state_idx][action_idx]
            q_table[state_idx][action_idx] += learning_rate * td_error
            
            state_idx = next_state_idx
            step += 1
        
        # Update epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode + 1}, Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.3f}")
    
    # Test the learned policy
    print("\nTesting learned policy...")
    test_rewards = []
    
    for test_ep in range(10):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        state_idx = (
            min(int((obs[0] + 5) * state_bins / 10), state_bins-1),
            min(int((obs[1] + 5) * state_bins / 10), state_bins-1),
            min(int((obs[2] + 5) * state_bins / 10), state_bins-1)
        )
        
        while not done and step < max_steps:
            action_idx = np.argmax(q_table[state_idx])
            action = continuous_from_discrete(action_idx, num_bins)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            state_idx = (
                min(int((obs[0] + 5) * state_bins / 10), state_bins-1),
                min(int((obs[1] + 5) * state_bins / 10), state_bins-1),
                min(int((obs[2] + 5) * state_bins / 10), state_bins-1)
            )
            step += 1
        
        test_rewards.append(total_reward)
        print(f"Test episode {test_ep + 1}: Reward = {total_reward:.2f}")
    
    print(f"\nAverage test reward: {np.mean(test_rewards):.2f}")
    env.close()
    
    return rewards_history


if __name__ == "__main__":
    # Run simple Q-learning
    rewards = simple_q_learning()
    
    # Plot results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Simple Q-Learning Progress')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('simple_q_learning.png', dpi=100)
    plt.show()