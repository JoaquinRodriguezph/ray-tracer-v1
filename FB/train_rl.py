"""Train RL agent for your custom scene"""

import numpy as np
import torch
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.logger import configure
from tqdm import tqdm
import gymnasium as gym

from vector import Vector
from object import Sphere
from material import Material
from colour import Colour

class ProgressBarCallback(BaseCallback):
    """Custom callback to display a progress bar during training"""
    
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
    
    def _on_training_start(self):
        """Initialize the progress bar"""
        self.pbar = tqdm(total=self.total_timesteps, desc="Training RL Agent")
    
    def _on_step(self):
        """Update the progress bar"""
        self.pbar.update(self.locals.get("num_envs", 1))
        self.pbar.set_postfix({
            "reward": f"{self.locals.get('rewards', [0])[0]:.2f}",
            "loss": f"{self.locals.get('loss', 0):.4f}"
        })
        return True
    
    def _on_training_end(self):
        """Close the progress bar"""
        self.pbar.close()

def create_scene_for_rl():
    """Create your custom scene for RL training"""
    
    # Materials
    base_material = Material(reflective=False)
    reflective_material = Material(reflective=True)
    glass = Material(reflective=False, transparent=True, refractive_index=1.52)
    emitive_material = Material(emitive=True)
    
    # Your exact scene
    spheres = [
        Sphere(id=1, centre=Vector(-0.8, 0.6, 0), radius=0.3, 
               material=glass, colour=Colour(255, 100, 100)),
        Sphere(id=2, centre=Vector(0.8, -0.8, -10), radius=2.2,
               material=base_material, colour=Colour(204, 204, 255)),
        Sphere(id=3, centre=Vector(0.3, 0.34, 0.1), radius=0.2,
               material=base_material, colour=Colour(0, 51, 204)),
        Sphere(id=4, centre=Vector(5.6, 3, -2), radius=5,
               material=reflective_material, colour=Colour(153, 51, 153)),
        Sphere(id=5, centre=Vector(-0.8, -0.8, -0.2), radius=0.25,
               material=base_material, colour=Colour(153, 204, 0)),
        Sphere(id=6, centre=Vector(-3, 10, -75), radius=30,
               material=base_material, colour=Colour(255, 204, 102)),
        # SUN - MUST be id=7 for RL environment to recognize it!
        Sphere(id=7, centre=Vector(-0.6, 0.2, 6), radius=0.1,
               material=emitive_material, colour=Colour(255, 255, 204))
    ]
    
    return spheres

def train_rl_agent():
    """Main RL training"""
    from ray_tracer_env import RayTracerEnv
    
    print("\n" + "="*80)
    print("TRAINING RL AGENT FOR YOUR CUSTOM SCENE")
    print("="*80)
    
    # Create environment with YOUR scene
    spheres = create_scene_for_rl()
    
    print("\nCreating RL environment...")
    env = RayTracerEnv(
        spheres=spheres,
        image_width=200,  # Small for training
        image_height=150,
        camera_position=Vector(0, 0, 1),  # Your camera position
        max_bounces=5,
        background_colour=Colour(2, 2, 5),
        render_mode=None
    )
    
    print(f"✓ Environment created with {len(spheres)} spheres")
    print(f"✓ Sun is sphere id=7 at position: ({spheres[-1].centre.x}, {spheres[-1].centre.y}, {spheres[-1].centre.z})")
    
    # Wrap for Stable Baselines3
    env = DummyVecEnv([lambda: env])
    
    # Add monitoring for better progress tracking
    env = VecMonitor(env, "./rl_tensorboard/")
    
    # Create output directory
    os.makedirs("./rl_checkpoints", exist_ok=True)
    os.makedirs("./rl_tensorboard", exist_ok=True)
    
    # Training configuration
    total_timesteps = 100000  # Reduced for testing - increase later
    
    print(f"\nTraining Configuration:")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Checkpoint frequency: Every 10,000 steps")
    print(f"  Tensorboard log: ./rl_tensorboard/")
    print(f"  Checkpoints saved to: ./rl_checkpoints/")
    
    # Create agent with smaller network for faster training
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,  # Increased for stability
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 64], vf=[128, 64])]  # Smaller network
        ),
        tensorboard_log="./rl_tensorboard/"
    )
    
    # Callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./rl_checkpoints/",
        name_prefix="rl_ray_tracer"
    )
    callbacks.append(checkpoint_callback)
    
    # Progress bar callback
    progress_callback = ProgressBarCallback(total_timesteps=total_timesteps)
    callbacks.append(progress_callback)
    
    print("\n" + "-"*80)
    print("STARTING TRAINING...")
    print("-"*80)
    
    # Train
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=100,  # Log every 100 steps
            tb_log_name="ray_tracer_ppo",
            progress_bar=False  # We use our own progress bar
        )
        
        # Save final model
        final_path = "./rl_ray_tracer_final"
        model.save(final_path)
        print(f"\n✓ Training complete!")
        print(f"✓ Model saved as: {final_path}.zip")
        
        # Test the trained model
        print("\n" + "-"*80)
        print("TESTING TRAINED MODEL...")
        print("-"*80)
        
        test_model(model, spheres)
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        # Save current model
        interrupted_path = "./rl_ray_tracer_interrupted"
        model.save(interrupted_path)
        print(f"✓ Model saved as: {interrupted_path}.zip")
    
    return model

def test_model(model, spheres):
    """Test the trained model"""
    from ray_tracer_env import RayTracerEnv
    
    # Create test environment
    test_env = RayTracerEnv(
        spheres=spheres,
        image_width=200,
        image_height=150,
        camera_position=Vector(0, 0, 1),
        max_bounces=5,
        background_colour=Colour(2, 2, 5)
    )
    
    # Run a few test episodes
    num_test_episodes = 5
    total_rewards = []
    
    print(f"\nRunning {num_test_episodes} test episodes...")
    
    for episode in range(num_test_episodes):
        obs, _ = test_env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < 20:  # Limit steps
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            # Check if we hit the sun
            if info.get('hit_sun', False):
                print(f"  Episode {episode+1}: HIT SUN at step {steps}! Reward: {reward:.2f}")
        
        total_rewards.append(episode_reward)
        print(f"  Episode {episode+1}: Total reward = {episode_reward:.2f}, Steps = {steps}")
    
    avg_reward = np.mean(total_rewards)
    print(f"\n✓ Average test reward: {avg_reward:.2f}")
    
    if avg_reward > 0:
        print("✓ Model appears to be learning!")
    else:
        print("⚠ Model may need more training")

if __name__ == "__main__":
    # Check if Stable-Baselines3 is installed
    try:
        import stable_baselines3
        print("✓ Stable-Baselines3 is installed")
    except ImportError:
        print("\n✗ ERROR: Stable-Baselines3 is not installed!")
        print("Install it with: pip install stable-baselines3")
        print("Also install: pip install tensorboard")
        exit(1)
    
    # Check if GPU is available
    if torch.cuda.is_available():
        print("✓ GPU is available")
    else:
        print("⚠ Training on CPU (slower)")
    
    # Start training
    train_rl_agent()