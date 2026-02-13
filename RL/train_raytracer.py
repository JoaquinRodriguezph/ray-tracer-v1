# train_raytracer.py
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import matplotlib.pyplot as plt

from ray_tracer_env import RayTracerEnv
from vector import Vector
from colour import Colour
from object import Sphere
from material import Material
from light import GlobalLight, PointLight


def create_scene():
    """Create a training scene"""
    # Define materials
    matte_ground = Material(reflective=0, transparent=0, emitive=0.05, refractive_index=1)
    reflective_sphere = Material(reflective=1, transparent=0, emitive=0, refractive_index=1)
    glass_sphere = Material(reflective=0, transparent=1, emitive=0, refractive_index=1.5)
    light_material = Material(reflective=0, transparent=0, emitive=1, refractive_index=1)
    
    # Create scene
    scene_spheres = [
        # Ground plane
        Sphere(Vector(0, -100.5, -3), 100, matte_ground, Colour(150, 150, 150), id=1),
        # Reflective sphere
        Sphere(Vector(0, 0, -3), 0.5, reflective_sphere, Colour(255, 255, 255), id=2),
        # Glass sphere
        Sphere(Vector(-1.5, 0.2, -3), 0.5, glass_sphere, Colour(200, 200, 255), id=3),
        # Another reflective sphere
        Sphere(Vector(1.5, -0.2, -3), 0.5, reflective_sphere, Colour(255, 200, 200), id=4),
        # Light source (top)
        Sphere(Vector(0, 2.5, -3), 0.4, light_material, Colour(255, 255, 200), id=99),
        # Light source (left)
        Sphere(Vector(-2, 1, -3), 0.3, light_material, Colour(200, 255, 200), id=100),
    ]
    
    # Create lights
    global_lights = [
        GlobalLight(
            vector=Vector(0, -1, -0.3).normalise(),
            colour=Colour(150, 150, 200),
            strength=0.2,
            max_angle=np.pi/4
        )
    ]
    
    point_lights = [
        PointLight(
            id=99,
            position=Vector(0, 2.5, -3),
            colour=Colour(255, 255, 200),
            strength=8.0,
            max_angle=np.pi,
            func=0
        ),
        PointLight(
            id=100,
            position=Vector(-2, 1, -3),
            colour=Colour(200, 255, 200),
            strength=6.0,
            max_angle=np.pi,
            func=0
        )
    ]
    
    return scene_spheres, global_lights, point_lights


def create_env():
    """Create the RayTracer environment"""
    spheres, global_lights, point_lights = create_scene()
    
    env = RayTracerEnv(
        spheres=spheres,
        global_light_sources=global_lights,
        point_light_sources=point_lights,
        max_bounces=8,
        image_width=400,
        image_height=300,
        fov=75
    )
    
    return env


def test_random_agent(env, num_episodes=10):
    """Test a random agent for baseline performance"""
    print("\n=== Testing Random Agent ===")
    rewards = []
    steps = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done and episode_steps < 20:  # Limit steps per episode
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1
        
        rewards.append(episode_reward)
        steps.append(episode_steps)
        
        if (episode + 1) % 5 == 0:
            print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Steps={episode_steps}")
    
    print(f"\nRandom Agent Performance:")
    print(f"  Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Average Steps: {np.mean(steps):.1f}")
    
    return np.mean(rewards)


def train_ppo(env, total_timesteps=100000):
    """Train a PPO agent"""
    print("\n=== Training PPO Agent ===")
    
    # Create the model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./raytracer_ppo_tensorboard/"
    )
    
    # Train the model
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Save the model
    model.save("ppo_raytracer")
    print("Model saved as 'ppo_raytracer'")
    
    return model


def train_sac(env, total_timesteps=100000):
    """Train a SAC agent (good for continuous action spaces)"""
    print("\n=== Training SAC Agent ===")
    
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=100000,
        learning_starts=100,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        tensorboard_log="./raytracer_sac_tensorboard/"
    )
    
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save("sac_raytracer")
    print("Model saved as 'sac_raytracer'")
    
    return model


def evaluate_agent(model, env, num_episodes=10):
    """Evaluate a trained agent"""
    print(f"\n=== Evaluating {type(model).__name__} Agent ===")
    
    rewards = []
    steps = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        # Record the path for visualization
        path = []
        
        while not done and episode_steps < 30:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1
            
            # Record position if we hit something
            if env.current_intersection and env.current_intersection.intersects:
                pos = env.current_intersection.point
                path.append((pos.x, pos.y, pos.z, env.current_intersection.object.id))
        
        rewards.append(episode_reward)
        steps.append(episode_steps)
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Steps={episode_steps}")
        
        # Optional: visualize one path
        if episode == 0 and len(path) > 0:
            print("  Sample path (object hits):")
            for i, (x, y, z, obj_id) in enumerate(path[:5]):  # Show first 5 hits
                print(f"    Step {i}: Object {obj_id} at ({x:.2f}, {y:.2f}, {z:.2f})")
    
    print(f"\nEvaluation Results:")
    print(f"  Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Average Steps: {np.mean(steps):.1f}")
    print(f"  Best Episode: {max(rewards):.2f}")
    print(f"  Worst Episode: {min(rewards):.2f}")
    
    return rewards, steps


def plot_training_progress(agent_name, rewards_history):
    """Plot training progress"""
    plt.figure(figsize=(10, 6))
    
    # Smooth the rewards for better visualization
    window_size = 50
    smoothed_rewards = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')
    
    plt.plot(smoothed_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(f'{agent_name} Training Progress')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{agent_name.lower()}_training.png', dpi=100)
    plt.show()


def interactive_demo(env, model=None):
    """Interactive demo to see the agent in action"""
    print("\n=== Interactive Demo ===")
    print("Press Enter to continue steps, 'q' to quit")
    
    obs, info = env.reset()
    print(f"Starting at pixel: {info['pixel']}")
    
    if env.current_intersection and env.current_intersection.intersects:
        print(f"Initial hit: Object {env.current_intersection.object.id}")
    
    step = 0
    total_reward = 0
    
    while True:
        if model:
            action, _ = model.predict(obs, deterministic=True)
            print(f"Step {step}: Using trained policy")
        else:
            action = env.action_space.sample()
            print(f"Step {step}: Random action")
        
        print(f"  Action: theta={action[0]:.2f}, phi={action[1]:.2f}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        if env.current_intersection and env.current_intersection.intersects:
            obj = env.current_intersection.object
            print(f"  Hit object {obj.id} (reward: {reward:.4f})")
            if obj.id == 99 or obj.id == 100:
                print(f"  *** HIT LIGHT SOURCE! ***")
        else:
            print(f"  Missed (reward: {reward:.4f})")
        
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Bounce count: {info['bounce_count']}")
        
        if terminated or truncated:
            print(f"\nEpisode ended: {info.get('reason', 'unknown')}")
            break
        
        # Wait for user input
        user_input = input("\nPress Enter for next step, 'q' to quit: ")
        if user_input.lower() == 'q':
            break
    
    return total_reward


def main():
    """Main training and evaluation function"""
    # Create environment
    env = create_env()
    
    print("=" * 60)
    print("Ray Tracing RL Environment")
    print("=" * 60)
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Max bounces: {env.max_bounces}")
    
    # Test random agent for baseline
    baseline_reward = test_random_agent(env, num_episodes=20)
    
    # Choose which algorithm to train
    algorithm = "ppo"  # Change to "sac" if you want to try SAC
    
    try:
        if algorithm == "ppo":
            # Train PPO agent
            model = train_ppo(env, total_timesteps=50000)  # Start with 50k timesteps
            
            # Evaluate trained agent
            eval_rewards, eval_steps = evaluate_agent(model, env, num_episodes=20)
            
            # Compare with baseline
            improvement = (np.mean(eval_rewards) - baseline_reward) / abs(baseline_reward) * 100
            print(f"\nImprovement over random agent: {improvement:.1f}%")
            
        elif algorithm == "sac":
            # Train SAC agent
            model = train_sac(env, total_timesteps=50000)
            
            # Evaluate trained agent
            eval_rewards, eval_steps = evaluate_agent(model, env, num_episodes=20)
            
            # Compare with baseline
            improvement = (np.mean(eval_rewards) - baseline_reward) / abs(baseline_reward) * 100
            print(f"\nImprovement over random agent: {improvement:.1f}%")
        
        # Interactive demo with trained model
        print("\n" + "=" * 60)
        interactive_demo(env, model)
        
        # Demo with random actions for comparison
        print("\n" + "=" * 60)
        print("Now trying with random actions for comparison...")
        interactive_demo(env, model=None)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        print("\nEnvironment closed.")


if __name__ == "__main__":
    main()