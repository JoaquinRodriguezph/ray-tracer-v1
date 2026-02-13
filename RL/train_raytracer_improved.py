# train_raytracer_improved.py
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque

from ray_tracer_env import RayTracerEnv
from vector import Vector
from colour import Colour
from object import Sphere
from material import Material
from light import GlobalLight, PointLight


class RewardLoggerCallback(BaseCallback):
    """Custom callback for logging rewards"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Get info from the environment
        infos = self.locals.get('infos', [{}])
        
        for info in infos:
            if 'episode' in info:
                reward = info['episode']['r']
                length = info['episode']['l']
                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
                
                if self.num_timesteps % 5000 == 0:
                    recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
                    recent_lengths = self.episode_lengths[-100:] if len(self.episode_lengths) >= 100 else self.episode_lengths
                    
                    print(f"\nStep {self.num_timesteps}:")
                    print(f"  Recent avg reward: {np.mean(recent_rewards):.3f} ± {np.std(recent_rewards):.3f}")
                    print(f"  Recent avg length: {np.mean(recent_lengths):.2f}")
                    print(f"  Recent success rate: {sum(1 for r in recent_rewards if r > 0)/len(recent_rewards):.1%}")
        
        return True


def create_optimized_scene():
    """Create a scene optimized for RL learning"""
    # Materials
    matte = Material(reflective=0, transparent=0, emitive=0.1, refractive_index=1)
    reflective = Material(reflective=1, transparent=0, emitive=0, refractive_index=1)
    light_mat = Material(reflective=0, transparent=0, emitive=1, refractive_index=1)
    
    # Scene with clear learning objectives
    scene_spheres = [
        # Large ground (easy to hit)
        Sphere(Vector(0, -100, -3), 99, matte, Colour(100, 100, 100), id=1),
        # Central reflective sphere (learning target)
        Sphere(Vector(0, 0, -3), 0.7, reflective, Colour(255, 255, 255), id=2),
        # Secondary reflective sphere
        Sphere(Vector(-1.8, 0.3, -3), 0.5, reflective, Colour(200, 200, 255), id=3),
        # Light source 1 (main target)
        Sphere(Vector(0, 2, -3), 0.5, light_mat, Colour(255, 255, 200), id=99),
        # Light source 2 (secondary target)
        Sphere(Vector(-2, 1.5, -3), 0.4, light_mat, Colour(200, 255, 200), id=100),
    ]
    
    # Lights corresponding to light spheres
    point_lights = [
        PointLight(
            id=99,
            position=Vector(0, 2, -3),
            colour=Colour(255, 255, 200),
            strength=12.0,
            max_angle=np.pi,
            func=0
        ),
        PointLight(
            id=100,
            position=Vector(-2, 1.5, -3),
            colour=Colour(200, 255, 200),
            strength=8.0,
            max_angle=np.pi,
            func=0
        )
    ]
    
    return scene_spheres, [], point_lights


def create_env_optimized():
    """Create optimized environment"""
    spheres, global_lights, point_lights = create_optimized_scene()
    
    env = RayTracerEnv(
        spheres=spheres,
        global_light_sources=global_lights,
        point_light_sources=point_lights,
        max_bounces=6,
        image_width=320,
        image_height=240,
        fov=80
    )
    
    return env


def analyze_agent_behavior(env, model, num_episodes=1000):
    """Detailed analysis of agent behavior"""
    print("\n" + "="*70)
    print("ANALYZING AGENT BEHAVIOR")
    print("="*70)
    
    # Track various metrics
    metrics = {
        'rewards': [],
        'steps': [],
        'hit_object_ids': [],
        'hit_counts': {},
        'success_rates': {'bounce1': [], 'bounce2': [], 'bounce3+': []},
        'final_reasons': {'ray_missed': 0, 'ray_escaped': 0, 'max_bounces': 0}
    }
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        objects_hit = []
        
        while not done and episode_steps < 10:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1
            
            # Track what was hit
            if env.current_intersection and env.current_intersection.intersects:
                obj_id = env.current_intersection.object.id
                objects_hit.append(obj_id)
                metrics['hit_counts'][obj_id] = metrics['hit_counts'].get(obj_id, 0) + 1
        
        # Record episode metrics
        metrics['rewards'].append(episode_reward)
        metrics['steps'].append(episode_steps)
        metrics['hit_object_ids'].append(objects_hit)
        
        # Track success by bounce count
        if episode_steps == 1 and episode_reward > 0:
            metrics['success_rates']['bounce1'].append(1)
        elif episode_steps == 2 and episode_reward > 0:
            metrics['success_rates']['bounce2'].append(1)
        elif episode_steps >= 3 and episode_reward > 0:
            metrics['success_rates']['bounce3+'].append(1)
        
        # Track termination reasons
        if 'reason' in info:
            metrics['final_reasons'][info['reason']] = metrics['final_reasons'].get(info['reason'], 0) + 1
        
        # Progress reporting
        if (episode + 1) % 200 == 0:
            recent_rewards = metrics['rewards'][-200:]
            avg_reward = np.mean(recent_rewards)
            success_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
            print(f"  Episode {episode + 1}: Avg Reward={avg_reward:.3f}, Success Rate={success_rate:.1%}")
    
    # Print analysis
    print("\n" + "-"*70)
    print("BEHAVIOR ANALYSIS RESULTS")
    print("-"*70)
    
    print(f"\nReward Statistics:")
    print(f"  Average Reward: {np.mean(metrics['rewards']):.3f} ± {np.std(metrics['rewards']):.3f}")
    print(f"  Median Reward: {np.median(metrics['rewards']):.3f}")
    print(f"  Success Rate: {sum(1 for r in metrics['rewards'] if r > 0)/len(metrics['rewards']):.1%}")
    
    print(f"\nEpisode Length:")
    print(f"  Average Steps: {np.mean(metrics['steps']):.2f}")
    print(f"  Max Steps: {max(metrics['steps'])}")
    
    print(f"\nObjects Hit (frequency):")
    for obj_id, count in sorted(metrics['hit_counts'].items(), key=lambda x: x[1], reverse=True):
        obj_name = {1: "Ground", 2: "Center Sphere", 3: "Left Sphere", 
                   99: "Light 1", 100: "Light 2"}.get(obj_id, f"Object {obj_id}")
        percentage = count / sum(metrics['hit_counts'].values()) * 100
        print(f"  {obj_name}: {count} hits ({percentage:.1f}%)")
    
    print(f"\nSuccess by Bounce Count:")
    for bounce_type in ['bounce1', 'bounce2', 'bounce3+']:
        if metrics['success_rates'][bounce_type]:
            rate = np.mean(metrics['success_rates'][bounce_type])
            print(f"  {bounce_type}: {rate:.1%}")
    
    print(f"\nTermination Reasons:")
    for reason, count in metrics['final_reasons'].items():
        percentage = count / num_episodes * 100
        print(f"  {reason}: {count} ({percentage:.1f}%)")
    
    return metrics


def train_with_curriculum():
    """Train with curriculum learning - start easy, get harder"""
    print("\n" + "="*70)
    print("CURRICULUM LEARNING - PHASE 1: Simple Scene")
    print("="*70)
    
    # Phase 1: Very simple scene
    spheres_phase1 = [
        Sphere(Vector(0, -100, -3), 99, Material(emitive=0.1), Colour(100, 100, 100), id=1),
        Sphere(Vector(0, 0, -3), 0.7, Material(reflective=1), Colour(255, 255, 255), id=2),
        Sphere(Vector(0, 2, -3), 0.6, Material(emitive=1), Colour(255, 255, 200), id=99),
    ]
    
    lights_phase1 = [
        PointLight(id=99, position=Vector(0, 2, -3), colour=Colour(255, 255, 200), strength=15.0, max_angle=np.pi, func=0)
    ]
    
    env_phase1 = RayTracerEnv(
        spheres=spheres_phase1,
        point_light_sources=lights_phase1,
        max_bounces=4,
        image_width=160,
        image_height=120,
        fov=90
    )
    
    # Train on simple scene
    model_phase1 = PPO(
        "MlpPolicy",
        env_phase1,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,  # Smaller for faster learning
        batch_size=32,
        n_epochs=5,
        gamma=0.95,
        ent_coef=0.05,  # Higher entropy for exploration
    )
    
    print("Training Phase 1 (10k steps)...")
    model_phase1.learn(total_timesteps=10000, progress_bar=True)
    
    # Phase 2: More complex scene
    print("\n" + "="*70)
    print("CURRICULUM LEARNING - PHASE 2: Complex Scene")
    print("="*70)
    
    env_phase2 = create_env_optimized()
    
    # Continue training with loaded weights
    model_phase2 = PPO(
        "MlpPolicy",
        env_phase2,
        verbose=1,
        learning_rate=1e-4,  # Lower learning rate for fine-tuning
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.98,
        ent_coef=0.01,  # Lower entropy for exploitation
    )
    
    # Transfer learning: use the learned policy
    model_phase2.set_parameters(model_phase1.get_parameters())
    
    print("Training Phase 2 (40k steps)...")
    model_phase2.learn(total_timesteps=40000, progress_bar=True)
    
    return model_phase2, env_phase2


def visualize_learned_policy(env, model, num_visualizations=5):
    """Visualize what the agent has learned"""
    print("\n" + "="*70)
    print("VISUALIZING LEARNED POLICY")
    print("="*70)
    
    for vis in range(num_visualizations):
        print(f"\nVisualization {vis + 1}:")
        
        # Start from different pixels
        if vis == 0:
            pixel = (env.image_width // 2, env.image_height // 2)  # Center
        elif vis == 1:
            pixel = (env.image_width // 4, env.image_height // 2)  # Left
        elif vis == 2:
            pixel = (3 * env.image_width // 4, env.image_height // 2)  # Right
        elif vis == 3:
            pixel = (env.image_width // 2, env.image_height // 4)  # Top
        else:
            pixel = (env.image_width // 2, 3 * env.image_height // 4)  # Bottom
        
        obs, info = env.reset(options={'pixel': pixel})
        print(f"  Starting pixel: {pixel}")
        
        done = False
        step = 0
        total_reward = 0
        path = []
        
        while not done and step < 8:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            
            if env.current_intersection and env.current_intersection.intersects:
                obj = env.current_intersection.object
                pos = env.current_intersection.point
                normal = env.current_intersection.normal
                
                obj_type = "LIGHT" if obj.id in [99, 100] else f"Sphere {obj.id}"
                material_type = "Reflective" if obj.material.reflective > 0.5 else "Matte"
                
                path.append({
                    'step': step,
                    'object': obj_type,
                    'position': (pos.x, pos.y, pos.z),
                    'normal': (normal.x, normal.y, normal.z),
                    'reward': reward,
                    'material': material_type
                })
                
                print(f"    Step {step}: Hit {obj_type} at ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
                print(f"        Reward: {reward:.3f}, Material: {material_type}")
                if obj.id in [99, 100]:
                    print(f"        *** LIGHT HIT! ***")
            else:
                print(f"    Step {step}: Missed (reward: {reward:.3f})")
        
        print(f"  Total reward: {total_reward:.3f}")
        print(f"  Termination: {info.get('reason', 'unknown')}")
        
        # Analyze the path
        if path:
            lights_hit = sum(1 for p in path if p['object'] == 'LIGHT')
            reflective_hits = sum(1 for p in path if p['material'] == 'Reflective')
            print(f"  Path analysis: {lights_hit} light(s) hit, {reflective_hits} reflective surface(s)")


def main_improved():
    """Main improved training routine"""
    print("="*80)
    print("RAY TRACING RL - ADVANCED TRAINING")
    print("="*80)
    
    # Choose training mode
    print("\nTraining Modes:")
    print("1. Standard training (faster)")
    print("2. Curriculum learning (better results, slower)")
    print("3. Continue from saved model")
    
    try:
        choice = int(input("\nSelect mode (1-3): "))
    except:
        choice = 1
    
    if choice == 1:
        # Standard training
        env = create_env_optimized()
        
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=2.5e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[128, 128])  # Larger network
        )
        
        print("\nTraining for 75,000 timesteps...")
        callback = RewardLoggerCallback()
        model.learn(total_timesteps=75000, callback=callback, progress_bar=True)
        model.save("ppo_raytracer_improved")
        
    elif choice == 2:
        # Curriculum learning
        model, env = train_with_curriculum()
        model.save("ppo_raytracer_curriculum")
        
    elif choice == 3:
        # Load and continue
        env = create_env_optimized()
        try:
            model = PPO.load("ppo_raytracer", env=env)
            print("Loaded existing model")
        except:
            print("No saved model found, starting fresh")
            model = PPO("MlpPolicy", env, verbose=1)
        
        additional_steps = int(input("Additional timesteps to train: "))
        model.learn(total_timesteps=additional_steps, progress_bar=True)
        model.save("ppo_raytracer_continued")
    
    # Analyze the trained agent
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)
    
    metrics = analyze_agent_behavior(env, model, num_episodes=500)
    
    # Visualize learned behavior
    visualize_learned_policy(env, model, num_visualizations=5)
    
    # Save analysis to file
    try:
        analysis_df = pd.DataFrame({
            'rewards': metrics['rewards'],
            'steps': metrics['steps'],
            'success': [1 if r > 0 else 0 for r in metrics['rewards']]
        })
        analysis_df.to_csv('agent_analysis.csv', index=False)
        print("\nAnalysis saved to 'agent_analysis.csv'")
    except:
        print("\nCould not save analysis (pandas not installed)")
    
    # Plot results
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Reward distribution
        axes[0, 0].hist(metrics['rewards'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Total Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Reward Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Steps distribution
        axes[0, 1].hist(metrics['steps'], bins=range(1, max(metrics['steps'])+2), 
                       edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Steps per Episode')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Episode Length Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Object hits
        obj_names = []
        hit_counts = []
        for obj_id, count in sorted(metrics['hit_counts'].items(), key=lambda x: x[1], reverse=True):
            name = {1: "Ground", 2: "Center", 3: "Left", 
                   99: "Light 1", 100: "Light 2"}.get(obj_id, f"Obj {obj_id}")
            obj_names.append(name)
            hit_counts.append(count)
        
        axes[0, 2].barh(obj_names[:8], hit_counts[:8])
        axes[0, 2].set_xlabel('Hit Count')
        axes[0, 2].set_title('Objects Hit (Top 8)')
        axes[0, 2].grid(True, alpha=0.3, axis='x')
        
        # 4. Success rate over time (moving average)
        success_rates = []
        window = 50
        for i in range(len(metrics['rewards']) - window + 1):
            window_rewards = metrics['rewards'][i:i+window]
            success_rate = sum(1 for r in window_rewards if r > 0) / window
            success_rates.append(success_rate)
        
        axes[1, 0].plot(success_rates)
        axes[1, 0].set_xlabel(f'Episode (window={window})')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_title('Success Rate Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Reward over time (moving average)
        reward_ma = []
        for i in range(len(metrics['rewards']) - window + 1):
            reward_ma.append(np.mean(metrics['rewards'][i:i+window]))
        
        axes[1, 1].plot(reward_ma)
        axes[1, 1].set_xlabel(f'Episode (window={window})')
        axes[1, 1].set_ylabel('Average Reward')
        axes[1, 1].set_title('Reward Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Termination reasons pie chart
        reasons = list(metrics['final_reasons'].keys())
        counts = list(metrics['final_reasons'].values())
        axes[1, 2].pie(counts, labels=reasons, autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title('Termination Reasons')
        
        plt.tight_layout()
        plt.savefig('advanced_analysis.png', dpi=100, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("\nMatplotlib not available for plotting")
    
    env.close()
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main_improved()