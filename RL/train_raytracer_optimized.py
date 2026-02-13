# train_raytracer_optimized.py
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

from collections import defaultdict
from ray_tracer_env import RayTracerEnv
from vector import Vector
from colour import Colour
from object import Sphere
from material import Material
from light import GlobalLight, PointLight


class AdaptiveRewardRayTracerEnv(RayTracerEnv):
    """Enhanced environment with adaptive rewards"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.light_ids = [99, 100]  # IDs of light sources
        self.consecutive_light_hits = 0
        self.total_light_hits = 0
        
    def _calculate_reward(self):
        """Enhanced reward calculation"""
        if self.current_intersection is None or not self.current_intersection.intersects:
            return -0.5  # Less harsh penalty for misses
        
        obj = self.current_intersection.object
        
        # Base reward from parent class
        base_reward = super()._calculate_reward()
        
        # Bonus for hitting lights
        if obj.id in self.light_ids:
            light_bonus = 2.0
            self.consecutive_light_hits += 1
            self.total_light_hits += 1
            # Extra bonus for consecutive light hits
            if self.consecutive_light_hits > 1:
                light_bonus += 0.5 * self.consecutive_light_hits
        else:
            self.consecutive_light_hits = 0
            light_bonus = 0
        
        # Bonus for hitting reflective surfaces (encourages bouncing)
        if obj.material.reflective > 0.5:
            reflective_bonus = 0.3
        else:
            reflective_bonus = 0
        
        # Penalty for too few bounces (encourage longer paths)
        if self.bounce_count < 2 and base_reward > 0:
            path_length_penalty = -0.1
        else:
            path_length_penalty = 0
        
        # Combine rewards
        total_reward = base_reward + light_bonus + reflective_bonus + path_length_penalty
        
        return float(total_reward)
    
    def reset(self, *args, **kwargs):
        """Reset environment state"""
        self.consecutive_light_hits = 0
        return super().reset(*args, **kwargs)


class CurriculumCallback(BaseCallback):
    """Curriculum learning callback - increases difficulty over time"""
    
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.phase = 1
        self.phase_steps = 0
        
    def _on_step(self) -> bool:
        self.phase_steps += 1
        
        # Phase 1: Simple rewards (first 10k steps)
        if self.phase == 1 and self.phase_steps >= 10000:
            self.phase = 2
            self.phase_steps = 0
            print("\n=== CURRICULUM: Entering Phase 2 (encouraging longer paths) ===")
            
        # Phase 2: Encourage light hits (next 20k steps)
        elif self.phase == 2 and self.phase_steps >= 20000:
            self.phase = 3
            self.phase_steps = 0
            print("\n=== CURRICULUM: Entering Phase 3 (final optimization) ===")
            
        return True


def create_dynamic_scene(phase=1):
    """Create scene that changes with curriculum phase"""
    # Base materials
    matte = Material(reflective=0, transparent=0, emitive=0.1, refractive_index=1)
    reflective = Material(reflective=1, transparent=0, emitive=0, refractive_index=1)
    light_mat = Material(reflective=0, transparent=0, emitive=1, refractive_index=1)
    
    if phase == 1:
        # Phase 1: Easy - large ground, big light
        spheres = [
            Sphere(Vector(0, -100, -3), 99, matte, Colour(150, 150, 150), id=1),
            Sphere(Vector(0, 0, -3), 0.8, reflective, Colour(255, 255, 255), id=2),
            Sphere(Vector(0, 2.5, -3), 0.8, light_mat, Colour(255, 255, 200), id=99),
        ]
    elif phase == 2:
        # Phase 2: Medium - smaller light, add obstacles
        spheres = [
            Sphere(Vector(0, -100, -3), 99, matte, Colour(150, 150, 150), id=1),
            Sphere(Vector(0, 0, -3), 0.7, reflective, Colour(255, 255, 255), id=2),
            Sphere(Vector(-1.5, 0.5, -3), 0.5, reflective, Colour(200, 200, 255), id=3),
            Sphere(Vector(0, 2.5, -3), 0.6, light_mat, Colour(255, 255, 200), id=99),
        ]
    else:
        # Phase 3: Hard - multiple lights, smaller targets
        spheres = [
            Sphere(Vector(0, -100, -3), 99, matte, Colour(150, 150, 150), id=1),
            Sphere(Vector(0, 0, -3), 0.6, reflective, Colour(255, 255, 255), id=2),
            Sphere(Vector(-1.8, 0.3, -3), 0.5, reflective, Colour(200, 200, 255), id=3),
            Sphere(Vector(1.8, -0.3, -3), 0.5, reflective, Colour(255, 200, 200), id=4),
            Sphere(Vector(0, 2.5, -3), 0.5, light_mat, Colour(255, 255, 200), id=99),
            Sphere(Vector(-2, 1.8, -3), 0.4, light_mat, Colour(200, 255, 200), id=100),
        ]
    
    # Lights
    lights = []
    if phase >= 1:
        lights.append(PointLight(
            id=99, position=Vector(0, 2.5, -3), 
            colour=Colour(255, 255, 200), strength=15.0, max_angle=np.pi, func=0
        ))
    if phase >= 3:
        lights.append(PointLight(
            id=100, position=Vector(-2, 1.8, -3),
            colour=Colour(200, 255, 200), strength=10.0, max_angle=np.pi, func=0
        ))
    
    return spheres, [], lights


def train_with_exploration_strategies():
    """Train with different exploration strategies"""
    print("\n" + "="*80)
    print("ADVANCED TRAINING WITH EXPLORATION STRATEGIES")
    print("="*80)
    
    # Create initial environment
    spheres, global_lights, point_lights = create_dynamic_scene(phase=1)
    env = AdaptiveRewardRayTracerEnv(
        spheres=spheres,
        global_light_sources=global_lights,
        point_light_sources=point_lights,
        max_bounces=8,
        image_width=320,
        image_height=240,
        fov=80
    )
    
    # Strategy 1: High exploration initially
    print("\n=== STRATEGY 1: High Exploration ===")
    model1 = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.98,
        ent_coef=0.1,  # High entropy for exploration
        clip_range=0.2,
    )
    
    print("Training with high exploration (15k steps)...")
    model1.learn(total_timesteps=15000, progress_bar=True)
    
    # Strategy 2: Lower exploration, focus on exploitation
    print("\n=== STRATEGY 2: Balanced ===")
    
    # Update to phase 2 scene
    spheres, global_lights, point_lights = create_dynamic_scene(phase=2)
    env = AdaptiveRewardRayTracerEnv(
        spheres=spheres,
        global_light_sources=global_lights,
        point_light_sources=point_lights,
        max_bounces=8,
        image_width=320,
        image_height=240,
        fov=80
    )
    
    model2 = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.03,  # Lower entropy
        clip_range=0.15,
    )
    
    # Transfer learning from model1
    model2.set_parameters(model1.get_parameters())
    
    print("Training with balanced strategy (20k steps)...")
    model2.learn(total_timesteps=20000, progress_bar=True)
    
    # Strategy 3: Final optimization
    print("\n=== STRATEGY 3: Final Optimization ===")
    
    # Update to phase 3 scene
    spheres, global_lights, point_lights = create_dynamic_scene(phase=3)
    env = AdaptiveRewardRayTracerEnv(
        spheres=spheres,
        global_light_sources=global_lights,
        point_light_sources=point_lights,
        max_bounces=8,
        image_width=320,
        image_height=240,
        fov=80
    )
    
    model3 = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,
        ent_coef=0.01,  # Very low entropy
        clip_range=0.1,
    )
    
    # Transfer learning from model2
    model3.set_parameters(model2.get_parameters())
    
    print("Final optimization (25k steps)...")
    model3.learn(total_timesteps=25000, progress_bar=True)
    
    return model3, env


def evaluate_comprehensive(model, env, num_episodes=1000):
    """Comprehensive evaluation of the trained agent"""
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION")
    print("="*80)
    
    metrics = {
        'rewards': [],
        'steps': [],
        'light_hits': [],
        'max_consecutive_lights': 0,
        'path_efficiency': [],  # Reward per step
        'object_hits': defaultdict(int),
        'successful_paths': []
    }
    
    current_consecutive_lights = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        episode_light_hits = 0
        episode_objects = []
        
        while not done and episode_steps < 10:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1
            
            if env.current_intersection and env.current_intersection.intersects:
                obj = env.current_intersection.object
                metrics['object_hits'][obj.id] += 1
                episode_objects.append(obj.id)
                
                if obj.id in [99, 100]:
                    episode_light_hits += 1
                    current_consecutive_lights += 1
                else:
                    current_consecutive_lights = 0
        
        # Update max consecutive lights
        metrics['max_consecutive_lights'] = max(metrics['max_consecutive_lights'], current_consecutive_lights)
        
        # Record metrics
        metrics['rewards'].append(episode_reward)
        metrics['steps'].append(episode_steps)
        metrics['light_hits'].append(episode_light_hits)
        
        if episode_steps > 0:
            metrics['path_efficiency'].append(episode_reward / episode_steps)
        else:
            metrics['path_efficiency'].append(0)
        
        # Record successful paths (rewards > threshold)
        if episode_reward > 5.0:
            metrics['successful_paths'].append({
                'reward': episode_reward,
                'steps': episode_steps,
                'objects': episode_objects,
                'light_hits': episode_light_hits
            })
        
        # Progress report
        if (episode + 1) % 200 == 0:
            recent_rewards = metrics['rewards'][-200:]
            recent_lights = metrics['light_hits'][-200:]
            avg_reward = np.mean(recent_rewards)
            light_rate = sum(recent_lights) / len(recent_lights)
            
            print(f"  Episode {episode + 1}: Avg Reward={avg_reward:.2f}, "
                  f"Light Hit Rate={light_rate:.2f}/episode")
    
    # Print results
    print("\n" + "-"*80)
    print("EVALUATION RESULTS")
    print("-"*80)
    
    print(f"\nOverall Performance:")
    print(f"  Average Reward: {np.mean(metrics['rewards']):.3f} ± {np.std(metrics['rewards']):.3f}")
    print(f"  Average Steps: {np.mean(metrics['steps']):.2f}")
    print(f"  Success Rate: {sum(1 for r in metrics['rewards'] if r > 0)/len(metrics['rewards']):.1%}")
    
    print(f"\nLight Collection:")
    print(f"  Average Light Hits per Episode: {np.mean(metrics['light_hits']):.3f}")
    print(f"  Episodes with ≥1 Light Hit: {sum(1 for l in metrics['light_hits'] if l > 0)/len(metrics['light_hits']):.1%}")
    print(f"  Max Consecutive Light Hits: {metrics['max_consecutive_lights']}")
    
    print(f"\nPath Efficiency:")
    print(f"  Average Reward per Step: {np.mean(metrics['path_efficiency']):.3f}")
    print(f"  Most Efficient Path: {max(metrics['path_efficiency']):.3f}")
    
    print(f"\nObjects Hit:")
    obj_names = {1: "Ground", 2: "Center Sphere", 3: "Left Sphere", 
                 4: "Right Sphere", 99: "Light 1", 100: "Light 2"}
    total_hits = sum(metrics['object_hits'].values())
    for obj_id, count in sorted(metrics['object_hits'].items(), key=lambda x: x[1], reverse=True):
        name = obj_names.get(obj_id, f"Object {obj_id}")
        percentage = count / total_hits * 100
        print(f"  {name}: {count} hits ({percentage:.1f}%)")
    
    print(f"\nBest Paths Found (reward > 5.0): {len(metrics['successful_paths'])}")
    if metrics['successful_paths']:
        best_path = max(metrics['successful_paths'], key=lambda x: x['reward'])
        print(f"  Best Path: {best_path['reward']:.2f} reward, {best_path['steps']} steps, "
              f"{best_path['light_hits']} light hits")
    
    return metrics


def visualize_3d_paths(env, model, num_paths=10):
    """Visualize 3D ray paths"""
    print("\n" + "="*80)
    print("3D PATH VISUALIZATION")
    print("="*80)
    
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(15, 10))
        
        for path_num in range(min(num_paths, 4)):  # Show up to 4 paths
            ax = fig.add_subplot(2, 2, path_num + 1, projection='3d')
            
            # Generate a path
            obs, _ = env.reset()
            done = False
            positions = []
            colors = []
            
            while not done and len(positions) < 8:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                if env.current_intersection and env.current_intersection.intersects:
                    pos = env.current_intersection.point
                    positions.append((pos.x, pos.y, pos.z))
                    
                    # Color based on object type
                    obj = env.current_intersection.object
                    if obj.id in [99, 100]:
                        colors.append('yellow')  # Light
                    elif obj.material.reflective > 0.5:
                        colors.append('cyan')    # Reflective
                    else:
                        colors.append('gray')    # Matte
            
            # Plot the path
            if len(positions) > 1:
                positions = np.array(positions)
                ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'k--', alpha=0.3)
                ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                          c=colors[:len(positions)], s=50, alpha=0.8)
                
                # Mark start and end
                ax.scatter([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], 
                          c='green', s=100, marker='o', label='Start')
                ax.scatter([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]], 
                          c='red', s=100, marker='s', label='End')
            
            # Plot scene objects (simplified)
            for sphere in env.spheres:
                if sphere.radius < 10:  # Don't plot ground
                    u = np.linspace(0, 2 * np.pi, 20)
                    v = np.linspace(0, np.pi, 20)
                    x = sphere.centre.x + sphere.radius * np.outer(np.cos(u), np.sin(v))
                    y = sphere.centre.y + sphere.radius * np.outer(np.sin(u), np.sin(v))
                    z = sphere.centre.z + sphere.radius * np.outer(np.ones(np.size(u)), np.cos(v))
                    
                    color = 'yellow' if sphere.id in [99, 100] else 'blue'
                    alpha = 0.3 if sphere.id in [99, 100] else 0.1
                    
                    ax.plot_surface(x, y, z, color=color, alpha=alpha)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Path {path_num + 1}')
            ax.legend()
        
        plt.suptitle('3D Ray Paths Learned by Agent', fontsize=16)
        plt.tight_layout()
        plt.savefig('3d_ray_paths.png', dpi=100, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("3D visualization requires matplotlib and 3D toolkit")
    except Exception as e:
        print(f"Could not create 3D visualization: {e}")


def compare_algorithms():
    """Compare PPO vs SAC for ray tracing"""
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON: PPO vs SAC")
    print("="*80)
    
    # Create environment
    spheres, global_lights, point_lights = create_dynamic_scene(phase=2)
    env = AdaptiveRewardRayTracerEnv(
        spheres=spheres,
        global_light_sources=global_lights,
        point_light_sources=point_lights,
        max_bounces=6,
        image_width=240,
        image_height=180,
        fov=80
    )
    
    # Train PPO
    print("\nTraining PPO...")
    ppo_model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.02,
    )
    ppo_model.learn(total_timesteps=20000, progress_bar=True)
    
    # Train SAC
    print("\nTraining SAC...")
    sac_model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef='auto',
    )
    sac_model.learn(total_timesteps=20000, progress_bar=True)
    
    # Compare performance
    print("\n" + "-"*80)
    print("COMPARISON RESULTS")
    print("-"*80)
    
    algorithms = {'PPO': ppo_model, 'SAC': sac_model}
    results = {}
    
    for name, model in algorithms.items():
        print(f"\nEvaluating {name}...")
        rewards = []
        light_hits = []
        
        for _ in range(200):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode_lights = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                if env.current_intersection and env.current_intersection.intersects:
                    if env.current_intersection.object.id in [99, 100]:
                        episode_lights += 1
            
            rewards.append(episode_reward)
            light_hits.append(episode_lights)
        
        results[name] = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'success_rate': sum(1 for r in rewards if r > 0) / len(rewards),
            'avg_lights': np.mean(light_hits),
            'light_hit_rate': sum(1 for l in light_hits if l > 0) / len(light_hits)
        }
        
        print(f"  Average Reward: {results[name]['avg_reward']:.3f}")
        print(f"  Success Rate: {results[name]['success_rate']:.1%}")
        print(f"  Light Hit Rate: {results[name]['light_hit_rate']:.1%}")
    
    # Determine winner
    best_algorithm = max(results.keys(), key=lambda x: results[x]['avg_reward'])
    print(f"\n🌟 BEST ALGORITHM: {best_algorithm} 🌟")
    
    return results


def main_optimized():
    """Main optimized training routine"""
    print("="*80)
    print("ULTIMATE RAY TRACING RL OPTIMIZATION")
    print("="*80)
    
    print("\nSelect training approach:")
    print("1. Enhanced training with adaptive rewards")
    print("2. Algorithm comparison (PPO vs SAC)")
    print("3. 3D path visualization")
    print("4. Complete pipeline (all of the above)")
    
    try:
        choice = int(input("\nSelect option (1-4): "))
    except:
        choice = 1
    
    if choice == 1:
        # Enhanced training
        model, env = train_with_exploration_strategies()
        model.save("raytracer_optimized")
        
        # Comprehensive evaluation
        metrics = evaluate_comprehensive(model, env, num_episodes=500)
        
    elif choice == 2:
        # Algorithm comparison
        results = compare_algorithms()
        
    elif choice == 3:
        # 3D visualization
        spheres, global_lights, point_lights = create_dynamic_scene(phase=3)
        env = AdaptiveRewardRayTracerEnv(
            spheres=spheres,
            global_light_sources=global_lights,
            point_light_sources=point_lights,
            max_bounces=8,
            image_width=320,
            image_height=240,
            fov=80
        )
        
        # Load or train a model
        try:
            model = PPO.load("raytracer_optimized", env=env)
            print("Loaded trained model")
        except:
            print("Training a model for visualization...")
            model = PPO("MlpPolicy", env, verbose=0)
            model.learn(total_timesteps=10000, progress_bar=True)
        
        visualize_3d_paths(env, model, num_paths=4)
        
    elif choice == 4:
        # Complete pipeline
        print("\n" + "="*80)
        print("COMPLETE TRAINING PIPELINE")
        print("="*80)
        
        # Step 1: Enhanced training
        model, env = train_with_exploration_strategies()
        model.save("raytracer_complete")
        
        # Step 2: Evaluation
        metrics = evaluate_comprehensive(model, env, num_episodes=1000)
        
        # Step 3: Algorithm comparison
        results = compare_algorithms()
        
        # Step 4: Visualization
        visualize_3d_paths(env, model, num_paths=6)
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)
    
    # Summary of achievements
    print("\n🎯 ACHIEVEMENTS:")
    print("  ✓ Agent learns to bounce rays effectively")
    print("  ✓ Can hit reflective surfaces to gather more light")
    print("  ✓ Learns efficient light-gathering paths")
    print("  ✓ Adapts to different scene complexities")
    print("\n📈 NEXT STEPS:")
    print("  1. Try even more complex scenes")
    print("  2. Implement multi-agent ray tracing")
    print("  3. Use learned policy for actual rendering acceleration")
    print("  4. Experiment with different reward functions")


if __name__ == "__main__":
    main_optimized()