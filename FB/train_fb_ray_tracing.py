"""
Training script for FB ray tracing agent.

This trains the FB agent in the same environment as your RL agent
for fair comparison.
"""

import numpy as np
from sklearn import metrics
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime
from typing import Dict

from fb_ray_tracing import FBResearchAgent as FBRayTracingAgent, FBConfig
from vector import Vector
from colour import Colour
from object import Sphere
from material import Material
from ray import Ray, Intersection
from train_rl import create_scene_for_rl

def create_training_scene_varied():
    """Train FB on the SAME scene as RL"""
    
    scenes = []
    
    # Use multiple variations of your scene
    for scene_idx in range(5):
        # Get your exact scene
        spheres = create_scene_for_rl()
        
        # Add small variations (optional)
        if scene_idx > 0:
            for sphere in spheres:
                if sphere.id != 7:  # Don't move the sun too much
                    small_offset = Vector(
                        np.random.uniform(-0.1, 0.1),
                        np.random.uniform(-0.1, 0.1),
                        np.random.uniform(-0.1, 0.1)
                    )
                    sphere.centre = sphere.centre.addVector(small_offset)
        
        scenes.append(spheres)
    
    return scenes


class FBTrainer:
    """Trainer for FB ray tracing agent"""
    
    def __init__(self, config: FBConfig, output_dir: str = "./fb_training_outputs"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create agent - REMOVE obs_shape parameter
        self.agent = FBRayTracingAgent(config, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_light_hits': [],
            'episode_steps': [],
            'losses': [],
            'fb_guided_ratio': [],
            'planning_success_rate': []
        }
        
        # Create varied scenes for training
        self.training_scenes = create_training_scene_varied()
        
        print(f"FB Trainer initialized with {len(self.training_scenes)} training scenes")
        print(f"Output directory: {self.output_dir}")
    
    def collect_initial_experience(self, num_episodes: int = 1000):
        """Collect initial random experience for FB learning"""
        print(f"\nCollecting {num_episodes} episodes of initial experience...")
        
        scene_idx = 0
        for episode in tqdm(range(num_episodes)):
            # Cycle through scenes
            spheres = self.training_scenes[scene_idx % len(self.training_scenes)]
            scene_idx += 1
            
            # Simplified environment interaction
            self._collect_random_episode(spheres)
        
        print(f"Initial experience collected: {len(self.agent.fb_learner.replay_buffer)} transitions")
    
    def _collect_random_episode(self, spheres, max_steps: int = 10):
        """Collect random episode for given scene"""
        # Start from random pixel
        camera_pos = Vector(0, 0, 5)
        
        # Random initial ray
        init_theta = np.random.uniform(0, np.pi/4)
        init_phi = np.random.uniform(0, 2*np.pi)
        init_dir = Vector(
            np.sin(init_theta) * np.cos(init_phi),
            np.sin(init_theta) * np.sin(init_phi),
            -np.cos(init_theta)
        ).normalise()
        
        current_ray = Ray(camera_pos, init_dir)
        accumulated_color = Colour(0, 0, 0)
        
        for step in range(max_steps):
            # Find intersection
            intersection = current_ray.nearestSphereIntersect(spheres, max_bounces=3)
            
            # Create observation
            obs = self.agent.create_observation(intersection, current_ray, step, accumulated_color)
            
            # Random action
            action = np.random.uniform(-1, 1, 2)
            
            # Apply action (simplified - in practice, use action to generate new ray)
            if intersection and intersection.intersects:
                # Simple bounce
                new_dir = Vector(
                    np.random.uniform(-1, 1),
                    np.random.uniform(-1, 1),
                    np.random.uniform(-1, 1)
                ).normalise()
                next_ray = Ray(intersection.point, new_dir)
                
                # Create next observation
                next_obs = self.agent.create_observation(None, next_ray, step + 1, accumulated_color)
                
                # Simple reward: 1 if hit light, 0 otherwise
                reward = 1.0 if intersection.object.id in [99, 100] else 0.0
                
                # Store transition
                is_light_hit = reward > 0
                self.agent.record_success(obs, action, next_obs, reward, is_light_hit)       

                #if reward > 0:
                    #self.agent.record_light_hit(obs)
                
                current_ray = next_ray
            else:
                # Missed - episode ends
                break
    
    def train(self, num_episodes: int = 5000, train_every: int = 10):
        """Main training loop"""
        print(f"\n{'='*80}")
        print(f"TRAINING FB RAY TRACING AGENT")
        print(f"{'='*80}")
        
        scene_idx = 0
        
        for episode in tqdm(range(num_episodes)):
            # Select scene
            spheres = self.training_scenes[scene_idx % len(self.training_scenes)]
            
            # Run episode - PASS scene_idx
            episode_stats = self._run_training_episode(spheres, episode, scene_idx)
            
            # Update statistics
            for key, value in episode_stats.items():
                if key in self.training_stats:
                    self.training_stats[key].append(value)
            
            # Train FB learner
            if episode % train_every == 0 and episode > 0:
                for _ in range(5):  # Multiple training steps
                    loss_info = self.agent.fb_learner.train_step()
                    if loss_info:
                        self.training_stats['losses'].append(loss_info)
            
            # Log progress
            if episode % 100 == 0:
                self._log_progress(episode)
            
            # Save checkpoint
            if episode % 1000 == 0 and episode > 0:
                self.save_checkpoint(episode)
            
            scene_idx += 1  # Increment scene_idx
        
        print("\nTraining completed!")
        self._save_training_results()
    
    def _run_training_episode(self, spheres, episode_num: int, scene_idx: int, max_steps: int = 15) -> Dict:
        """Run one training episode"""
        # Simplified - use your full RayTracerEnv for real training
        
        camera_pos = Vector(0, 0, 5)
        
        # Vary camera for diversity
        if episode_num % 10 == 0:
            camera_pos = Vector(
                np.random.uniform(-2, 2),
                np.random.uniform(0, 3),
                np.random.uniform(3, 8)
            )
        
        # Initial ray
        init_theta = np.pi/6  # 30 degrees down
        init_phi = np.random.uniform(0, 2*np.pi)
        init_dir = Vector(
            np.sin(init_theta) * np.cos(init_phi),
            np.sin(init_theta) * np.sin(init_phi),
            -np.cos(init_theta)
        ).normalise()
        
        current_ray = Ray(camera_pos, init_dir)
        accumulated_color = Colour(0, 0, 0)
        
        episode_reward = 0
        episode_light_hits = 0
        
        for step in range(max_steps):
            # Find intersection
            intersection = current_ray.nearestSphereIntersect(spheres, max_bounces=3)
            
            # Create observation
            obs = self.agent.create_observation(intersection, current_ray, step, accumulated_color)
            
            # Choose action using FB agent
            exploration = (episode_num < 1000)  # More exploration early
            action, action_info = self.agent.choose_direction_research(
                obs, 
                scene_context=f"scene_{scene_idx}", 
                exploration_phase="explore" if exploration else "exploit"
            )
            
            # Convert action to direction (theta, phi to vector)
            theta = (action[0] + 1) * np.pi/4  # Map [-1,1] to [0, pi/2]
            phi = action[1] * np.pi  # Map [-1,1] to [-pi, pi]
            
            new_dir = Vector(
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ).normalise()
            
            if intersection and intersection.intersects:
                # Transform to local coordinate system around normal
                normal = intersection.normal
                
                # Create coordinate frame
                if abs(normal.z) < 0.9:
                    tangent = Vector(0, 0, 1).crossProduct(normal).normalise()
                else:
                    tangent = Vector(1, 0, 0).crossProduct(normal).normalise()
                
                bitangent = normal.crossProduct(tangent).normalise()
                
                # Transform direction to world space
                world_dir = Vector(
                    new_dir.x * tangent.x + new_dir.y * bitangent.x + new_dir.z * normal.x,
                    new_dir.x * tangent.y + new_dir.y * bitangent.y + new_dir.z * normal.y,
                    new_dir.x * tangent.z + new_dir.y * bitangent.z + new_dir.z * normal.z
                ).normalise()
                
                next_ray = Ray(intersection.point.addVector(normal.scaleByLength(0.001)), world_dir)
                
                # Reward
                if intersection.object.id in [99, 100]:
                    reward = 10.0  # High reward for light hit
                    episode_light_hits += 1
                    #self.agent.record_light_hit(obs)
                else:
                    # Reward based on material properties
                    if intersection.object.material.reflective:
                        reward = 0.5  # Reflective surfaces often lead to interesting paths
                    else:
                        reward = 0.1  # Matte surfaces less interesting
                
                episode_reward += reward
                
                # Create next observation
                next_obs = self.agent.create_observation(None, next_ray, step + 1, accumulated_color)
                
                # Store transition
                is_light_hit = intersection.object.id in [99, 100] if intersection and intersection.intersects else False
                self.agent.record_success(obs, action, next_obs, reward, is_light_hit)
                
                current_ray = next_ray
            else:
                # Missed - negative reward
                reward = -0.1
                episode_reward += reward
                break
        
        # Get agent statistics
        metrics = self.agent.get_research_metrics()
        stats = metrics.get('performance', {})
        
        return {
            'reward': episode_reward,
            'light_hits': episode_light_hits,
            'steps': step + 1,
            'fb_guided_ratio': stats.get('fb_guided_ratio', 0),
            'planning_success_rate': stats.get('planning_success_rate', 0)
        }
    
    def _log_progress(self, episode: int):
        """Log training progress"""
        if len(self.training_stats['episode_rewards']) == 0:
            return
        
        recent_rewards = self.training_stats['episode_rewards'][-100:]
        recent_lights = self.training_stats['episode_light_hits'][-100:]
        
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        avg_lights = np.mean(recent_lights) if recent_lights else 0
        
        print(f"\nEpisode {episode}:")
        print(f"  Avg reward (last 100): {avg_reward:.3f}")
        print(f"  Avg light hits: {avg_lights:.2f}")
        print(f"  Buffer size: {len(self.agent.fb_learner.replay_buffer)}")
        print(f"  Light encodings: {len(self.agent.light_encodings)}")
        
        # Print loss info if available
        if self.training_stats['losses']:
            recent_losses = self.training_stats['losses'][-10:]
            if recent_losses:
                avg_loss = np.mean([l.get('total_loss', 0) for l in recent_losses])
                print(f"  Avg loss: {avg_loss:.4f}")
    
    def save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        checkpoint_path = self.output_dir / f"fb_checkpoint_ep{episode}.pth"
        
        # Use agent's save method
        self.agent.save(str(checkpoint_path))
        
        # Save training stats
        stats_path = self.output_dir / f"training_stats_ep{episode}.json"
        with open(stats_path, 'w') as f:
            json.dump({
                'episode': episode,
                'stats': {k: v[-1000:] for k, v in self.training_stats.items() if v}
            }, f, indent=2)
        
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    def _save_training_results(self):
        """Save final training results"""
        # Save final model
        final_path = self.output_dir / "fb_ray_tracer_final.pth"
        self.agent.save(str(final_path))
        
        # Save training curves
        self._plot_training_curves()
        
        # Save summary
        summary = {
            'config': self.config.__dict__,
            'final_stats': self.agent.get_statistics(),
            'training_summary': {
                'total_episodes': len(self.training_stats['episode_rewards']),
                'avg_final_reward': np.mean(self.training_stats['episode_rewards'][-100:]),
                'avg_light_hits': np.mean(self.training_stats['episode_light_hits'][-100:]),
                'max_reward': max(self.training_stats['episode_rewards']) if self.training_stats['episode_rewards'] else 0,
                'max_light_hits': max(self.training_stats['episode_light_hits']) if self.training_stats['episode_light_hits'] else 0
            }
        }
        
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nFinal model saved: {final_path}")
        print(f"Training summary: {summary_path}")
    
    def _plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Episode rewards
        rewards = self.training_stats['episode_rewards']
        if rewards:
            axes[0, 0].plot(rewards, alpha=0.3)
            # Moving average
            window = 100
            moving_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
            axes[0, 0].plot(moving_avg, linewidth=2, color='blue')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Light hits
        light_hits = self.training_stats['episode_light_hits']
        if light_hits:
            axes[0, 1].plot(light_hits, alpha=0.3, color='orange')
            window = 100
            moving_avg = [np.mean(light_hits[max(0, i-window):i+1]) for i in range(len(light_hits))]
            axes[0, 1].plot(moving_avg, linewidth=2, color='red')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Light Hits')
            axes[0, 1].set_title('Light Hits per Episode')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. FB guided ratio
        fb_ratios = self.training_stats['fb_guided_ratio']
        if fb_ratios:
            axes[0, 2].plot(fb_ratios, linewidth=2, color='green')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Ratio')
            axes[0, 2].set_title('FB-Guided Decision Ratio')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].set_ylim([0, 1])
        
        # 4. Losses
        losses = self.training_stats['losses']
        if losses and len(losses) > 10:
            total_losses = [l.get('total_loss', 0) for l in losses]
            fb_losses = [l.get('fb_loss', 0) for l in losses]
            
            axes[1, 0].plot(total_losses, label='Total', linewidth=2)
            axes[1, 0].plot(fb_losses, label='FB', alpha=0.7)
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Training Losses')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Planning success
        planning_rates = self.training_stats['planning_success_rate']
        if planning_rates:
            axes[1, 1].plot(planning_rates, linewidth=2, color='purple')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Success Rate')
            axes[1, 1].set_title('Planning Success Rate')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim([0, 1])
        
        # 6. Cumulative rewards
        if rewards:
            cumulative = np.cumsum(rewards)
            axes[1, 2].plot(cumulative, linewidth=2, color='brown')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Cumulative Reward')
            axes[1, 2].set_title('Cumulative Performance')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('FB Ray Tracing Training Progress', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        plot_path = self.output_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved: {plot_path}")


def main():
    """Main training function"""
    print("FB RAY TRACING TRAINING")
    print("="*80)
    
    # Configuration
    config = FBConfig(
        z_dim=64,
        f_hidden_dim=512,
        b_hidden_dim=256,
        num_forward_heads=3,  # Fixed attribute name
        num_layers=2,
        learning_rate=3e-4,
        batch_size=256,
        buffer_capacity=100000,
        fb_weight=1.0,
        contrastive_weight=0.3,
        predictive_weight=0.2,
        norm_weight=0.1
    )
    
    # Create output directory with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./fb_training_outputs_{timestamp}"
    
    # Create and run trainer
    trainer = FBTrainer(config, output_dir)
    
    # Phase 1: Initial experience collection
    trainer.collect_initial_experience(num_episodes=2000)
    
    # Phase 2: Main training
    trainer.train(num_episodes=5000, train_every=10)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    
    # Show final statistics
    final_stats = trainer.agent.get_statistics()
    print("\nFINAL AGENT STATISTICS:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nModel saved in: {output_dir}")
    print("\nNext: Run comparison with RL using compare_fb_vs_rl()")


if __name__ == "__main__":
    main()