"""
Train FB agent on YOUR custom scene - FIXED OBSERVATION DIMENSIONS
"""

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import random
import math

from fb_ray_tracing import FBResearchAgent, FBConfig
from vector import Vector
from colour import Colour
from object import Sphere
from material import Material
from ray import Ray

def create_your_scene():
    """Create YOUR exact custom scene for FB training"""
    print("Creating YOUR custom scene for FB training...")
    
    # Your materials
    base_material = Material(reflective=False)
    reflective_material = Material(reflective=True)
    glass = Material(reflective=False, transparent=True, refractive_index=1.52)
    emitive_material = Material(emitive=True)
    
    # YOUR exact spheres
    spheres = [
        # Glass sphere
        Sphere(id=1, centre=Vector(-0.8, 0.6, 0), radius=0.3, 
               material=glass, colour=Colour(255, 100, 100)),
        # Large blue sphere
        Sphere(id=2, centre=Vector(0.8, -0.8, -10), radius=2.2,
               material=base_material, colour=Colour(204, 204, 255)),
        # Small blue sphere
        Sphere(id=3, centre=Vector(0.3, 0.34, 0.1), radius=0.2,
               material=base_material, colour=Colour(0, 51, 204)),
        # Reflective sphere
        Sphere(id=4, centre=Vector(5.6, 3, -2), radius=5,
               material=reflective_material, colour=Colour(153, 51, 153)),
        # Green sphere
        Sphere(id=5, centre=Vector(-0.8, -0.8, -0.2), radius=0.25,
               material=base_material, colour=Colour(153, 204, 0)),
        # Background sphere
        Sphere(id=6, centre=Vector(-3, 10, -75), radius=30,
               material=base_material, colour=Colour(255, 204, 102)),
        # SUN - the light source!
        Sphere(id=7, centre=Vector(-0.6, 0.2, 6), radius=0.1,
               material=emitive_material, colour=Colour(255, 255, 204))
    ]
    
    print(f"Scene created with {len(spheres)} spheres")
    print(f"Sun is at: ({spheres[-1].centre.x}, {spheres[-1].centre.y}, {spheres[-1].centre.z})")
    
    return spheres

class YourSceneFBTrainer:
    """Train FB agent on your scene - PROPER TRAINING with correct observation dimensions"""
    
    def __init__(self):
        # FB Configuration - Tuned for light finding
        # Must match the dimensions expected by the FB agent!
        self.config = FBConfig(
            z_dim=32,
            f_hidden_dim=256,
            b_hidden_dim=128,
            num_forward_heads=2,
            num_layers=2,
            learning_rate=3e-4,
            batch_size=128,
            buffer_capacity=50000,
            fb_weight=1.0,
            contrastive_weight=0.5,
            predictive_weight=0.3,
            max_bounces=6
        )
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"./fb_your_scene_{timestamp}")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create agent
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent = FBResearchAgent(self.config, device=self.device)
        
        # Your scene
        self.scene = create_your_scene()
        self.sun_position = Vector(-0.6, 0.2, 6)
        
        # Training stats
        self.training_stats = {
            'losses': [],
            'light_hits': [],
            'sun_hits': 0,
            'total_steps': 0
        }
        
        print(f"\nFB Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Output: {self.output_dir}")
        print(f"  Sun position: ({self.sun_position.x}, {self.sun_position.y}, {self.sun_position.z})")
        print(f"  Training goal: Learn to find the small sun (radius=0.1)")
        print(f"  Observation dimensions: {self.agent.obs_dim}")  # Should be 22
    
    def create_full_observation(self, intersection_point, normal, ray_dir, bounce_count, accumulated_color, material=None, sphere_id=None):
        """Create a full 22-dimension observation matching FB agent expectations"""
        # Default material if not provided
        if material is None:
            material = Material(reflective=False)
        
        # Default color if not provided
        if accumulated_color is None:
            accumulated_color = Colour(0, 0, 0)
        
        # Sun direction
        sun_dir = self.sun_position.subtractVector(intersection_point).normalise()
        
        # Material properties
        is_reflective = float(getattr(material, 'reflective', False))
        is_transparent = float(getattr(material, 'transparent', False))
        is_emitive = float(getattr(material, 'emitive', False))
        refractive_index = float(getattr(material, 'refractive_index', 1.0))
        
        # Normalize color
        color_norm = np.array([
            accumulated_color.r / 255.0,
            accumulated_color.g / 255.0,
            accumulated_color.b / 255.0
        ], dtype=np.float32)
        
        # Create full 22-dimensional observation
        obs = np.array([
            # Position (3)
            intersection_point.x, intersection_point.y, intersection_point.z,
            # Direction (3)
            ray_dir.x, ray_dir.y, ray_dir.z,
            # Normal (3)
            normal.x, normal.y, normal.z,
            # Material properties (4)
            is_reflective, is_transparent, is_emitive, refractive_index,
            # Current color (3)
            color_norm[0], color_norm[1], color_norm[2],
            # Bounce and history (3)
            float(bounce_count) / self.config.max_bounces,
            0.0,  # through_count
            float(sphere_id if sphere_id is not None else 0) / 100.0,
            # Sun direction (3) - CRITICAL for learning!
            sun_dir.x, sun_dir.y, sun_dir.z
        ], dtype=np.float32)
        
        # Ensure it's exactly 22 dimensions
        assert len(obs) == 22, f"Observation has {len(obs)} dimensions, expected 22"
        
        return obs
    
    def get_optimal_sun_direction(self, intersection_point, normal):
        """Get the optimal direction to hit the sun from this point"""
        # Direct line to sun center
        to_sun_center = self.sun_position.subtractVector(intersection_point)
        
        # For a small sun, we need to aim precisely
        # But also account for surface normal (can't go through surface)
        
        if to_sun_center.dotProduct(normal) < 0:
            # Sun is behind surface, reflect
            to_sun_reflected = to_sun_center.reflectInVector(normal)
            return to_sun_reflected.normalise()
        
        return to_sun_center.normalise()
    
    def generate_training_episode(self, episode_idx):
        """Generate one training episode with full observation"""
        # Start from random point in scene
        # Choose random sphere surface point
        sphere = random.choice(self.scene)
        sphere_id = sphere.id
        sphere_material = sphere.material
        
        # Generate random point on sphere surface
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, math.pi)
        
        point_on_sphere = Vector(
            sphere.centre.x + sphere.radius * math.sin(phi) * math.cos(theta),
            sphere.centre.y + sphere.radius * math.sin(phi) * math.sin(theta),
            sphere.centre.z + sphere.radius * math.cos(phi)
        )
        
        # Surface normal (pointing outward)
        normal = point_on_sphere.subtractVector(sphere.centre).normalise()
        
        # Generate random incoming ray direction
        incoming_theta = random.uniform(0, math.pi/2)
        incoming_phi = random.uniform(0, 2*math.pi)
        
        # Make sure ray direction is opposite to normal (hitting surface)
        if abs(normal.z) > 0.9:
            tangent = Vector(1, 0, 0)
        else:
            tangent = Vector(0, 0, 1).crossProduct(normal)
        tangent = tangent.normalise()
        bitangent = normal.crossProduct(tangent).normalise()
        
        local_dir = Vector(
            math.sin(incoming_theta) * math.cos(incoming_phi),
            math.sin(incoming_theta) * math.sin(incoming_phi),
            -math.cos(incoming_theta)  # Coming toward surface
        )
        
        ray_dir = Vector(
            local_dir.x * tangent.x + local_dir.y * bitangent.x + local_dir.z * normal.x,
            local_dir.x * tangent.y + local_dir.y * bitangent.y + local_dir.z * normal.y,
            local_dir.x * tangent.z + local_dir.y * bitangent.z + local_dir.z * normal.z
        ).normalise()
        
        # Create full observation (22 dimensions)
        bounce_count = random.randint(0, 3)
        accumulated_color = Colour(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        
        obs = self.create_full_observation(
            point_on_sphere, normal, ray_dir, bounce_count, 
            accumulated_color, sphere_material, sphere_id
        )
        
        # What's the optimal action? (direction toward sun)
        optimal_dir = self.get_optimal_sun_direction(point_on_sphere, normal)
        
        # Convert direction to action (spherical coordinates)
        # Action is [theta_angle, phi_angle] normalized to [-1, 1]
        optimal_theta = math.acos(max(-1, min(1, optimal_dir.z)))
        optimal_phi = math.atan2(optimal_dir.y, optimal_dir.x)
        
        # Normalize to [-1, 1]
        action = np.array([
            (optimal_theta / (math.pi/2)) * 2 - 1,  # theta ∈ [0, π/2] -> [-1, 1]
            optimal_phi / math.pi  # phi ∈ [-π, π] -> [-1, 1]
        ], dtype=np.float32)
        
        # Check if this action would actually hit sun
        # Create ray in optimal direction
        test_ray = Ray(point_on_sphere.addVector(normal.scaleByLength(0.001)), optimal_dir)
        
        # Check intersection with sun
        hit_sun = False
        for test_sphere in self.scene:
            if test_sphere.id == 7:  # Sun
                intersect = test_ray.sphereDiscriminant(test_sphere)
                if intersect and intersect.intersects:
                    hit_sun = True
                    self.training_stats['sun_hits'] += 1
                    break
        
        # Create next observation
        if hit_sun:
            # If we hit sun, create a "sun hit" observation
            # The sun itself (sphere_id=7, emitive material)
            sun_material = Material(emitive=True)
            next_obs = self.create_full_observation(
                point_on_sphere, normal, optimal_dir, bounce_count + 1,
                Colour(255, 255, 200),  # Bright sun color
                sun_material, 7  # Sun sphere ID
            )
        else:
            # Next observation is similar but with updated bounce count
            next_obs = self.create_full_observation(
                point_on_sphere, normal, optimal_dir, bounce_count + 1,
                accumulated_color, sphere_material, sphere_id
            )
        
        # Reward: high for hitting sun, low otherwise
        reward = 10.0 if hit_sun else 0.1
        
        # Record for training
        self.agent.record_success(obs, action, next_obs, reward, hit_sun)
        
        return hit_sun
    
    def collect_initial_data(self, num_episodes=500):
        """Collect initial experience"""
        print(f"\nCollecting {num_episodes} episodes of initial experience...")
        
        sun_hits = 0
        
        for episode in tqdm(range(num_episodes)):
            hit_sun = self.generate_training_episode(episode)
            if hit_sun:
                sun_hits += 1
            
            # Occasionally add pure exploration
            if episode % 10 == 0:
                self.add_exploration_data()
        
        print(f"  Sun hits during collection: {sun_hits}/{num_episodes} ({sun_hits/num_episodes*100:.1f}%)")
        print(f"  Buffer size: {len(self.agent.fb_learner.replay_buffer)}")
    
    def add_exploration_data(self):
        """Add some purely exploratory data with correct observation dimensions"""
        # Random point in space
        point = Vector(
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(-5, 5)
        )
        
        # Random normal
        normal = Vector(
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        ).normalise()
        
        # Random ray direction
        ray_dir = Vector(
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        ).normalise()
        
        # Random material
        material = Material(
            reflective=random.random() > 0.8,
            transparent=random.random() > 0.9,
            emitive=False
        )
        
        # Random accumulated color
        color = Colour(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        
        # Create full observation
        obs = self.create_full_observation(
            point, normal, ray_dir, random.randint(0, 3),
            color, material, random.randint(1, 10)
        )
        
        # Random action
        action = np.random.uniform(-1, 1, 2)
        
        # Create next observation (similar with slight changes)
        next_point = Vector(
            point.x + random.uniform(-0.1, 0.1),
            point.y + random.uniform(-0.1, 0.1),
            point.z + random.uniform(-0.1, 0.1)
        )
        
        next_obs = self.create_full_observation(
            next_point, normal, ray_dir, random.randint(1, 4),
            color, material, random.randint(1, 10)
        )
        
        # Small reward
        reward = random.uniform(-0.1, 0.1)
        
        self.agent.record_success(obs, action, next_obs, reward, False)
    
    def train(self, num_steps=3000):
        """Train FB agent"""
        print(f"\nTraining FB agent for {num_steps} steps...")
        
        losses = []
        light_hit_rates = []
        
        # Create progress bar
        pbar = tqdm(range(num_steps), desc="Training")
        
        for step in pbar:
            # Generate new experience every 5 steps
            if step % 5 == 0:
                hit_sun = self.generate_training_episode(step)
                if hit_sun:
                    self.training_stats['sun_hits'] += 1
            
            # Train if we have enough data
            if len(self.agent.fb_learner.replay_buffer) >= self.config.batch_size:
                stats = self.agent.fb_learner.train_step()
                loss = stats.get('total_loss', 0)
                losses.append(loss)
                
                # Calculate recent light hit rate
                recent_hits = self.training_stats['sun_hits']
                hit_rate = recent_hits / max(1, min(100, step))
                light_hit_rates.append(hit_rate)
                
                # Update progress bar
                if step % 100 == 0:
                    avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else loss
                    avg_hit_rate = np.mean(light_hit_rates[-100:]) if light_hit_rates else 0
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'hit_rate': f'{avg_hit_rate*100:.1f}%',
                        'buffer': len(self.agent.fb_learner.replay_buffer)
                    })
            
            self.training_stats['total_steps'] += 1
        
        print(f"\nTraining complete!")
        print(f"Final buffer size: {len(self.agent.fb_learner.replay_buffer)}")
        print(f"Total sun hits during training: {self.training_stats['sun_hits']}")
        
        if losses:
            final_loss = np.mean(losses[-100:]) if len(losses) >= 100 else losses[-1]
            print(f"Final loss: {final_loss:.4f}")
        
        # Save training curves
        self.save_training_curves(losses, light_hit_rates)
    
    def save_training_curves(self, losses, hit_rates):
        """Save training curves"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Loss curve
        if losses:
            ax1.plot(losses, alpha=0.7)
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')
            ax1.grid(True, alpha=0.3)
        
        # Hit rate curve
        if hit_rates:
            ax2.plot(hit_rates, alpha=0.7, color='green')
            ax2.set_ylabel('Sun Hit Rate')
            ax2.set_xlabel('Training Step')
            ax2.set_title('Sun Hit Rate During Training')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        curves_path = self.output_dir / "training_curves.png"
        plt.savefig(curves_path, dpi=150)
        plt.close()
        
        print(f"Training curves saved: {curves_path}")
    
    def save_model(self):
        """Save trained model"""
        save_path = self.output_dir / "fb_your_scene_final.pth"
        
        # Save using agent's method
        self.agent.save(str(save_path))
        
        # Also save training stats
        stats_path = self.output_dir / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        print(f"\nModel saved: {save_path}")
        print(f"Stats saved: {stats_path}")
        
        return str(save_path)
    
    # In train_fb_custom_scene.py, modify the test_model method:

    def test_model(self, num_tests=100):
        """Test the trained model"""
        print("\nTesting trained model...")
        
        test_hits = 0
        
        for test_idx in range(num_tests):
            # Generate test scenario
            sphere = random.choice(self.scene)
            sphere_id = sphere.id
            sphere_material = sphere.material
            
            # Random point on sphere
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi)
            
            point = Vector(
                sphere.centre.x + sphere.radius * math.sin(phi) * math.cos(theta),
                sphere.centre.y + sphere.radius * math.sin(phi) * math.sin(theta),
                sphere.centre.z + sphere.radius * math.cos(phi)
            )
            
            normal = point.subtractVector(sphere.centre).normalise()
            
            # Create observation
            obs = self.create_full_observation(
                point, normal, Vector(0, 0, -1), 0,
                Colour(0, 0, 0), sphere_material, sphere_id
            )
            
            # Get action from trained agent
            action, info = self.agent.choose_direction_research(  # Changed from choose_direction
                obs, 
                scene_context="test", 
                exploration_phase="test"
            )
            
            # Convert action to direction (action is already a numpy array)
            theta_act = (action[0] + 1) * math.pi/4
            phi_act = action[1] * math.pi
            
            # Transform to world coordinates
            if abs(normal.z) > 0.9:
                tangent = Vector(1, 0, 0)
            else:
                tangent = Vector(0, 0, 1).crossProduct(normal)
            tangent = tangent.normalise()
            bitangent = normal.crossProduct(tangent).normalise()
            
            local_dir = Vector(
                math.sin(theta_act) * math.cos(phi_act),
                math.sin(theta_act) * math.sin(phi_act),
                math.cos(theta_act)
            )
            
            world_dir = Vector(
                local_dir.x * tangent.x + local_dir.y * bitangent.x + local_dir.z * normal.x,
                local_dir.x * tangent.y + local_dir.y * bitangent.y + local_dir.z * normal.y,
                local_dir.x * tangent.z + local_dir.y * bitangent.z + local_dir.z * normal.z
            ).normalise()
            
            # Test if this direction hits sun
            test_ray = Ray(point.addVector(normal.scaleByLength(0.001)), world_dir)
            
            for test_sphere in self.scene:
                if test_sphere.id == 7:  # Sun
                    intersect = test_ray.sphereDiscriminant(test_sphere)
                    if intersect and intersect.intersects:
                        test_hits += 1
                        break
        
        hit_rate = test_hits / num_tests
        print(f"Test results: {test_hits}/{num_tests} sun hits ({hit_rate*100:.1f}%)")
        
        return hit_rate

def main():
    """Main training function"""
    print("="*80)
    print("PROPER FB TRAINING FOR YOUR CUSTOM SCENE")
    print("="*80)
    print("Training goal: Learn to find the small sun (position: -0.6, 0.2, 6)")
    print("Observation dimensions: 22")
    print("="*80)
    
    # Create trainer
    trainer = YourSceneFBTrainer()
    
    # Phase 1: Collect initial data
    trainer.collect_initial_data(num_episodes=200)  # Start with fewer episodes
    
    # Phase 2: Train
    trainer.train(num_steps=2000)  # Train for fewer steps initially
    
    # Phase 3: Test
    hit_rate = trainer.test_model(num_tests=100)
    
    # Phase 4: Save model
    model_path = trainer.save_model()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    
    if hit_rate > 0.1:  # At least 10% hit rate
        print(f"✓ Model learned to find sun! ({hit_rate*100:.1f}% hit rate)")
    else:
        print(f"⚠ Model may need more training ({hit_rate*100:.1f}% hit rate)")
        print("Try increasing num_steps in train()")
    
    print(f"\nTo use this model in output6.py:")
    print(f"Update the model path to: {model_path}")
    
    # Also update output6.py to use the correct observation creation
    print(f"\nImportant: Make sure output6.py uses the same 22-dimension observations!")
    
    return model_path

if __name__ == "__main__":
    main()