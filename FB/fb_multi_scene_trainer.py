"""
Multi-Scene FB Trainer - Train FB on 100+ varied scenes
This trains FB to generalize across different lighting, materials, and geometries
"""

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import random
import math
import sys
from typing import List, Dict, Tuple, Any

# Import FB agent
from fb_ray_tracing import FBResearchAgent, FBConfig

# Import scene modules
from vector import Vector
from colour import Colour
from object import Sphere
from material import Material
from ray import Ray

# Import our complex scene
from complex_scene import create_complex_scene, create_camera_for_scene

class SceneGenerator:
    """Generate 100+ varied training scenes"""
    
    def __init__(self):
        self.scene_count = 0
        self.scene_templates = []
        
        # Initialize with basic scene types
        self._initialize_scene_templates()
    
    def _initialize_scene_templates(self):
        """Initialize different scene templates"""
        
        # 1. Complex scene (our main challenging scene)
        self.scene_templates.append({
            'name': 'complex_scene',
            'generator': self._generate_complex_scene,
            'difficulty': 'hard',
            'description': 'Main challenging scene with small lights'
        })
        
        # 2. Cornell box variations
        self.scene_templates.append({
            'name': 'cornell_box',
            'generator': self._generate_cornell_box,
            'difficulty': 'medium',
            'description': 'Classic Cornell box with varying lights'
        })
        
        # 3. Mirror maze
        self.scene_templates.append({
            'name': 'mirror_maze',
            'generator': self._generate_mirror_maze,
            'difficulty': 'hard',
            'description': 'Many mirrors creating complex light paths'
        })
        
        # 4. Glass gallery
        self.scene_templates.append({
            'name': 'glass_gallery',
            'generator': self._generate_glass_gallery,
            'difficulty': 'hard',
            'description': 'Many glass objects creating caustics'
        })
        
        # 5. Simple scene with challenging light
        self.scene_templates.append({
            'name': 'simple_challenging',
            'generator': self._generate_simple_challenging,
            'difficulty': 'easy',
            'description': 'Simple geometry with one difficult light'
        })
        
        # 6. Many small lights
        self.scene_templates.append({
            'name': 'many_lights',
            'generator': self._generate_many_lights,
            'difficulty': 'medium',
            'description': 'Many small lights to learn from'
        })
        
        # 7. Occluded lights
        self.scene_templates.append({
            'name': 'occluded_lights',
            'generator': self._generate_occluded_lights,
            'difficulty': 'hard',
            'description': 'Lights hidden behind objects'
        })
        
        print(f"Initialized {len(self.scene_templates)} scene templates")
    
    def _generate_complex_scene(self, variation=0):
        """Generate variations of our complex scene"""
        # Use our existing complex scene as base
        spheres = create_complex_scene()
        
        # Apply variations
        if variation > 0:
            # Vary light positions
            for sphere in spheres:
                if sphere.material.emitive:
                    # Move lights slightly
                    sphere.centre = Vector(
                        sphere.centre.x + random.uniform(-0.5, 0.5),
                        sphere.centre.y + random.uniform(-0.2, 0.2),
                        sphere.centre.z + random.uniform(-0.5, 0.5)
                    )
        
        return spheres
    
    def _generate_cornell_box(self, variation=0):
        """Generate Cornell box scene"""
        spheres = []
        
        # Materials
        matte_red = Material(reflective=0.1, transparent=0, emitive=0)
        matte_green = Material(reflective=0.1, transparent=0, emitive=0)
        matte_white = Material(reflective=0.1, transparent=0, emitive=0)
        emitive_white = Material(reflective=0, transparent=0, emitive=1)
        
        # Box walls (large spheres as planes)
        # Floor
        spheres.append(Sphere(
            id=1,
            centre=Vector(0, -100, 0),
            radius=99,
            material=matte_white,
            colour=Colour(220, 220, 220)
        ))
        
        # Ceiling
        spheres.append(Sphere(
            id=2,
            centre=Vector(0, 100, 0),
            radius=99,
            material=matte_white,
            colour=Colour(220, 220, 220)
        ))
        
        # Back wall
        spheres.append(Sphere(
            id=3,
            centre=Vector(0, 0, -100),
            radius=99,
            material=matte_white,
            colour=Colour(200, 200, 220)
        ))
        
        # Left wall (red)
        spheres.append(Sphere(
            id=4,
            centre=Vector(-100, 0, 0),
            radius=99,
            material=matte_red,
            colour=Colour(220, 150, 150)
        ))
        
        # Right wall (green)
        spheres.append(Sphere(
            id=5,
            centre=Vector(100, 0, 0),
            radius=99,
            material=matte_green,
            colour=Colour(150, 220, 150)
        ))
        
        # Ceiling light
        light_size = 0.8 + variation * 0.2
        spheres.append(Sphere(
            id=6,
            centre=Vector(0, 99, 0),
            radius=light_size,
            material=emitive_white,
            colour=Colour(255, 255, 240)
        ))
        
        # Objects in the box
        # Reflective sphere
        reflective = Material(reflective=0.7, transparent=0, emitive=0)
        spheres.append(Sphere(
            id=7,
            centre=Vector(-1.5, -0.8, 2),
            radius=0.7,
            material=reflective,
            colour=Colour(255, 255, 255)
        ))
        
        # Glass sphere
        glass = Material(reflective=0, transparent=0.9, emitive=0, refractive_index=1.5)
        spheres.append(Sphere(
            id=8,
            centre=Vector(1.5, -0.8, 3),
            radius=0.6,
            material=glass,
            colour=Colour(255, 255, 255)
        ))
        
        # Small difficult light (optional)
        if variation % 3 != 0:  # Add small light 2/3 of the time
            small_light = Material(reflective=0, transparent=0, emitive=1)
            spheres.append(Sphere(
                id=9,
                centre=Vector(0, -0.5, 5),
                radius=0.1 + variation * 0.05,
                material=small_light,
                colour=Colour(255, 240, 200)
            ))
        
        return spheres
    
    def _generate_mirror_maze(self, variation=0):
        """Generate scene with many mirrors"""
        spheres = []
        
        # Materials
        mirror = Material(reflective=0.9, transparent=0, emitive=0)
        matte = Material(reflective=0.1, transparent=0, emitive=0)
        emitive_mat = Material(reflective=0, transparent=0, emitive=1)
        
        # Floor and ceiling
        spheres.append(Sphere(
            id=1, centre=Vector(0, -100, 0), radius=99,
            material=matte, colour=Colour(200, 200, 200)
        ))
        spheres.append(Sphere(
            id=2, centre=Vector(0, 100, 0), radius=99,
            material=matte, colour=Colour(200, 200, 200)
        ))
        
        # Main light
        spheres.append(Sphere(
            id=3, centre=Vector(0, 99, 3), radius=1.2,
            material=emitive_mat, colour=Colour(255, 255, 240)
        ))
        
        # Create mirror maze
        num_mirrors = 8 + variation
        maze_radius = 4.0
        
        for i in range(num_mirrors):
            angle = i * (360 / num_mirrors) * math.pi / 180
            x = maze_radius * math.cos(angle)
            z = maze_radius * math.sin(angle) + 3
            
            # Vary mirror heights
            height = -0.5 + (i % 3) * 0.3
            
            spheres.append(Sphere(
                id=10 + i,
                centre=Vector(x, height, z),
                radius=0.8,
                material=mirror,
                colour=Colour(255, 255, 255)
            ))
        
        # Small light in center (hard to reach)
        small_light_radius = 0.08 + variation * 0.02
        spheres.append(Sphere(
            id=100,
            centre=Vector(0, 0, 3),
            radius=small_light_radius,
            material=emitive_mat,
            colour=Colour(255, 220, 180)
        ))
        
        return spheres
    
    def _generate_glass_gallery(self, variation=0):
        """Generate scene with many glass objects"""
        spheres = []
        
        # Materials
        glass = Material(reflective=0.1, transparent=0.9, emitive=0, refractive_index=1.5)
        matte = Material(reflective=0.1, transparent=0, emitive=0)
        emitive_mat = Material(reflective=0, transparent=0, emitive=1)
        
        # Room
        spheres.append(Sphere(
            id=1, centre=Vector(0, -100, 0), radius=99,
            material=matte, colour=Colour(220, 220, 220)
        ))
        spheres.append(Sphere(
            id=2, centre=Vector(0, 100, 0), radius=99,
            material=matte, colour=Colour(220, 220, 220)
        ))
        
        # Lights
        spheres.append(Sphere(
            id=3, centre=Vector(-2, 99, 2), radius=1.0,
            material=emitive_mat, colour=Colour(255, 255, 240)
        ))
        spheres.append(Sphere(
            id=4, centre=Vector(2, 99, 2), radius=1.0,
            material=emitive_mat, colour=Colour(240, 240, 255)
        ))
        
        # Glass objects
        num_glass = 10 + variation
        gallery_radius = 3.5
        
        for i in range(num_glass):
            angle = i * (360 / num_glass) * math.pi / 180
            x = gallery_radius * math.cos(angle)
            z = gallery_radius * math.sin(angle) + 4
            
            # Vary sizes and positions
            radius = 0.2 + 0.15 * (i % 4)
            height = -0.7 + 0.4 * (i % 3)
            
            # Vary glass color slightly
            color_val = 200 + (i % 3) * 20
            color = Colour(color_val, color_val, 255)
            
            spheres.append(Sphere(
                id=20 + i,
                centre=Vector(x, height, z),
                radius=radius,
                material=glass,
                colour=color
            ))
        
        # Small light for caustics
        spheres.append(Sphere(
            id=100,
            centre=Vector(0, 1.5, 4),
            radius=0.12,
            material=emitive_mat,
            colour=Colour(255, 255, 220)
        ))
        
        return spheres
    
    def _generate_simple_challenging(self, variation=0):
        """Simple scene with one challenging light"""
        spheres = []
        
        # Materials
        matte = Material(reflective=0.1, transparent=0, emitive=0)
        emitive_mat = Material(reflective=0, transparent=0, emitive=1)
        
        # Simple room
        spheres.append(Sphere(
            id=1, centre=Vector(0, -100, 0), radius=99,
            material=matte, colour=Colour(220, 220, 220)
        ))
        
        # Main light (easy)
        spheres.append(Sphere(
            id=2, centre=Vector(0, 5, 3), radius=1.5,
            material=emitive_mat, colour=Colour(255, 255, 240)
        ))
        
        # A few objects
        for i in range(3):
            spheres.append(Sphere(
                id=10 + i,
                centre=Vector(-2 + i * 2, -0.8, 4),
                radius=0.5,
                material=matte,
                colour=Colour(180 + i*20, 200, 220 - i*20)
            ))
        
        # ONE challenging small light
        light_radius = 0.1 + variation * 0.02
        # Make it partially occluded by positioning
        occluder_x = 3.0 if variation % 2 == 0 else -3.0
        
        # Occluding object
        spheres.append(Sphere(
            id=20,
            centre=Vector(occluder_x, 0, 6),
            radius=0.7,
            material=matte,
            colour=Colour(200, 180, 180)
        ))
        
        # The challenging light (behind/next to occluder)
        light_x = occluder_x * 0.7
        light_z = 6 + (0.5 if variation % 2 == 0 else -0.5)
        
        spheres.append(Sphere(
            id=30,
            centre=Vector(light_x, 0.5, light_z),
            radius=light_radius,
            material=emitive_mat,
            colour=Colour(255, 230, 200)
        ))
        
        return spheres
    
    def _generate_many_lights(self, variation=0):
        """Scene with many small lights to learn from"""
        spheres = []
        
        # Materials
        matte = Material(reflective=0.1, transparent=0, emitive=0)
        emitive_mat = Material(reflective=0, transparent=0, emitive=1)
        
        # Room
        spheres.append(Sphere(
            id=1, centre=Vector(0, -100, 0), radius=99,
            material=matte, colour=Colour(200, 200, 200)
        ))
        
        # Many small lights
        num_lights = 15 + variation * 2
        room_radius = 5.0
        
        for i in range(num_lights):
            # Spherical distribution
            phi = random.uniform(0, math.pi)
            theta = random.uniform(0, 2 * math.pi)
            
            x = room_radius * math.sin(phi) * math.cos(theta)
            y = room_radius * math.cos(phi) - 2.0  # Center vertically
            z = room_radius * math.sin(phi) * math.sin(theta) + 5
            
            # Random small radius
            radius = random.uniform(0.08, 0.25)
            
            # Random color
            r = random.randint(200, 255)
            g = random.randint(200, 255)
            b = random.randint(200, 255)
            
            spheres.append(Sphere(
                id=10 + i,
                centre=Vector(x, y, z),
                radius=radius,
                material=emitive_mat,
                colour=Colour(r, g, b)
            ))
        
        # Some objects to occlude lights
        for i in range(5):
            spheres.append(Sphere(
                id=100 + i,
                centre=Vector(
                    random.uniform(-3, 3),
                    random.uniform(-1, 1),
                    random.uniform(3, 7)
                ),
                radius=random.uniform(0.3, 0.7),
                material=matte,
                colour=Colour(
                    random.randint(150, 220),
                    random.randint(150, 220),
                    random.randint(150, 220)
                )
            ))
        
        return spheres
    
    def _generate_occluded_lights(self, variation=0):
        """Lights hidden behind various objects"""
        spheres = []
        
        # Materials
        matte = Material(reflective=0.1, transparent=0, emitive=0)
        emitive_mat = Material(reflective=0, transparent=0, emitive=1)
        
        # Room
        spheres.append(Sphere(
            id=1, centre=Vector(0, -100, 0), radius=99,
            material=matte, colour=Colour(210, 210, 210)
        ))
        
        # Main light (not occluded)
        spheres.append(Sphere(
            id=2, centre=Vector(0, 4, 3), radius=1.2,
            material=emitive_mat, colour=Colour(255, 255, 240)
        ))
        
        # Create occluded lights
        num_occluded_lights = 8 + variation
        
        for i in range(num_occluded_lights):
            # Light position
            angle = i * (360 / num_occluded_lights) * math.pi / 180
            distance = 3.5 + (i % 3) * 0.5
            x = distance * math.cos(angle)
            z = distance * math.sin(angle) + 4
            y = -0.5 + (i % 2) * 0.3
            
            # Light
            light_radius = 0.1 + (i % 4) * 0.03
            spheres.append(Sphere(
                id=10 + i,
                centre=Vector(x, y, z),
                radius=light_radius,
                material=emitive_mat,
                colour=Colour(255, 240 - i*10, 220)
            ))
            
            # Occluding object in front of light
            # Position between camera and light
            occlusion_distance = random.uniform(0.3, 0.8)
            occluder_x = x * occlusion_distance
            occluder_z = (z - 4) * occlusion_distance + 4  # Keep same pattern
            occluder_y = y + random.uniform(-0.1, 0.2)
            
            spheres.append(Sphere(
                id=100 + i,
                centre=Vector(occluder_x, occluder_y, occluder_z),
                radius=random.uniform(0.3, 0.6),
                material=matte,
                colour=Colour(180, 190, 200)
            ))
        
        return spheres
    
    def generate_scene(self, scene_idx: int) -> Tuple[List[Sphere], str]:
        """Generate a scene for training"""
        # Cycle through templates
        template_idx = scene_idx % len(self.scene_templates)
        template = self.scene_templates[template_idx]
        
        # Generate variation based on scene_idx
        variation = scene_idx // len(self.scene_templates)
        
        # Generate scene
        spheres = template['generator'](variation)
        
        # Add scene metadata
        scene_name = f"{template['name']}_v{variation}"
        
        return spheres, scene_name
    
    def generate_batch(self, num_scenes: int) -> List[Tuple[List[Sphere], str]]:
        """Generate a batch of scenes"""
        scenes = []
        
        for i in range(num_scenes):
            spheres, name = self.generate_scene(i)
            scenes.append((spheres, name))
            self.scene_count += 1
        
        return scenes

class MultiSceneFBTrainer:
    """Train FB agent across multiple scenes"""
    
    def __init__(self, num_training_scenes=100):
        self.num_training_scenes = num_training_scenes
        
        # FB Configuration - tuned for multi-scene learning
        self.config = FBConfig(
            z_dim=64,
            f_hidden_dim=512,
            b_hidden_dim=256,
            num_forward_heads=3,
            num_layers=2,
            learning_rate=2e-4,
            batch_size=256,
            buffer_capacity=200000,  # Larger buffer for multiple scenes
            fb_weight=1.0,
            contrastive_weight=0.6,  # Higher for better generalization
            predictive_weight=0.4,
            max_bounces=6
        )
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"./fb_multi_scene_training_{timestamp}")
        self.output_dir.mkdir(exist_ok=True)
        
        # Scene generator
        self.scene_generator = SceneGenerator()
        
        # Create agent
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent = FBResearchAgent(self.config, device=self.device)
        
        # Training statistics
        self.stats = {
            'scenes_trained': [],
            'losses': [],
            'light_hit_rates': [],
            'scene_performance': {},
            'generalization_scores': []
        }
        
        print(f"\nMulti-Scene FB Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Output: {self.output_dir}")
        print(f"  Target scenes: {num_training_scenes}")
        print(f"  Scene templates: {len(self.scene_generator.scene_templates)}")
        print(f"  FB observation dimensions: {self.agent.obs_dim}")
    
    def create_training_observation(self, intersection_point, normal, ray_dir, 
                                   bounce_count, accumulated_color, material, sphere_id,
                                   scene_context=None):
        """Create observation for training"""
        # Default material if not provided
        if material is None:
            material = Material(reflective=False)
        
        # Default color if not provided
        if accumulated_color is None:
            accumulated_color = Colour(0, 0, 0)
        
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
        
        # Create observation (22 dimensions as expected by FB agent)
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
            # Scene context encoding (3) - encode scene type
            float(hash(scene_context) % 1000) / 1000.0 if scene_context else 0.0,
            0.0,  # reserved
            0.0   # reserved
        ], dtype=np.float32)
        
        # Ensure it's exactly 22 dimensions
        assert len(obs) == 22, f"Observation has {len(obs)} dimensions, expected 22"
        
        return obs
    
    def generate_training_experience(self, spheres, scene_name, num_episodes=50):
        """Generate training experience from a scene"""
        print(f"  Generating {num_episodes} episodes for {scene_name}...")
        
        episodes_generated = 0
        light_hits = 0
        
        # Find all lights in the scene
        light_spheres = [s for s in spheres if s.material.emitive]
        if not light_spheres:
            print(f"    Warning: No lights in scene {scene_name}")
            return 0, 0
        
        # Also find non-light spheres for starting points
        non_light_spheres = [s for s in spheres if not s.material.emitive]
        if not non_light_spheres:
            non_light_spheres = spheres  # Fall back to all spheres
        
        for episode in range(num_episodes):
            # Choose random starting sphere
            sphere = random.choice(non_light_spheres)
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
            
            # Surface normal
            normal = point_on_sphere.subtractVector(sphere.centre).normalise()
            
            # Random incoming ray direction
            incoming_theta = random.uniform(0, math.pi/2)
            incoming_phi = random.uniform(0, 2*math.pi)
            
            # Create local coordinate system
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
            
            # Choose a target light (for supervised learning)
            target_light = random.choice(light_spheres)
            
            # Optimal direction to target light
            to_light = target_light.centre.subtractVector(point_on_sphere)
            
            # Adjust direction based on normal (can't go through surface)
            if to_light.dotProduct(normal) < 0:
                # Light is behind surface, reflect
                optimal_dir = to_light.reflectInVector(normal)
            else:
                optimal_dir = to_light
            
            optimal_dir = optimal_dir.normalise()
            
            # Create observation
            bounce_count = random.randint(0, 2)
            accumulated_color = Colour(
                random.randint(0, 100),
                random.randint(0, 100),
                random.randint(0, 100)
            )
            
            obs = self.create_training_observation(
                point_on_sphere, normal, ray_dir, bounce_count,
                accumulated_color, sphere_material, sphere_id, scene_name
            )
            
            # Convert optimal direction to action
            # Transform to local coordinates
            local_optimal = Vector(
                optimal_dir.x * tangent.x + optimal_dir.y * tangent.y + optimal_dir.z * tangent.z,
                optimal_dir.x * bitangent.x + optimal_dir.y * bitangent.y + optimal_dir.z * bitangent.z,
                optimal_dir.x * normal.x + optimal_dir.y * normal.y + optimal_dir.z * normal.z
            )
            
            # Convert to spherical coordinates
            local_optimal = local_optimal.normalise()
            optimal_theta = math.acos(max(-1, min(1, local_optimal.z)))
            optimal_phi = math.atan2(local_optimal.y, local_optimal.x)
            
            # Normalize to [-1, 1]
            action = np.array([
                (optimal_theta / (math.pi/2)) * 2 - 1,  # theta ∈ [0, π/2] -> [-1, 1]
                optimal_phi / math.pi  # phi ∈ [-π, π] -> [-1, 1]
            ], dtype=np.float32)
            
            # Check if this action would hit the light
            test_ray = Ray(point_on_sphere.addVector(normal.scaleByLength(0.001)), optimal_dir)
            hit_light = False
            
            for test_sphere in spheres:
                if test_sphere.material.emitive:
                    intersect = test_ray.sphereDiscriminant(test_sphere)
                    if intersect and intersect.intersects:
                        hit_light = True
                        light_hits += 1
                        break
            
            # Create next observation
            if hit_light:
                # Create light hit observation
                light_material = Material(emitive=True)
                next_obs = self.create_training_observation(
                    point_on_sphere, normal, optimal_dir, bounce_count + 1,
                    Colour(255, 255, 200),  # Bright color
                    light_material, target_light.id, scene_name
                )
                reward = 10.0
            else:
                # Create regular next observation
                next_obs = self.create_training_observation(
                    point_on_sphere, normal, optimal_dir, bounce_count + 1,
                    accumulated_color, sphere_material, sphere_id, scene_name
                )
                reward = 0.1
            
            # Record for training
            self.agent.record_success(obs, action, next_obs, reward, hit_light)
            
            episodes_generated += 1
            
            # Add some random exploration
            if episode % 5 == 0:
                self._add_random_exploration(scene_name)
        
        hit_rate = light_hits / max(1, episodes_generated)
        return episodes_generated, hit_rate
    
    def _add_random_exploration(self, scene_name):
        """Add random exploration data"""
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
            reflective=random.random() > 0.7,
            transparent=random.random() > 0.9,
            emitive=False
        )
        
        # Random accumulated color
        color = Colour(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        
        # Create observation
        obs = self.create_training_observation(
            point, normal, ray_dir, random.randint(0, 3),
            color, material, random.randint(1, 50), scene_name
        )
        
        # Random action
        action = np.random.uniform(-1, 1, 2)
        
        # Random next observation
        next_point = Vector(
            point.x + random.uniform(-0.2, 0.2),
            point.y + random.uniform(-0.2, 0.2),
            point.z + random.uniform(-0.2, 0.2)
        )
        
        next_obs = self.create_training_observation(
            next_point, normal, ray_dir, random.randint(1, 4),
            color, material, random.randint(1, 50), scene_name
        )
        
        # Small random reward
        reward = random.uniform(-0.1, 0.2)
        
        self.agent.record_success(obs, action, next_obs, reward, False)
    
    def train_on_scenes(self, scenes, training_steps_per_scene=100):
        """Train on a batch of scenes"""
        total_scenes = len(scenes)
        
        print(f"\nTraining on {total_scenes} scenes...")
        print(f"Training steps per scene: {training_steps_per_scene}")
        
        scene_performances = []
        
        for scene_idx, (spheres, scene_name) in enumerate(scenes):
            print(f"\n[{scene_idx+1}/{total_scenes}] Scene: {scene_name}")
            print(f"  Objects: {len(spheres)}")
            print(f"  Lights: {len([s for s in spheres if s.material.emitive])}")
            
            # Generate initial experience
            episodes, hit_rate = self.generate_training_experience(spheres, scene_name, num_episodes=30)
            print(f"  Generated {episodes} episodes, initial hit rate: {hit_rate*100:.1f}%")
            
            # Train on this scene's experience
            scene_losses = []
            scene_hit_rates = []
            
            pbar = tqdm(range(training_steps_per_scene), desc=f"Training {scene_name[:20]}...")
            
            for step in pbar:
                # Generate more experience occasionally
                if step % 20 == 0:
                    more_episodes, current_hit_rate = self.generate_training_experience(
                        spheres, scene_name, num_episodes=5
                    )
                    scene_hit_rates.append(current_hit_rate)
                
                # Train if we have enough data
                if len(self.agent.fb_learner.replay_buffer) >= self.config.batch_size:
                    stats = self.agent.fb_learner.train_step()
                    loss = stats.get('total_loss', 0)
                    scene_losses.append(loss)
                    
                    # Update progress
                    if step % 50 == 0:
                        avg_loss = np.mean(scene_losses[-50:]) if len(scene_losses) >= 50 else loss
                        avg_hit = np.mean(scene_hit_rates[-10:]) if scene_hit_rates else 0
                        pbar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'hit_rate': f'{avg_hit*100:.1f}%',
                            'buffer': len(self.agent.fb_learner.replay_buffer)
                        })
            
            # Record scene performance
            final_hit_rate = np.mean(scene_hit_rates[-10:]) if scene_hit_rates else 0
            final_loss = np.mean(scene_losses[-50:]) if scene_losses else 0
            
            scene_perf = {
                'scene_name': scene_name,
                'objects': len(spheres),
                'lights': len([s for s in spheres if s.material.emitive]),
                'final_loss': float(final_loss),
                'final_hit_rate': float(final_hit_rate),
                'training_steps': training_steps_per_scene
            }
            
            scene_performances.append(scene_perf)
            
            # Update overall stats
            self.stats['scenes_trained'].append(scene_name)
            if scene_losses:
                self.stats['losses'].extend(scene_losses)
            if scene_hit_rates:
                self.stats['light_hit_rates'].extend(scene_hit_rates)
            
            self.stats['scene_performance'][scene_name] = scene_perf
            
            print(f"  Scene complete - Loss: {final_loss:.4f}, Hit rate: {final_hit_rate*100:.1f}%")
        
        return scene_performances
    
    def run_training(self, num_scenes=100, scenes_per_batch=20, training_steps_per_scene=150):
        """Run full multi-scene training"""
        print("\n" + "="*80)
        print("MULTI-SCENE FB TRAINING")
        print("="*80)
        print(f"Total target scenes: {num_scenes}")
        print(f"Scenes per batch: {scenes_per_batch}")
        print(f"Training steps per scene: {training_steps_per_scene}")
        print("="*80)
        
        total_batches = (num_scenes + scenes_per_batch - 1) // scenes_per_batch
        
        all_performances = []
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * scenes_per_batch
            batch_end = min(batch_start + scenes_per_batch, num_scenes)
            batch_size = batch_end - batch_start
            
            print(f"\n\n{'='*60}")
            print(f"BATCH {batch_idx+1}/{total_batches} (Scenes {batch_start+1}-{batch_end})")
            print(f"{'='*60}")
            
            # Generate scenes for this batch
            scenes = self.scene_generator.generate_batch(batch_size)
            
            # Train on this batch
            batch_performances = self.train_on_scenes(scenes, training_steps_per_scene)
            all_performances.extend(batch_performances)
            
            # Save checkpoint
            if (batch_idx + 1) % 2 == 0 or batch_idx == total_batches - 1:
                checkpoint_path = self.output_dir / f"checkpoint_batch_{batch_idx+1}.pth"
                self.agent.save(str(checkpoint_path))
                
                # Save performance report
                self._save_performance_report(all_performances, batch_idx + 1)
        
        # Final save
        final_path = self.output_dir / "fb_multi_scene_final.pth"
        self.agent.save(str(final_path))
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE!")
        print(f"{'='*80}")
        print(f"Final model saved: {final_path}")
        print(f"Total scenes trained: {len(all_performances)}")
        print(f"Total training steps: ~{len(all_performances) * training_steps_per_scene}")
        print(f"Final buffer size: {len(self.agent.fb_learner.replay_buffer)}")
        
        # Generate final report
        self._generate_final_report(all_performances)
        
        return final_path
    
    def _save_performance_report(self, performances, batch_num):
        """Save performance report"""
        report_path = self.output_dir / f"performance_batch_{batch_num}.json"
        
        report = {
            'batch': batch_num,
            'total_scenes': len(performances),
            'performances': performances,
            'overall_stats': {
                'avg_loss': np.mean([p['final_loss'] for p in performances if p['final_loss'] > 0]),
                'avg_hit_rate': np.mean([p['final_hit_rate'] for p in performances if p['final_hit_rate'] > 0]),
                'min_hit_rate': min([p['final_hit_rate'] for p in performances if p['final_hit_rate'] > 0]),
                'max_hit_rate': max([p['final_hit_rate'] for p in performances if p['final_hit_rate'] > 0])
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  Performance report saved: {report_path}")
    
    def _generate_final_report(self, performances):
        """Generate final training report"""
        # Calculate statistics
        successful_scenes = [p for p in performances if p['final_hit_rate'] > 0.01]
        
        stats = {
            'total_scenes_trained': len(performances),
            'successful_scenes': len(successful_scenes),
            'success_rate': len(successful_scenes) / len(performances) * 100,
            'avg_hit_rate': np.mean([p['final_hit_rate'] for p in performances if p['final_hit_rate'] > 0]),
            'avg_loss': np.mean([p['final_loss'] for p in performances if p['final_loss'] > 0]),
            'scene_types_trained': len(set([p['scene_name'].split('_')[0] for p in performances]))
        }
        
        # Group by scene type
        scene_types = {}
        for perf in performances:
            scene_type = perf['scene_name'].split('_')[0]
            if scene_type not in scene_types:
                scene_types[scene_type] = []
            scene_types[scene_type].append(perf['final_hit_rate'])
        
        type_stats = {}
        for scene_type, hit_rates in scene_types.items():
            type_stats[scene_type] = {
                'count': len(hit_rates),
                'avg_hit_rate': np.mean(hit_rates) * 100,
                'min_hit_rate': min(hit_rates) * 100,
                'max_hit_rate': max(hit_rates) * 100
            }
        
        # Create final report
        final_report = {
            'training_summary': {
                'config': self.config.to_dict(),
                'device': str(self.device),
                'total_training_time': 'N/A',  # Would need timing
                'final_buffer_size': len(self.agent.fb_learner.replay_buffer),
                'agent_stats': self.agent.get_research_metrics()
            },
            'performance_statistics': stats,
            'scene_type_performance': type_stats,
            'all_performances': performances
        }
        
        # Save report
        report_path = self.output_dir / "final_training_report.json"
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        # Print summary
        print(f"\n{'='*80}")
        print("FINAL TRAINING REPORT")
        print(f"{'='*80}")
        print(f"Total scenes trained: {stats['total_scenes_trained']}")
        print(f"Successful scenes (hit rate > 1%): {stats['successful_scenes']} ({stats['success_rate']:.1f}%)")
        print(f"Average hit rate: {stats['avg_hit_rate']*100:.2f}%")
        print(f"Scene types trained: {stats['scene_types_trained']}")
        
        print("\nPerformance by scene type:")
        for scene_type, type_stat in type_stats.items():
            print(f"  {scene_type:20s}: {type_stat['count']:3d} scenes, "
                  f"hit rate: {type_stat['avg_hit_rate']:5.1f}% "
                  f"(min: {type_stat['min_hit_rate']:.1f}%, "
                  f"max: {type_stat['max_hit_rate']:.1f}%)")
        
        print(f"\nFinal report saved: {report_path}")
        
        return final_report
    
    def test_on_complex_scene(self, num_tests=100):
        """Test the trained model on our complex scene"""
        print("\n" + "="*80)
        print("TESTING ON COMPLEX SCENE")
        print("="*80)
        
        # Load our complex scene
        spheres = create_complex_scene()
        camera_pos, _ = create_camera_for_scene()
        
        # Find lights
        all_lights = [s for s in spheres if s.material.emitive]
        small_lights = [s for s in all_lights if s.radius < 0.5]
        
        print(f"Test scene: Complex scene with {len(spheres)} objects")
        print(f"Total lights: {len(all_lights)}")
        print(f"Small challenging lights: {len(small_lights)}")
        
        test_hits = 0
        small_light_hits = 0
        
        for test_idx in tqdm(range(num_tests), desc="Testing"):
            # Choose random starting point
            non_light_spheres = [s for s in spheres if not s.material.emitive]
            if not non_light_spheres:
                non_light_spheres = spheres
            
            sphere = random.choice(non_light_spheres)
            
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
            obs = self.create_training_observation(
                point, normal, Vector(0, 0, -1), 0,
                Colour(0, 0, 0), sphere.material, sphere.id, "complex_test"
            )
            
            # Get action from trained agent
            action, info = self.agent.choose_direction_research(
                obs, scene_context="complex_test", exploration_phase="test"
            )
            
            # Convert action to direction
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
            
            # Test if this direction hits any light
            test_ray = Ray(point.addVector(normal.scaleByLength(0.001)), world_dir)
            
            hit_any_light = False
            hit_small_light = False
            
            for light_sphere in all_lights:
                intersect = test_ray.sphereDiscriminant(light_sphere)
                if intersect and intersect.intersects:
                    hit_any_light = True
                    if light_sphere in small_lights:
                        hit_small_light = True
                    break
            
            if hit_any_light:
                test_hits += 1
            if hit_small_light:
                small_light_hits += 1
        
        overall_hit_rate = test_hits / num_tests
        small_light_hit_rate = small_light_hits / num_tests
        
        print(f"\nTest Results:")
        print(f"  Overall light hit rate: {overall_hit_rate*100:.2f}%")
        print(f"  Small light hit rate: {small_light_hit_rate*100:.4f}%")
        print(f"  Small light hits: {small_light_hits}/{num_tests}")
        
        # Compare with random baseline
        # Calculate expected random hit probability for small lights
        if small_lights and camera_pos:
            avg_small_prob = 0
            for light in small_lights:
                distance = camera_pos.distanceFrom(light.centre)
                solid_angle = 2 * np.pi * (1 - math.sqrt(1 - (light.radius/distance)**2))
                avg_small_prob += solid_angle / (4 * math.pi)
            avg_small_prob /= len(small_lights)
            
            print(f"\nComparison with random sampling:")
            print(f"  Expected random hit rate for small lights: {avg_small_prob*100:.4f}%")
            print(f"  FB improvement factor: {small_light_hit_rate/avg_small_prob:.1f}x")
            
            if small_light_hit_rate > avg_small_prob:
                print(f"  ✓ FB learned to hit small lights better than random!")
            else:
                print(f"  ⚠ FB needs more training on small lights")
        
        return overall_hit_rate, small_light_hit_rate

def main():
    """Main training function"""
    print("="*80)
    print("MULTI-SCENE FB TRAINING FOR RAY TRACING")
    print("="*80)
    print("Goal: Train FB agent on 100+ varied scenes to generalize across:")
    print("1. Different lighting conditions")
    print("2. Various materials (mirror, glass, matte)")
    print("3. Complex geometries")
    print("4. Occluded and small light sources")
    print("="*80)
    
    # Configuration
    num_scenes = 100  # Target: train on 100 scenes
    scenes_per_batch = 20  # Process 20 scenes at a time
    training_steps_per_scene = 150  # Train 150 steps per scene
    
    # Create trainer
    trainer = MultiSceneFBTrainer(num_training_scenes=num_scenes)
    
    # Run training
    print(f"\nStarting training with configuration:")
    print(f"  Total scenes: {num_scenes}")
    print(f"  Scenes per batch: {scenes_per_batch}")
    print(f"  Training steps per scene: {training_steps_per_scene}")
    print(f"  Estimated total training steps: {num_scenes * training_steps_per_scene}")
    print(f"  Output directory: {trainer.output_dir}")
    
    # Optional: Ask for confirmation
    response = input("\nStart training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Run training
    try:
        model_path = trainer.run_training(
            num_scenes=num_scenes,
            scenes_per_batch=scenes_per_batch,
            training_steps_per_scene=training_steps_per_scene
        )
        
        # Test on complex scene
        print("\n" + "="*80)
        print("EVALUATION ON COMPLEX SCENE")
        print("="*80)
        
        overall_hit, small_hit = trainer.test_on_complex_scene(num_tests=200)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE - NEXT STEPS")
        print("="*80)
        print(f"1. Model saved: {model_path}")
        print(f"2. Test results: {small_hit*100:.4f}% hit rate on small lights")
        print(f"3. Use this model in output6.py for FB-accelerated rendering")
        print(f"4. Compare with traditional ray tracing from our baseline tests")
        
        if small_hit > 0.001:  # Better than 0.1%
            print(f"\n✓ SUCCESS: FB learned to hit small lights!")
            print(f"  Expected speedup on complex scene: {small_hit/0.0002:.1f}x")
        else:
            print(f"\n⚠ May need more training or adjustment")
        
        return model_path
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # For quick testing with fewer scenes
    import argparse
    
    parser = argparse.ArgumentParser(description="Train FB on multiple scenes")
    parser.add_argument("--quick", action="store_true", help="Quick test with 10 scenes")
    parser.add_argument("--test-only", action="store_true", help="Only test, don't train")
    
    args = parser.parse_args()
    
    if args.quick:
        # Quick test with fewer scenes
        print("Running quick test with 10 scenes...")
        
        trainer = MultiSceneFBTrainer(num_training_scenes=10)
        
        # Generate scenes
        scenes = trainer.scene_generator.generate_batch(10)
        
        # Quick training
        trainer.train_on_scenes(scenes, training_steps_per_scene=50)
        
        # Test
        trainer.test_on_complex_scene(num_tests=50)
        
    elif args.test_only:
        # Just test existing model
        print("Testing mode - looking for existing model...")
        
        # Look for latest model
        model_dir = Path(".")
        model_files = list(model_dir.glob("fb_multi_scene_*/fb_multi_scene_final.pth"))
        if not model_files:
            model_files = list(model_dir.glob("fb_your_scene_*/fb_your_scene_final.pth"))
        
        if model_files:
            model_files.sort(key=lambda x: x.parent.stat().st_mtime, reverse=True)
            latest_model = model_files[0]
            
            print(f"Found model: {latest_model}")
            
            # Load and test
            trainer = MultiSceneFBTrainer(num_training_scenes=1)
            trainer.agent.load(str(latest_model))
            trainer.test_on_complex_scene(num_tests=100)
        else:
            print("No trained model found. Run training first.")
    
    else:
        # Full training
        main()
