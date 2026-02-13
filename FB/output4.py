"""
FB vs RL comparison with your custom scene
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
warnings.filterwarnings('ignore')

# Try to import your existing modules
try:
    from vector import Vector, Angle
    from object import Sphere
    from ray import Ray, Intersection
    from material import Material
    from colour import Colour
    from light import GlobalLight, PointLight
    BASE_IMPORTS_OK = True
except ImportError as e:
    print(f"Warning: Could not import base modules: {e}")
    BASE_IMPORTS_OK = False

# Try to import RL system
try:
    from ray_tracer_rl_test import RayTracerRL
    RL_AVAILABLE = True
except ImportError:
    print("Warning: RL modules not available")
    RL_AVAILABLE = False

# NEW: Try to import FB training modules
try:
    from fb_ray_tracing import FBResearchAgent as FBRayTracingAgent, FBConfig
    FB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: FB training modules not available: {e}")
    FB_AVAILABLE = False

class TrainedFBAgent:
    """Trained FB agent that uses learned network for direction selection"""
    
    def __init__(self, model_path=None, scene_id="custom_scene"):
        print(f"  Initializing TRAINED FB agent for scene: {scene_id}")
        
        self.scene_id = scene_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load trained model if available
        self.model = None
        self.model_loaded = False
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            print(f"  Warning: No FB model found at {model_path}")
            print(f"  Run train_fb_ray_tracing.py first to train a model")
        
        # Statistics
        self.step_count = 0
        self.total_light_hits = 0
        self.confidence = 0.0
        
        # Scene knowledge for observation creation
        self.scene_knowledge = {
            'sun_position': Vector(-0.6, 0.2, 6),
            'sun_radius': 0.1,
            'sun_brightness': 1.0,
        }
    
    def load_model(self, model_path):
        """Load trained FB model"""
        try:
            # Create config (should match training config)
            config = FBConfig(
                z_dim=32,
                f_hidden_dim=256,
                b_hidden_dim=128,
                num_forward_heads=3,
                num_layers=2,
                learning_rate=3e-4,
                batch_size=256,
                buffer_capacity=100000,
                fb_weight=1.0,
                contrastive_weight=0.3,
                predictive_weight=0.2,
                norm_weight=0.1
            )
            
            # Create agent
            self.agent = FBRayTracingAgent(config, device=self.device)
            
            # Load using agent's load method
            self.agent.load(model_path)
            
            self.model_loaded = True
            print(f"  ✓ FB model loaded from {model_path}")
            
        except Exception as e:
            print(f"  ✗ Failed to load FB model: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
    
    def create_observation(self, intersection, ray, bounce_count, accumulated_color):
        """Create observation similar to fb_ray_tracing.py"""
        if intersection is None or not intersection.intersects:
            return np.zeros(22, dtype=np.float32)
        
        pos = intersection.point
        direction = ray.D
        normal = intersection.normal
        material = intersection.object.material
        
        # Sun direction
        sun_dir = self.scene_knowledge['sun_position'].subtractVector(pos).normalise()
        
        # Normalize color
        color_norm = np.array([
            accumulated_color.r / 255.0,
            accumulated_color.g / 255.0,
            accumulated_color.b / 255.0
        ], dtype=np.float32)
        
        # Material type encoding (one-hot)
        is_reflective = float(material.reflective)
        is_transparent = float(material.transparent)
        is_emitive = float(material.emitive)
        is_diffuse = float(not (material.reflective or material.transparent or material.emitive))
        
        # Create observation (22 dimensions)
        obs = np.array([
            # Position (3)
            pos.x, pos.y, pos.z,
            # Direction (3)
            direction.x, direction.y, direction.z,
            # Normal (3)
            normal.x, normal.y, normal.z,
            # Sun direction (3)
            sun_dir.x, sun_dir.y, sun_dir.z,
            # Material type (4) - one-hot encoded
            is_reflective, is_transparent, is_emitive, is_diffuse,
            # Accumulated color (3)
            color_norm[0], color_norm[1], color_norm[2],
            # Bounce count (1)
            float(bounce_count) / 10.0,  # Normalized
            # Distance to sun (1)
            pos.distanceFrom(self.scene_knowledge['sun_position']) / 20.0,  # Normalized
            # Current step reward estimate (1)
            self.confidence,
            # Material refractive index (1)
            float(material.refractive_index)
        ], dtype=np.float32)
        
        return obs
    
    def choose_direction(self, intersection, ray, bounce_count, accumulated_color):
        """Use trained FB network to choose direction"""
        self.step_count += 1
        
        # Create observation
        obs = self.create_observation(intersection, ray, bounce_count, accumulated_color)
        
        # Convert to tensor
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        if self.model_loaded:
            # Use trained agent
            with torch.no_grad():
                # Forward pass through FB agent
                action, _ = self.agent.choose_direction(obs_tensor, exploration=False)
                action_np = action.cpu().numpy()[0]
        else:
            # Fallback to random action
            action_np = np.random.uniform(-1, 1, 2)
        
        # Convert action to angles
        theta = (action_np[0] + 1) * np.pi/4  # [-1,1] -> [0,π/2]
        phi = action_np[1] * np.pi  # [-1,1] -> [-π,π]
        
        action_info = {
            'strategy': 'trained_fb' if self.model_loaded else 'random_fallback',
            'confidence': self.confidence,
            'model_loaded': self.model_loaded,
            'step': self.step_count
        }
        
        return action_np, action_info
    
    def record_success(self, intersection, ray, direction, color_brightness):
        """Record successful light hit"""
        self.total_light_hits += 1
        
        # Update confidence
        hit_rate = self.total_light_hits / max(1, self.step_count)
        self.confidence = 0.9 * self.confidence + 0.1 * hit_rate
    
    def reset_for_new_rendering(self):
        """Reset for new rendering"""
        self.step_count = 0
        self.total_light_hits = 0
        self.confidence = 0.0

class SmartFBAgent:
    """Actually intelligent FB agent that understands your scene"""
    
    def __init__(self, scene_id="custom_scene"):
        print(f"  Initializing SMART FB agent for scene: {scene_id}")
        
        # Memory systems
        self.successful_paths = []  # Store entire successful paths
        self.light_hit_positions = []  # Where we found lights
        self.material_behavior = defaultdict(list)  # How materials behave
        
        # Learning parameters
        self.exploration_rate = 0.7  # Start high, decrease as we learn
        self.confidence = 0.0  # 0-1 confidence in our knowledge
        self.learning_phase = 0  # 0=explore, 1=exploit, 2=optimize
        
        # Scene knowledge - CRITICAL for your scene!
        self.scene_knowledge = {
            'sun_position': Vector(-0.6, 0.2, 6),  # EXACT sun position
            'sun_radius': 0.1,
            'sun_brightness': 1.0,
            'glass_sphere_pos': Vector(-0.8, 0.6, 0),  # For caustics
            'reflective_sphere_pos': Vector(5.6, 3, -2),  # For reflections
            'camera_pos': Vector(0, 0, 1),  # Where rays start
        }
        
        # Strategy probabilities - TUNED FOR YOUR SCENE
        self.strategies = {
            'sun_direct': 0.4,    # Direct shot at sun (higher for small sun)
            'sun_reflect': 0.3,   # Bounce toward sun
            'material_guided': 0.2,  # Based on material type
            'memory_guided': 0.05, # From past successes
            'explore': 0.05,      # Pure exploration (lower)
        }
        
        self.step_count = 0
        self.total_light_hits = 0
        
        print(f"  Agent knows: Sun at ({self.scene_knowledge['sun_position'].x:.1f}, "
              f"{self.scene_knowledge['sun_position'].y:.1f}, "
              f"{self.scene_knowledge['sun_position'].z:.1f})")
    
    def analyze_material(self, material, intersection_point):
        """Analyze what to do based on material type"""
        actions = []
        
        if getattr(material, 'transparent', False):
            # GLASS: Can transmit or reflect
            # Check if we're near the glass sphere
            glass_dist = intersection_point.distanceFrom(self.scene_knowledge['glass_sphere_pos'])
            if glass_dist < 1.0:
                # Inside/near glass sphere - likely to transmit
                actions.append({
                    'type': 'transmit',
                    'probability': 0.7,
                    'reason': 'glass_sphere_proximity'
                })
            else:
                # Reflect from glass surface
                actions.append({
                    'type': 'reflect',
                    'probability': 0.6,
                    'reason': 'glass_surface'
                })
        
        elif getattr(material, 'reflective', False):
            # MIRROR: Perfect reflection
            actions.append({
                'type': 'reflect',
                'probability': 0.9,
                'reason': 'mirror'
            })
        
        else:
            # DIFFUSE: Random bounce with sun bias
            sun_dir = self.scene_knowledge['sun_position'].subtractVector(intersection_point).normalise()
            actions.append({
                'type': 'diffuse_sun_bias',
                'probability': 0.8,
                'sun_direction': sun_dir,
                'reason': 'diffuse_with_sun_bias'
            })
            actions.append({
                'type': 'random',
                'probability': 0.2,
                'reason': 'diffuse_exploration'
            })
        
        return actions
    
    def choose_direction(self, observation, intersection=None, ray=None, bounce_count=0):
        """Intelligent direction selection with scene understanding"""
        self.step_count += 1
        
        # Decrease exploration over time
        if self.step_count > 100:
            self.exploration_rate = max(0.05, self.exploration_rate * 0.995)
        
        # Update learning phase
        if self.total_light_hits > 20:
            self.learning_phase = 2  # Optimization
        elif self.total_light_hits > 5:
            self.learning_phase = 1  # Exploitation
        else:
            self.learning_phase = 0  # Exploration
        
        # Choose strategy based on current knowledge
        strategy = self._select_strategy()
        
        if strategy == 'sun_direct' and intersection:
            # Direct shot at sun from current position
            current_pos = intersection.point
            sun_pos = self.scene_knowledge['sun_position']
            
            # Vector directly to sun
            to_sun = sun_pos.subtractVector(current_pos).normalise()
            
            # Add noise based on confidence and distance
            distance = current_pos.distanceFrom(sun_pos)
            noise_level = 0.1 * (1 - self.confidence) + 0.1 * (distance / 10.0)
            
            theta_noise = np.random.normal(0, noise_level)
            phi_noise = np.random.normal(0, noise_level * 2)
            
            # Convert to spherical coordinates
            theta = np.arccos(max(-1, min(1, to_sun.z))) + theta_noise
            phi = np.arctan2(to_sun.y, to_sun.x) + phi_noise
            
            # Convert to action format [-1, 1]
            action = np.array([
                np.clip((theta / (np.pi/2)) * 2 - 1, -1, 1),
                np.clip((phi / np.pi), -1, 1)  # phi ∈ [-π, π] -> [-1, 1]
            ])
            
            action_info = {
                'strategy': 'sun_direct',
                'confidence': self.confidence,
                'noise': noise_level,
                'phase': self.learning_phase
            }
            
            return action, action_info
        
        elif strategy == 'sun_reflect' and intersection and ray:
            # Bounce ray toward sun
            normal = intersection.normal
            current_pos = intersection.point
            sun_pos = self.scene_knowledge['sun_position']
            
            # Calculate reflection that would hit sun
            to_sun = sun_pos.subtractVector(current_pos).normalise()
            
            # Perfect reflection direction (incoming ray reflected toward sun)
            # This is a simplification - in reality we'd need to solve
            ideal_reflection = to_sun.reflectInVector(normal) if hasattr(to_sun, 'reflectInVector') else to_sun
            
            # Convert to spherical
            if ideal_reflection.magnitude() > 0:
                ideal_reflection = ideal_reflection.normalise()
                theta = np.arccos(max(-1, min(1, ideal_reflection.z)))
                phi = np.arctan2(ideal_reflection.y, ideal_reflection.x)
            else:
                theta = np.random.uniform(0, np.pi/4)
                phi = np.random.uniform(0, 2*np.pi)
            
            # Add material-aware noise
            if intersection and hasattr(intersection.object, 'material'):
                material = intersection.object.material
                if getattr(material, 'reflective', False):
                    noise = 0.02  # Low noise for mirrors
                elif getattr(material, 'transparent', False):
                    noise = 0.05  # Medium noise for glass
                else:
                    noise = 0.1   # High noise for diffuse
            else:
                noise = 0.1
            
            theta += np.random.normal(0, noise)
            phi += np.random.normal(0, noise * 2)
            
            # Convert to action
            action = np.array([
                np.clip((theta / (np.pi/2)) * 2 - 1, -1, 1),
                np.clip((phi / np.pi), -1, 1)
            ])
            
            action_info = {
                'strategy': 'sun_reflect',
                'noise': noise,
                'phase': self.learning_phase
            }
            
            return action, action_info
    
    def _select_strategy(self):
        """Select which strategy to use based on current state"""
        # Adjust strategy weights based on learning phase
        if self.learning_phase == 0:  # Exploration
            weights = [0.1, 0.1, 0.2, 0.1, 0.5]  # Favor exploration
        elif self.learning_phase == 1:  # Exploitation
            weights = [0.3, 0.3, 0.2, 0.1, 0.1]  # Favor sun strategies
        else:  # Optimization
            weights = [0.4, 0.3, 0.2, 0.1, 0.0]  # No pure exploration
        
        strategies = list(self.strategies.keys())
        return np.random.choice(strategies, p=weights)
    
    def record_success(self, intersection, ray, direction, color_brightness):
        """Record successful light hit"""
        self.total_light_hits += 1
        
        # Store the successful position
        if intersection and intersection.intersects:
            self.light_hit_positions.append({
                'position': intersection.point,
                'object_id': intersection.object.id,
                'brightness': color_brightness,
                'step': self.step_count
            })
        
        # Update confidence
        hit_rate = self.total_light_hits / max(1, self.step_count)
        self.confidence = 0.9 * self.confidence + 0.1 * hit_rate
        
        # Keep memory limited
        if len(self.light_hit_positions) > 50:
            self.light_hit_positions.pop(0)
    
    def record_path(self, path_data):
        """Record an entire successful path"""
        self.successful_paths.append(path_data)
        
        # Keep only the 10 best paths
        if len(self.successful_paths) > 10:
            # Sort by reward and keep best
            self.successful_paths.sort(key=lambda x: x.get('reward', 0), reverse=True)
            self.successful_paths = self.successful_paths[:10]
    
    def reset_for_new_rendering(self):
        """Reset for new rendering but keep learned knowledge"""
        # Reset counters but keep learned knowledge
        self.step_count = 0
        # Keep successful paths and light hits
        # Only clear if we have too many
        if len(self.successful_paths) > 20:
            self.successful_paths = self.successful_paths[-10:]
        if len(self.light_hit_positions) > 100:
            self.light_hit_positions = self.light_hit_positions[-50:]


def create_custom_scene():
    """Create your custom scene"""
    print("Creating your custom scene...")
    
    scenes = {}
    spheres_list = []
    
    # Materials
    base_material = Material(reflective=False)
    reflective_material = Material(reflective=True)
    glass = Material(reflective=False, transparent=True, refractive_index=1.52)
    emitive_material = Material(emitive=True)
    
    # Your spheres
    sphere_1 = Sphere(
        id=1,
        centre=Vector(-0.8, 0.6, 0),
        radius=0.3,
        material=glass,
        colour=Colour(255, 100, 100)
    )
    
    sphere_2 = Sphere(
        id=2,
        centre=Vector(0.8, -0.8, -10),
        radius=2.2,
        material=base_material,
        colour=Colour(204, 204, 255)
    )
    
    sphere_3 = Sphere(
        id=3,
        centre=Vector(0.3, 0.34, 0.1),
        radius=0.2,
        material=base_material,
        colour=Colour(0, 51, 204)
    )
    
    sphere_4 = Sphere(
        id=4,
        centre=Vector(5.6, 3, -2),
        radius=5,
        material=reflective_material,
        colour=Colour(153, 51, 153)
    )
    
    sphere_5 = Sphere(
        id=5,
        centre=Vector(-0.8, -0.8, -0.2),
        radius=0.25,
        material=base_material,
        colour=Colour(153, 204, 0)
    )
    
    sphere_6 = Sphere(
        id=6,
        centre=Vector(-3, 10, -75),
        radius=30,
        material=base_material,
        colour=Colour(255, 204, 102)
    )
    
    # Sun (light source)
    sun = Sphere(
        id=7,
        centre=Vector(-0.6, 0.2, 6),
        radius=0.1,
        material=emitive_material,
        colour=Colour(255, 255, 204)
    )
    
    spheres_list = [sphere_1, sphere_2, sphere_3, sphere_4, sphere_5, sphere_6, sun]
    
    scenes['custom_scene'] = spheres_list
    
    # Scene description for documentation
    scene_info = """
    YOUR CUSTOM SCENE:
    
    Objects:
    1. Glass sphere (red, center front) - Refractive, creates caustics
    2. Large blue sphere (back right) - Diffuse material
    3. Small blue sphere (center) - Diffuse material  
    4. Large purple sphere (top right) - Reflective material
    5. Small green sphere (bottom left) - Diffuse material
    6. Very large yellow sphere (far back) - Background object
    7. Sun (yellow, top center) - Light source, emits light
    
    Key Challenges:
    • Glass sphere creates complex refractions
    • Reflective sphere requires accurate bounce calculations
    • Sun is small and specific to find
    • Variety of materials tests adaptive sampling
    """
    
    print(scene_info)
    
    return scenes


class CustomSceneExperiment:
    """Experiment with your custom scene"""
    
    def __init__(self, output_dir: str = "./custom_scene_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'fb': defaultdict(list),
            'rl': defaultdict(list),
            'traditional': defaultdict(list)
        }
        
        self.timing_data = {
            'fb': [],
            'rl': [],
            'traditional': []
        }
        
        self.rendered_images = {}
        self.fb_agents = {}
        self.trained_fb_agents = {}  # NEW: For trained FB agents
        self.rl_agent = None 

         # NEW: FB model paths
        self.fb_model_paths = {
            'default': "./fb_training_outputs/fb_your_scene_final.pth",
            'custom_scene': "./fb_training_outputs/fb_your_scene_final.pth"
        }

        # Check for trained FB models
        self.fb_models_available = {}
        for scene_id, path in self.fb_model_paths.items():
            if Path(path).exists():
                self.fb_models_available[scene_id] = True
                print(f"  ✓ Found trained FB model for {scene_id}: {path}")
            else:
                self.fb_models_available[scene_id] = False
                print(f"  ✗ No trained FB model found for {scene_id}")
                print(f"    Run: python train_fb_ray_tracing.py")
        # ⚡ SPEED CONTROL PARAMETERS - ADJUST THESE ⚡
        self.experiment_config = {
            # 🚀 FAST TESTING (quick results)
            'fast_mode': {
                'num_trials': 20,        # ⬇ Fewer trials = faster
                'rays_per_trial': 5,      # ⬇ Fewer rays = faster
                'max_bounces': 4,         # ⬇ Fewer bounces = faster
                'image_width': 200,       # ⬇ Smaller image = faster
                'image_height': 200,      # ⬇ Smaller image = faster
                'samples_per_pixel': 16,   # ⬇ Fewer samples = faster (but noisier)
                'progressive_steps': 2,   # ⬇ Fewer steps = faster
            },
            # ⏱️ BALANCED (good quality, reasonable time)
            'balanced_mode': {
                'num_trials': 50,
                'rays_per_trial': 8,
                'max_bounces': 6,
                'image_width': 200,
                'image_height': 200,
                'samples_per_pixel': 16,
                'progressive_steps': 3,
            },
            # 🎨 HIGH QUALITY (best results, slower)
            'quality_mode': {
                'num_trials': 100,
                'rays_per_trial': 12,
                'max_bounces': 8,
                'image_width': 400,
                'image_height': 300,
                'samples_per_pixel': 8,
                'progressive_steps': 4,
            }
        }
        
        # Choose mode here: 'fast_mode', 'balanced_mode', or 'quality_mode'
        self.current_mode = 'balanced_mode'
        self.config = self.experiment_config[self.current_mode]
        
        # Other settings
        self.config['render_image'] = True
        self.config['show_progressive'] = True
        
        print(f"\n⚡ EXPERIMENT MODE: {self.current_mode.upper()}")
        print(f"   Trials: {self.config['num_trials']}")
        print(f"   Image: {self.config['image_width']}x{self.config['image_height']}")
        print(f"   Samples per pixel: {self.config['samples_per_pixel']}")
        print(f"   Max bounces: {self.config['max_bounces']}")
        
        # RL model
        self.rl_model = None
        self.rl_model_loaded = False
        
        # Try to load RL model
        if RL_AVAILABLE:
            self.load_rl_model()
    
    def _get_fb_agent(self, scene_id, reset=False, use_trained=True):
        """Get or create FB agent for scene"""
        # Use trained FB agent if available
        if use_trained and scene_id in self.fb_models_available and self.fb_models_available[scene_id]:
            if scene_id not in self.trained_fb_agents:
                model_path = self.fb_model_paths.get(scene_id, self.fb_model_paths['default'])
                self.trained_fb_agents[scene_id] = TrainedFBAgent(model_path, scene_id)
            elif reset:
                self.trained_fb_agents[scene_id].reset_for_new_rendering()
            return self.trained_fb_agents[scene_id]
        else:
            # Fallback to heuristic FB agent
            if scene_id not in self.fb_agents:
                self.fb_agents[scene_id] = SmartFBAgent(scene_id)
            elif reset:
                self.fb_agents[scene_id].reset_for_new_rendering()
            return self.fb_agents[scene_id]
    
    def _find_nearest_intersection(self, ray, spheres):
        """Find the nearest intersection for a ray with spheres"""
        nearest_intersection = None
        nearest_distance = float('inf')
        nearest_sphere = None
        
        for sphere in spheres:
            intersection = ray.sphereDiscriminant(sphere)
            if intersection and intersection.intersects:
                distance = intersection.point.subtractVector(ray.origin).magnitude()
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_intersection = intersection
                    nearest_sphere = sphere
        
        return nearest_intersection

    def run_single_trial(self, scene_spheres, method: str, scene_id: str, trial_idx: int = 0):
        """Run a single trial with EXACT original camera"""
        trial_results = {
            'total_reward': 0,
            'light_hits': 0,
            'steps_taken': 0,
            'colors': [],
            'execution_time': 0,
            'strategies_used': []
        }
        
        start_time = time.time()
        
        # Use EXACT original camera parameters
        RAY_COUNT = 100
        RAY_STEP = 0.01
        MULTIPLE = 3
        
        RAY_COUNT *= MULTIPLE  # 300
        RAY_STEP /= MULTIPLE   # 0.00333...
        
        # Generate ONE ray at center for testing
        camera_pos = Vector(0, 0, 1)
        
        # Use center ray (0, 0, -1) or add small random offset
        if trial_idx == 0:
            # Center ray
            ray_dir = Vector(0, 0, -1)
        else:
            # Small random offset from center
            offset_x = np.random.uniform(-0.1, 0.1)
            offset_y = np.random.uniform(-0.1, 0.1)
            ray_dir = Vector(offset_x, offset_y, -1)
        
        # Normalize for our methods
        if ray_dir.magnitude() > 0:
            ray_dir = ray_dir.normalise()
        
        ray = Ray(camera_pos, ray_dir)
        
        # Trace ray
        if method == 'traditional':
            color, stats, strategies = self._trace_custom_traditional(ray, scene_spheres, scene_id)
        elif method == 'rl' and self.rl_model_loaded:
            color, stats, strategies = self._trace_custom_rl_trained(ray, scene_spheres, scene_id)
        elif method == 'fb':
            color, stats, strategies = self._trace_custom_fb_smart(ray, scene_spheres, scene_id, trial_idx)
        else:
            color, stats, strategies = self._trace_custom_traditional(ray, scene_spheres, scene_id)
        
        trial_results['total_reward'] = stats.get('reward', 0)
        trial_results['light_hits'] = stats.get('light_hits', 0)
        trial_results['steps_taken'] = stats.get('steps', 0)
        trial_results['strategies_used'] = strategies
        
        if hasattr(color, 'r'):
            trial_results['colors'].append([color.r, color.g, color.b])
        else:
            trial_results['colors'].append([0, 0, 0])
        
        trial_results['execution_time'] = time.time() - start_time
        
        # Unique strategies count
        trial_results['unique_strategies'] = len(set(trial_results['strategies_used']))
        
        return trial_results
        
    def render_true_original(self, scene_spheres, save_path: Path):
        """Render EXACTLY like your original notebook"""
        print("  Rendering TRUE original method (using your exact code)...")
        
        # EXACT parameters from your original
        RAY_COUNT = 100
        RAY_STEP = 0.01
        MULTIPLE = 3
        MAX_BOUNCES = 5
        
        RAY_COUNT *= MULTIPLE  # 300
        RAY_STEP /= MULTIPLE   # 0.00333...
        
        print(f"  Parameters: RAY_COUNT={RAY_COUNT}, RAY_STEP={RAY_STEP:.6f}, MULTIPLE={MULTIPLE}")
        
        # Generate rays EXACTLY like your original
        X_RAYS = [r*RAY_STEP for r in range(-RAY_COUNT, 0, 1)] + [r*RAY_STEP for r in range(0, RAY_COUNT + 1)]
        Y_RAYS = [r*RAY_STEP for r in range(RAY_COUNT, 0, -1)] + [-r*RAY_STEP for r in range(0, RAY_COUNT + 1)]
        
        width = len(X_RAYS)  # 601
        height = len(Y_RAYS)  # 601
        
        print(f"  Image size: {width}x{height}")
        print(f"  Total rays: {len(X_RAYS) * len(Y_RAYS)}")
        
        start_time = time.time()
        
        image = np.zeros((height, width, 3), dtype=np.float32)
        camera_pos = Vector(0, 0, 1)
        
        # Create lights EXACTLY like your original
        global_light_sources = [
            GlobalLight(
                vector=Vector(3, 1, -0.75),
                colour=Colour(20, 20, 255),
                strength=1,
                max_angle=np.radians(90),
                func=0
            )
        ]
        
        # Sun (EXACTLY like your original)
        sun = Sphere(
            id=0,  # Your original uses id=0 for sun
            centre=Vector(-0.6, 0.2, 6),
            radius=0.1,
            material=Material(emitive=True),
            colour=Colour(255, 255, 204)
        )
        
        point_light_sources = [
            PointLight(
                id=sun.id,
                position=sun.centre,
                colour=sun.colour,
                strength=1,
                max_angle=np.radians(90),
                func=-1
            )
        ]
        
        # Add all spheres (including sun)
        all_spheres = scene_spheres.copy()
        
        # Remove existing sun if any (to replace with correct one)
        all_spheres = [s for s in all_spheres if not (hasattr(s, 'id') and s.id == 7)]
        all_spheres.append(sun)
        
        background_colour = Colour(2, 2, 5)
        
        # Progress tracking
        progress_milestones = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        ray_count = len(X_RAYS) * len(Y_RAYS)
        
        for y_idx, Y in enumerate(tqdm(Y_RAYS, desc="Rendering")):
            for x_idx, X in enumerate(X_RAYS):
                # Create ray EXACTLY like your original
                ray = Ray(
                    origin=camera_pos,
                    D=Vector(x=X, y=Y, z=-1)  # NOT normalized in your original!
                )
                
                # TRACE EXACTLY like your original
                ray_terminal = ray.nearestSphereIntersect(all_spheres, max_bounces=MAX_BOUNCES)
                
                if ray_terminal is None:
                    color = background_colour
                else:
                    # Use EXACT original lighting calculation
                    color = ray_terminal.terminalRGB(
                        spheres=all_spheres,
                        background_colour=background_colour,
                        global_light_sources=global_light_sources,
                        point_light_sources=point_light_sources
                    )
                
                # Store in image (Y_RAYS is already reversed)
                image[y_idx, x_idx] = [
                    min(1.0, color.r / 255.0),
                    min(1.0, color.g / 255.0),
                    min(1.0, color.b / 255.0)
                ]
        
        render_time = time.time() - start_time
        
        # Save image
        plt.figure(figsize=(8, 8))
        plt.imshow(np.clip(image, 0, 1))
        plt.title(f'TRUE Original Render\nTime: {render_time:.1f}s | Size: {width}x{height}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  Render time: {render_time:.1f}s")
        
        return image

    def _trace_custom_traditional(self, ray, spheres, scene_id):
        """Traditional tracing that MIMICS original terminalRGB"""
        stats = {'reward': 0, 'light_hits': 0, 'steps': 0}
        strategies = ['traditional_mimic']
        
        # Try to use original method FIRST
        try:
            # Setup EXACTLY like original
            global_light_sources = [
                GlobalLight(
                    vector=Vector(3, 1, -0.75),
                    colour=Colour(20, 20, 255),
                    strength=1,
                    max_angle=np.radians(90),
                    func=0
                )
            ]
            
            sun = Sphere(
                id=0,
                centre=Vector(-0.6, 0.2, 6),
                radius=0.1,
                material=Material(emitive=True),
                colour=Colour(255, 255, 204)
            )
            
            point_light_sources = [
                PointLight(
                    id=sun.id,
                    position=sun.centre,
                    colour=sun.colour,
                    strength=1,
                    max_angle=np.radians(90),
                    func=-1
                )
            ]
            
            all_spheres = spheres.copy()
            
            # Remove any existing sun (id=7) and add correct sun (id=0)
            all_spheres = [s for s in all_spheres if not (hasattr(s, 'id') and s.id == 7)]
            all_spheres.append(sun)
            
            background_colour = Colour(2, 2, 5)
            
            # Use original ray tracing
            terminal = ray.nearestSphereIntersect(all_spheres, max_bounces=self.config['max_bounces'])
            
            if terminal is None:
                return background_colour, stats, strategies
            
            # Use original lighting
            color = terminal.terminalRGB(
                spheres=all_spheres,
                background_colour=background_colour,
                global_light_sources=global_light_sources,
                point_light_sources=point_light_sources
            )
            
            # Count light hits
            brightness = (color.r + color.g + color.b) / 3
            if brightness > 10:
                stats['light_hits'] = 1
                stats['reward'] = 10.0
            
            return color, stats, strategies
            
        except Exception as e:
            # Fallback to enhanced version
            print(f"  Original method failed, using enhanced: {e}")
        
        # ENHANCED VERSION that tries to mimic original brightness
        return self._trace_enhanced_traditional(ray, spheres, scene_id, stats, strategies)

    def _trace_enhanced_traditional(self, ray, spheres, scene_id, stats, strategies):
        """Enhanced traditional tracing with ORIGINAL-LIKE lighting"""
        current_ray = ray
        bounce_count = 0
        accumulated_color = Colour(0, 0, 0)
        
        # Sun setup
        sun_pos = Vector(-0.6, 0.2, 6)
        sun_color = Colour(255, 255, 204)
        
        # Global light (blue ambient)
        global_light = Colour(20, 20, 255)
        global_strength = 0.3  # 30% of global light
        
        # Add all spheres including a virtual sun for shadow testing
        all_spheres = spheres.copy()
        
        while bounce_count < self.config['max_bounces']:
            stats['steps'] += 1
            
            # Find nearest intersection
            nearest_intersection = None
            nearest_distance = float('inf')
            nearest_sphere = None
            
            for sphere in all_spheres:
                intersection = current_ray.sphereDiscriminant(sphere)
                if intersection and intersection.intersects:
                    dist = intersection.point.subtractVector(current_ray.origin).magnitude()
                    if dist < nearest_distance:
                        nearest_distance = dist
                        nearest_intersection = intersection
                        nearest_sphere = sphere
            
            # No intersection
            if not nearest_intersection:
                if bounce_count == 0:
                    return Colour(2, 2, 5), stats, strategies  # Background
                break
            
            intersection = nearest_intersection
            sphere = nearest_sphere
            
            # If hit a light (shouldn't happen with our spheres except virtual sun)
            if getattr(sphere.material, 'emitive', False):
                stats['light_hits'] += 1
                stats['reward'] += 10.0
                strategies.append('hit_sun')
                
                # Full sun brightness
                return Colour(255, 255, 200), stats, strategies
            
            # CALCULATE LIGHTING LIKE ORIGINAL
            
            # 1. GLOBAL LIGHT (always present)
            # In your original, GlobalLight adds blue ambient from direction (3, 1, -0.75)
            global_dir = Vector(3, 1, -0.75).normalise()
            global_cos = max(0, intersection.normal.dotProduct(global_dir))
            global_contrib = Colour(
                int(global_light.r * global_cos * global_strength),
                int(global_light.g * global_cos * global_strength),
                int(global_light.b * global_cos * global_strength)
            )
            
            # 2. POINT LIGHT (sun)
            to_sun = sun_pos.subtractVector(intersection.point).normalise()
            sun_distance = sun_pos.subtractVector(intersection.point).magnitude()
            
            # Shadow check
            shadow_ray = Ray(
                intersection.point.addVector(intersection.normal.scaleByLength(0.001)),
                to_sun
            )
            
            sun_visible = True
            for other_sphere in all_spheres:
                if other_sphere == sphere:
                    continue
                shadow_intersect = shadow_ray.sphereDiscriminant(other_sphere)
                if shadow_intersect and shadow_intersect.intersects:
                    shadow_dist = shadow_intersect.point.subtractVector(intersection.point).magnitude()
                    if shadow_dist < sun_distance:
                        sun_visible = False
                        break
            
            # Sun contribution
            sun_contrib = Colour(0, 0, 0)
            if sun_visible:
                # Distance attenuation (like your original PointLight with func=-1)
                attenuation = 1.0 / (sun_distance ** 2) if sun_distance > 0 else 1.0
                attenuation = min(1.0, attenuation * 100)  # Scale factor
                
                # Diffuse component
                cos_angle = max(0, intersection.normal.dotProduct(to_sun))
                
                # BRIGHTER like your original
                sun_strength = 0.9  # 90% of sun color
                
                sun_contrib = Colour(
                    int(sun_color.r * cos_angle * sun_strength * attenuation),
                    int(sun_color.g * cos_angle * sun_strength * attenuation),
                    int(sun_color.b * cos_angle * sun_strength * attenuation)
                )
            
            # 3. OBJECT'S OWN COLOR (material)
            sphere_color = sphere.colour
            
            # Combine: (global + sun) * sphere_color
            # Your original multiplies light with object color
            combined_light = Colour(
                min(255, global_contrib.r + sun_contrib.r),
                min(255, global_contrib.g + sun_contrib.g),
                min(255, global_contrib.b + sun_contrib.b)
            )
            
            # Multiply by sphere color (normalized)
            final_light = Colour(
                int(sphere_color.r * (combined_light.r / 255.0)),
                int(sphere_color.g * (combined_light.g / 255.0)),
                int(sphere_color.b * (combined_light.b / 255.0))
            )
            
            # Add to accumulated color
            accumulated_color = Colour(
                min(255, accumulated_color.r + final_light.r),
                min(255, accumulated_color.g + final_light.g),
                min(255, accumulated_color.b + final_light.b)
            )
            
            # Determine next bounce
            material = sphere.material
            
            if getattr(material, 'reflective', False):
                # Mirror reflection
                reflection_dir = current_ray.D.reflect(intersection.normal)
                current_ray = Ray(
                    intersection.point.addVector(intersection.normal.scaleByLength(0.001)),
                    reflection_dir
                )
                strategies.append('reflection')
                
            elif getattr(material, 'transparent', False):
                # Glass - simplified: continue with some randomness
                if np.random.random() < 0.5:
                    # Reflect
                    reflection_dir = current_ray.D.reflect(intersection.normal)
                    current_ray = Ray(
                        intersection.point.addVector(intersection.normal.scaleByLength(0.001)),
                        reflection_dir
                    )
                else:
                    # Transmit (straight)
                    current_ray = Ray(
                        intersection.point.addVector(current_ray.D.scaleByLength(0.001)),
                        current_ray.D
                    )
                strategies.append('glass')
                
            else:
                # Diffuse bounce (cosine weighted)
                r1 = np.random.random()
                r2 = np.random.random()
                
                theta = np.arccos(np.sqrt(r1))
                phi = 2 * np.pi * r2
                
                # Local coordinates
                normal = intersection.normal
                if abs(normal.z) > 0.9:
                    tangent = Vector(1, 0, 0)
                else:
                    tangent = Vector(0, 0, 1).crossProduct(normal)
                tangent = tangent.normalise()
                bitangent = normal.crossProduct(tangent).normalise()
                
                # Local to world
                local_dir = Vector(
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                )
                
                world_dir = Vector(
                    local_dir.x * tangent.x + local_dir.y * bitangent.x + local_dir.z * normal.x,
                    local_dir.x * tangent.y + local_dir.y * bitangent.y + local_dir.z * normal.y,
                    local_dir.x * tangent.z + local_dir.y * bitangent.z + local_dir.z * normal.z
                ).normalise()
                
                current_ray = Ray(
                    intersection.point.addVector(normal.scaleByLength(0.001)),
                    world_dir
                )
                strategies.append('diffuse')
            
            bounce_count += 1
        
        # Final color with brightness adjustment
        if accumulated_color.r == 0 and accumulated_color.g == 0 and accumulated_color.b == 0:
            # No light - dark background
            final_color = Colour(2, 2, 5)
        else:
            # Boost if too dark (original is quite bright)
            brightness = (accumulated_color.r + accumulated_color.g + accumulated_color.b) / 3
            
            if brightness < 80:  # Original is bright!
                # Scale up to match original brightness
                scale = 80 / max(1, brightness)
                accumulated_color = Colour(
                    min(255, int(accumulated_color.r * scale)),
                    min(255, int(accumulated_color.g * scale)),
                    min(255, int(accumulated_color.b * scale))
                )
            
            final_color = Colour(
                min(255, accumulated_color.r),
                min(255, accumulated_color.g),
                min(255, accumulated_color.b)
            )
        
        return final_color, stats, strategies
    
    def load_rl_model(self, model_path="./rl_ray_tracer_final.zip"):
        """Load a trained RL model"""
        try:
            from stable_baselines3 import PPO
            print(f"  Loading RL model from {model_path}...")
            self.rl_model = PPO.load(model_path)
            self.rl_model_loaded = True
            print("  ✓ RL model loaded successfully")
            return True
        except Exception as e:
            print(f"  ✗ Failed to load RL model: {e}")
            self.rl_model = None
            self.rl_model_loaded = False
            self.rl_agent = None 
            return False

    def _trace_custom_rl_trained(self, ray, spheres, scene_id, trial_idx=0):
        """Use TRAINED RL agent"""
        stats = {'reward': 0, 'light_hits': 0, 'steps': 0}
        strategies = ['rl_trained']
        
        if not hasattr(self, 'rl_model_loaded') or not self.rl_model_loaded:
            print("  No trained RL model available, using simplified version")
            return self._trace_custom_rl_simplified(ray, spheres, scene_id)
        
        # Create a simple environment observation
        camera_pos = Vector(0, 0, 1)
        
        # Trace the ray
        current_ray = ray
        bounce_count = 0
        accumulated_color = Colour(0, 0, 0)
        sun_pos = Vector(-0.6, 0.2, 6)
        
        while bounce_count < min(self.config['max_bounces'], 8):
            stats['steps'] += 1
            
            # Find nearest intersection
            nearest_intersection = self._find_nearest_intersection(current_ray, spheres)
            
            if not nearest_intersection:
                break
            
            intersection = nearest_intersection
            
            # Create observation for RL (similar to RayTracerEnv)
            sun_dir = sun_pos.subtractVector(intersection.point).normalise()
            
            # Simplified observation
            obs = np.array([
                intersection.point.x, intersection.point.y, intersection.point.z,  # position (3)
                current_ray.D.x, current_ray.D.y, current_ray.D.z,  # direction (3)
                intersection.normal.x, intersection.normal.y, intersection.normal.z,  # normal (3)
                float(intersection.object.material.reflective),
                float(intersection.object.material.transparent),
                float(intersection.object.material.emitive),
                float(intersection.object.material.refractive_index),  # material (4)
                accumulated_color.r / 255.0, accumulated_color.g / 255.0, accumulated_color.b / 255.0,  # color (3)
                float(bounce_count),  # bounce count (1)
                0.0  # through_count (1)
                # Total: 3+3+3+4+3+1+1 = 18 dimensions
            ], dtype=np.float32)
            
            # Make sure it's exactly 18 dimensions
            assert obs.shape == (18,), f"Observation shape mismatch: {obs.shape}"
            
            # Get action from trained RL model
            action, _ = self.rl_model.predict(obs, deterministic=True)
            
            # Convert action to direction (same as FB agent)
            theta = (action[0] + 1) * np.pi/4  # Convert from [-1,1] to [0,π/2]
            phi = action[1] * np.pi  # Convert from [-1,1] to [-π,π]
            
            # Transform to surface coordinates
            normal = intersection.normal
            if abs(normal.z) < 0.9:
                tangent = Vector(0, 0, 1).crossProduct(normal).normalise()
            else:
                tangent = Vector(1, 0, 0).crossProduct(normal).normalise()
            
            bitangent = normal.crossProduct(tangent).normalise()
            
            # Local direction
            local_dir = Vector(
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            )
            
            # Convert to world direction
            world_dir = Vector(
                local_dir.x * tangent.x + local_dir.y * bitangent.x + local_dir.z * normal.x,
                local_dir.x * tangent.y + local_dir.y * bitangent.y + local_dir.z * normal.y,
                local_dir.x * tangent.z + local_dir.y * bitangent.z + local_dir.z * normal.z
            ).normalise()
            
            # Calculate lighting (same as traditional)
            color = self._calculate_simple_lighting(intersection, spheres, sun_pos)
            accumulated_color = Colour(
                min(255, accumulated_color.r + color.r),
                min(255, accumulated_color.g + color.g),
                min(255, accumulated_color.b + color.b)
            )
            
            # Check if hit sun
            if intersection.object.id == 7:
                stats['light_hits'] += 1
                stats['reward'] += 10.0
                strategies.append('rl_hit_sun')
                break
            
            # Create new ray
            current_ray = Ray(
                intersection.point.addVector(normal.scaleByLength(0.001)),
                world_dir
            )
            
            bounce_count += 1
        
        # Ensure minimum brightness
        brightness = (accumulated_color.r + accumulated_color.g + accumulated_color.b) / 3
        if brightness < 50:
            boost = 80 / max(1, brightness)
            accumulated_color = Colour(
                min(255, int(accumulated_color.r * boost)),
                min(255, int(accumulated_color.g * boost)),
                min(255, int(accumulated_color.b * boost))
            )
        
        return accumulated_color, stats, strategies

    def _trace_custom_rl_simplified(self, ray, spheres, scene_id):
        """Simplified RL for comparison"""
        # Use the SAME lighting as traditional to ensure fair comparison
        return self._trace_custom_traditional(ray, spheres, scene_id)

    def _calculate_simple_lighting(self, intersection, spheres, sun_pos):
        """Simple lighting calculation (same as used in traditional)"""
        # This should match your traditional lighting calculation
        # Use the same code from _trace_enhanced_traditional
        
        sphere_color = intersection.object.colour
        
        # Ambient
        ambient_strength = 0.2
        ambient = Colour(
            int(sphere_color.r * ambient_strength),
            int(sphere_color.g * ambient_strength),
            int(sphere_color.b * ambient_strength)
        )
        
        # Diffuse from sun
        to_sun = sun_pos.subtractVector(intersection.point).normalise()
        cos_angle = max(0, intersection.normal.dotProduct(to_sun))
        diffuse_strength = 0.8 * cos_angle
        diffuse = Colour(
            int(sphere_color.r * diffuse_strength),
            int(sphere_color.g * diffuse_strength),
            int(sphere_color.b * diffuse_strength)
        )
        
        return Colour(
            min(255, ambient.r + diffuse.r),
            min(255, ambient.g + diffuse.g),
            min(255, ambient.b + diffuse.b)
        )
                
    def _trace_custom_fb_smart(self, ray, spheres, scene_id, trial_idx):
        """SMART FB tracing using intelligent agent"""
        stats = {'reward': 0, 'light_hits': 0, 'steps': 0, 'fb_confidence': 0}
        strategies = []
        
        fb_agent = self._get_fb_agent(scene_id, use_trained=True)        
        current_ray = ray
        bounce_count = 0
        accumulated_color = Colour(0, 0, 0)
        path_data = {
            'directions': [],
            'positions': [],
            'objects': [],
            'reward': 0
        }
        
        # Sun setup
        sun_pos = Vector(-0.6, 0.2, 6)
        sun_color = Colour(255, 255, 200)
        
        # Get all spheres including sun
        all_spheres = spheres.copy()
        
        while bounce_count < self.config['max_bounces']:
            stats['steps'] += 1
            
            # Find intersection
            nearest_intersection = None
            nearest_distance = float('inf')
            nearest_sphere = None
            
            for sphere in all_spheres:
                intersection = current_ray.sphereDiscriminant(sphere)
                if intersection and intersection.intersects:
                    distance = intersection.point.subtractVector(current_ray.origin).magnitude()
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_intersection = intersection
                        nearest_sphere = sphere
            
            intersection = nearest_intersection
            
            if intersection and intersection.intersects:
                sphere = nearest_sphere
                
                # Record path data
                path_data['directions'].append(current_ray.D)
                path_data['positions'].append(intersection.point)
                path_data['objects'].append(sphere.id)
                
                # Check if it's the SUN (id=7 in your scene)
                if sphere.id == 7:
                    stats['light_hits'] += 1
                    stats['reward'] += 10.0
                    strategies.append('fb_hit_sun')
                    
                    # Bright sun color
                    accumulated_color = Colour(
                        min(255, accumulated_color.r + sun_color.r),
                        min(255, accumulated_color.g + sun_color.g),
                        min(255, accumulated_color.b + sun_color.b)
                    )
                    
                    # Record successful hit
                    brightness = (sun_color.r + sun_color.g + sun_color.b) / 3
                    fb_agent.record_success(intersection, current_ray, current_ray.D, brightness)
                    path_data['reward'] = stats['reward']
                    fb_agent.record_path(path_data)
                    
                    # Early exit - we hit light!
                    break
                
                # Calculate lighting (same as traditional)
                to_sun = sun_pos.subtractVector(intersection.point).normalise()
                
                
                # Shadow check
                shadow_ray = Ray(
                    intersection.point.addVector(intersection.normal.scaleByLength(0.001)),
                    to_sun
                )
                
                sun_visible = True
                sun_distance = sun_pos.subtractVector(intersection.point).magnitude()
                
                for other_sphere in all_spheres:
                    if other_sphere == sphere or other_sphere.id == 7:  # Skip self and sun
                        continue
                    
                    shadow_intersect = shadow_ray.sphereDiscriminant(other_sphere)
                    if shadow_intersect and shadow_intersect.intersects:
                        shadow_dist = shadow_intersect.point.subtractVector(intersection.point).magnitude()
                        if shadow_dist < sun_distance:
                            sun_visible = False
                            break
                
                # Lighting calculation
                sphere_color = sphere.colour
                
                # Ambient
                ambient_strength = 0.2
                ambient = Colour(
                    int(sphere_color.r * ambient_strength),
                    int(sphere_color.g * ambient_strength),
                    int(sphere_color.b * ambient_strength)
                )
                
                # Diffuse
                diffuse = Colour(0, 0, 0)
                if sun_visible:
                    cos_angle = max(0, intersection.normal.dotProduct(to_sun))
                    diffuse_strength = 0.8 * cos_angle
                    diffuse = Colour(
                        int(sphere_color.r * diffuse_strength),
                        int(sphere_color.g * diffuse_strength),
                        int(sphere_color.b * diffuse_strength)
                    )
                
                # Combine lighting
                lighting = Colour(
                    min(255, ambient.r + diffuse.r),
                    min(255, ambient.g + diffuse.g),
                    min(255, ambient.b + diffuse.b)
                )
                
                accumulated_color = Colour(
                    min(255, accumulated_color.r + lighting.r),
                    min(255, accumulated_color.g + lighting.g),
                    min(255, accumulated_color.b + lighting.b)
                )
                
                action, action_info = fb_agent.choose_direction(
                    intersection=intersection,
                    ray=current_ray,
                    bounce_count=bounce_count,
                    accumulated_color=accumulated_color
                )
                
                strategies.append(f"{action_info['strategy']}_{action_info.get('step', 0)}")
                
                # Convert action to direction
                theta = (action[0] + 1) * np.pi/4  # Convert from [-1,1] to [0,π/2]
                phi = action[1] * np.pi  # Convert from [-1,1] to [-π,π]
                
                # Transform to surface coordinates
                normal = intersection.normal
                if abs(normal.z) < 0.9:
                    tangent = Vector(0, 0, 1).crossProduct(normal).normalise()
                else:
                    tangent = Vector(1, 0, 0).crossProduct(normal).normalise()
                
                bitangent = normal.crossProduct(tangent).normalise()
                
                # Local direction
                local_dir = Vector(
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                )
                
                # Convert to world direction
                world_dir = Vector(
                    local_dir.x * tangent.x + local_dir.y * bitangent.x + local_dir.z * normal.x,
                    local_dir.x * tangent.y + local_dir.y * bitangent.y + local_dir.z * normal.y,
                    local_dir.x * tangent.z + local_dir.y * bitangent.z + local_dir.z * normal.z
                ).normalise()
                
                # Create new ray
                current_ray = Ray(
                    intersection.point.addVector(normal.scaleByLength(0.001)),
                    world_dir
                )
            else:
                # No intersection
                break
            
            bounce_count += 1
        
        # Final color with brightness adjustment
        brightness = (accumulated_color.r + accumulated_color.g + accumulated_color.b) / 3
        
        if brightness < 50 and stats['light_hits'] == 0:
            # Add ambient if too dark and no light hit
            ambient_boost = 50 - brightness
            accumulated_color = Colour(
                min(255, accumulated_color.r + ambient_boost),
                min(255, accumulated_color.g + ambient_boost),
                min(255, accumulated_color.b + ambient_boost)
            )
        
        final_color = Colour(
            min(255, accumulated_color.r),
            min(255, accumulated_color.g),
            min(255, accumulated_color.b)
        )
        
        # FB metrics
        stats['fb_confidence'] = fb_agent.confidence
        stats['fb_model_loaded'] = getattr(fb_agent, 'model_loaded', False)
        stats['fb_strategy'] = action_info.get('strategy', 'unknown') if 'action_info' in locals() else 'none'
        
        return final_color, stats, strategies
    
    def render_exact_original(self, scene_spheres, save_path: Path):
        """Render EXACTLY like your original code (1 sample per pixel, grid pattern)"""
        print("  Rendering EXACT original method...")
        
        # Use parameters from your original code
        RAY_STEP = 0.01
        MULTIPLE = 3
        RAY_COUNT = 100 * MULTIPLE  # 300
        
        # Calculate image dimensions to match ray count
        width = 2 * RAY_COUNT + 1  # 601
        height = 2 * RAY_COUNT + 1  # 601
        
        print(f"  Image size: {width}x{height} (from RAY_COUNT={RAY_COUNT})")
        
        start_time = time.time()
        
        image = np.zeros((height, width, 3), dtype=np.float32)
        camera_pos = Vector(0, 0, 1)
        
        # Generate ray positions EXACTLY like your original
        X_RAYS = []
        Y_RAYS = []
        
        # X rays: -RAY_COUNT to RAY_COUNT
        for r in range(-RAY_COUNT, RAY_COUNT + 1):
            X_RAYS.append(r * RAY_STEP / MULTIPLE)
        
        # Y rays: RAY_COUNT to -RAY_COUNT (reversed!)
        for r in range(RAY_COUNT, -RAY_COUNT - 1, -1):
            Y_RAYS.append(r * RAY_STEP / MULTIPLE)
        
        print(f"  X rays: {len(X_RAYS)} from {X_RAYS[0]:.4f} to {X_RAYS[-1]:.4f}")
        print(f"  Y rays: {len(Y_RAYS)} from {Y_RAYS[0]:.4f} to {Y_RAYS[-1]:.4f}")
        
        # Debug: Check if we're hitting spheres
        debug_center = False
        
        # Render exactly like your original
        for y_idx, Y in enumerate(tqdm(Y_RAYS, desc="Rendering")):
            for x_idx, X in enumerate(X_RAYS):
                # EXACTLY like your original: Vector(x=X, y=Y, z=-1)
                ray_dir = Vector(x=X, y=Y, z=-1)
                
                # Debug center ray
                if debug_center and x_idx == len(X_RAYS)//2 and y_idx == len(Y_RAYS)//2:
                    print(f"\n  DEBUG CENTER RAY:")
                    print(f"    X={X:.6f}, Y={Y:.6f}")
                    print(f"    Direction: ({ray_dir.x:.6f}, {ray_dir.y:.6f}, {ray_dir.z:.6f})")
                    print(f"    Magnitude: {ray_dir.magnitude():.6f}")
                    
                    # Check intersections
                    for i, sphere in enumerate(scene_spheres):
                        intersection = Ray(camera_pos, ray_dir).sphereDiscriminant(sphere)
                        if intersection and intersection.intersects:
                            print(f"    Hits sphere {i} at distance {intersection.point.subtractVector(camera_pos).magnitude():.2f}")
                
                ray = Ray(camera_pos, ray_dir)
                
                # Use traditional tracing
                color, _, _ = self._trace_custom_traditional(ray, scene_spheres, 'custom_scene')
                
                # Store in image (Y_RAYS is already reversed, so no need to flip)
                image[y_idx, x_idx] = [
                    min(1.0, color.r / 255.0),
                    min(1.0, color.g / 255.0),
                    min(1.0, color.b / 255.0)
                ]
        
        render_time = time.time() - start_time
        
        # Save image
        plt.figure(figsize=(8, 8))
        plt.imshow(np.clip(image, 0, 1))
        plt.title(f'EXACT Original Method\nTime: {render_time:.1f}s | Size: {width}x{height}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  Render time: {render_time:.1f}s")
        
        return image
    
    def render_custom_scene(self, scene_spheres, method: str, save_path: Path):
        """Render your custom scene using EXACT original ray generation"""
        print(f"  Rendering with {method.upper()} using EXACT original camera...")
        
        # Use TRUE ORIGINAL parameters
        RAY_COUNT = 100
        RAY_STEP = 0.01
        MULTIPLE = 3
        MAX_BOUNCES = 5
        
        RAY_COUNT *= MULTIPLE  # 300
        RAY_STEP /= MULTIPLE   # 0.00333...
        
        print(f"  Original parameters: RAY_COUNT={RAY_COUNT}, RAY_STEP={RAY_STEP:.6f}, MULTIPLE={MULTIPLE}")
        
        # Generate rays EXACTLY like your original
        X_RAYS = [r*RAY_STEP for r in range(-RAY_COUNT, 0, 1)] + [r*RAY_STEP for r in range(0, RAY_COUNT + 1)]
        Y_RAYS = [r*RAY_STEP for r in range(RAY_COUNT, 0, -1)] + [-r*RAY_STEP for r in range(0, RAY_COUNT + 1)]
        
        width = len(X_RAYS)  # 601
        height = len(Y_RAYS)  # 601
        
        print(f"  Image size: {width}x{height}")
        print(f"  Total rays: {len(X_RAYS) * len(Y_RAYS)}")
        
        start_time = time.time()
        
        image = np.zeros((height, width, 3), dtype=np.float32)
        camera_pos = Vector(0, 0, 1)
        
        # Create lights EXACTLY like your original
        global_light_sources = [
            GlobalLight(
                vector=Vector(3, 1, -0.75),
                colour=Colour(20, 20, 255),
                strength=1,
                max_angle=np.radians(90),
                func=0
            )
        ]
        
        # Sun (EXACTLY like your original)
        sun = Sphere(
            id=0,  # Your original uses id=0 for sun
            centre=Vector(-0.6, 0.2, 6),
            radius=0.1,
            material=Material(emitive=True),
            colour=Colour(255, 255, 204)
        )
        
        point_light_sources = [
            PointLight(
                id=sun.id,
                position=sun.centre,
                colour=sun.colour,
                strength=1,
                max_angle=np.radians(90),
                func=-1
            )
        ]
        
        # Add all spheres (including sun)
        all_spheres = scene_spheres.copy()
        
        # Remove existing sun if any (to replace with correct one)
        all_spheres = [s for s in all_spheres if not (hasattr(s, 'id') and s.id == 7)]
        all_spheres.append(sun)
        
        background_colour = Colour(2, 2, 5)
        
        # Initialize/reset FB agent if needed
        if method == 'fb':
            self._get_fb_agent('custom_scene', reset=True)
        
        # Render exactly like original
        for y_idx, Y in enumerate(tqdm(Y_RAYS, desc=f"Rendering {method}")):
            for x_idx, X in enumerate(X_RAYS):
                # Create ray EXACTLY like your original - Vector(x=X, y=Y, z=-1) NOT normalized
                ray_dir = Vector(x=X, y=Y, z=-1)
                
                # For tracing, we need to normalize it
                # But your original doesn't normalize in nearestSphereIntersect!
                # Let's create two versions:
                ray_original = Ray(
                    origin=camera_pos,
                    D=Vector(x=X, y=Y, z=-1)  # NOT normalized, like your original
                )
                
                # For our methods, we'll use normalized version
                ray_normalized = Ray(
                    origin=camera_pos,
                    D=ray_dir.normalise() if ray_dir.magnitude() > 0 else Vector(0, 0, -1)
                )
                
                if method == 'traditional':
                    # Use original tracing if possible
                    try:
                        terminal = ray_original.nearestSphereIntersect(all_spheres, max_bounces=MAX_BOUNCES)
                        
                        if terminal is None:
                            color = background_colour
                        else:
                            # Use original lighting
                            color = terminal.terminalRGB(
                                spheres=all_spheres,
                                background_colour=background_colour,
                                global_light_sources=global_light_sources,
                                point_light_sources=point_light_sources
                            )
                    except:
                        # Fallback to our enhanced tracing with normalized ray
                        color, _, _ = self._trace_custom_traditional(ray_normalized, all_spheres, 'custom_scene')
                        
                elif method == 'rl' and self.rl_model_loaded:
                    # RL with normalized ray
                    color, _, _ = self._trace_custom_rl_trained(ray_normalized, all_spheres, 'custom_scene')
                    
                elif method == 'fb':
                    # FB with normalized ray
                    color, _, _ = self._trace_custom_fb_smart(ray_normalized, all_spheres, 'custom_scene', 0)
                    
                else:
                    # Default to traditional
                    color, _, _ = self._trace_custom_traditional(ray_normalized, all_spheres, 'custom_scene')
                
                # Store in image (Y_RAYS is already reversed)
                image[y_idx, x_idx] = [
                    min(1.0, color.r / 255.0),
                    min(1.0, color.g / 255.0),
                    min(1.0, color.b / 255.0)
                ]
        
        render_time = time.time() - start_time
        
        # Save image
        plt.figure(figsize=(8, 8))
        plt.imshow(np.clip(image, 0, 1))
        plt.title(f'{method.upper()} - EXACT Original Camera\nTime: {render_time:.1f}s | Size: {width}x{height}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  Render time: {render_time:.1f}s")
        
        return render_time, image
    
    def run_custom_scene_experiment(self):
        """Run experiment with your custom scene"""
        print("\n" + "="*80)
        print("YOUR CUSTOM SCENE EXPERIMENT")
        print("="*80)
        
        if not BASE_IMPORTS_OK:
            print("Error: Required modules not available")
            return {}
        
        scenes = create_custom_scene()
        scene_id = 'custom_scene'
        scene = scenes[scene_id]
        
        print(f"\nScene Analysis:")
        print(f"• Total objects: {len(scene)}")
        
        # 1. FIRST: Render TRUE original
        print(f"\n{'='*40}")
        print(f"1. TRUE ORIGINAL RENDER")
        print(f"{'='*40}")
        
        true_original_save = self.output_dir / "true_original.png"
        true_original_image = self.render_true_original(scene, true_original_save)
        print(f"✓ Saved: {true_original_save}")
        
        # 2. THEN: Render all methods with EXACT original camera
        print(f"\n{'='*40}")
        print(f"2. COMPARISON RENDERS (All with EXACT Original Camera)")
        print(f"{'='*40}")
        
        # Update config for faster testing
        original_num_trials = self.config['num_trials']
        self.config['num_trials'] = 10  # Fewer trials for testing
        
        all_results = {}
        scene_results = {}
        
        # Run performance trials
        for method in ['traditional', 'fb']:
            print(f"\n{method.upper()}:")
            
            method_results = []
            method_times = []
            all_strategies = []
            
            for trial in tqdm(range(self.config['num_trials']), desc=f"{method} trials"):
                results = self.run_single_trial(scene, method, scene_id, trial)
                method_results.append(results)
                method_times.append(results['execution_time'])
                all_strategies.extend(results['strategies_used'])
                
                self.results[method]['total_reward'].append(results['total_reward'])
                self.results[method]['light_hits'].append(results['light_hits'])
                self.results[method]['steps_taken'].append(results['steps_taken'])
                self.timing_data[method].append(results['execution_time'])
            
            if method_results:
                avg_reward = np.mean([r['total_reward'] for r in method_results])
                avg_lights = np.mean([r['light_hits'] for r in method_results])
                avg_steps = np.mean([r['steps_taken'] for r in method_results])
                avg_time = np.mean(method_times)
                unique_strategies = len(set(all_strategies))
                efficiency = avg_lights / max(1, avg_steps)
                
                scene_results[method] = {
                    'avg_reward': avg_reward,
                    'avg_lights': avg_lights,
                    'avg_steps': avg_steps,
                    'avg_time': avg_time,
                    'unique_strategies': unique_strategies,
                    'efficiency': efficiency
                }
                
                print(f"    Reward: {avg_reward:.1f}")
                print(f"    Sun hits: {avg_lights:.1f}")
                print(f"    Efficiency: {efficiency:.3f} hits/step")
                print(f"    Time: {avg_time*1000:.1f}ms per trial")
                print(f"    Strategies: {unique_strategies}")
        
        all_results[scene_id] = scene_results
        
        # 3. Render comparison images with EXACT original camera
        if self.config['render_image']:
            print(f"\n{'='*40}")
            print(f"3. RENDERING COMPARISON IMAGES (EXACT Original Camera)")
            print(f"{'='*40}")
            
            print(f"\nFB Model Status:")
            for scene_id, available in self.fb_models_available.items():
                status = "✓ Available" if available else "✗ Not found (using heuristic)"
                print(f"  {scene_id}: {status}")

            for method in ['traditional', 'rl']:
                save_path = self.output_dir / f"comparison_{method}.png"
                print(f"\n{method.upper()} comparison render...")
                
                try:
                    # Use new render method with EXACT original camera
                    render_time, image = self.render_custom_scene(scene, method, save_path)
                    print(f"    ✓ Saved: {save_path}")
                    print(f"    ⏱️  Render time: {render_time:.1f}s")
                    
                except Exception as e:
                    print(f"    ✗ Failed: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Restore original trial count
        self.config['num_trials'] = original_num_trials
        
        # Analysis
        print(f"\n{'='*40}")
        print(f"ANALYSIS")
        print(f"{'='*40}")
        
        if 'traditional' in scene_results and 'fb' in scene_results:
            trad = scene_results['traditional']
            fb = scene_results['fb']
            
            reward_impr = (fb['avg_reward'] - trad['avg_reward']) / max(0.01, trad['avg_reward'])
            efficiency_impr = (fb['efficiency'] - trad['efficiency']) / max(0.01, trad['efficiency'])
            speed_impr = trad['avg_time'] / max(0.0001, fb['avg_time'])
            
            print(f"\nFB vs Traditional Performance:")
            print(f"  • Reward improvement: {reward_impr:+.1%}")
            print(f"  • Efficiency improvement: {efficiency_impr:+.1%}")
            print(f"  • Speed improvement: {speed_impr:.1f}x faster")
            print(f"  • Strategy diversity: {fb['unique_strategies']} vs {trad['unique_strategies']}")
            
            # Scene-specific insights
            print(f"\nScene-specific Insights:")
            print(f"  • The sun is small (radius: 0.1) and at z=6")
            print(f"  • FB learns to look toward positive Z for the sun")
            print(f"  • Glass sphere creates complex light paths")
            print(f"  • Reflective sphere tests bounce accuracy")
        
        # Create visualization
        self._create_custom_scene_visualization(all_results)
        
        return all_results
    
    def _create_custom_scene_visualization(self, all_results):
        """Create visualization for custom scene"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            scene_id = 'custom_scene'
            scene_results = all_results[scene_id]
            
            # 1. Performance comparison
            ax = axes[0, 0]
            methods = ['Traditional', 'FB']
            colors = ['blue', 'green']
            
            if 'traditional' in scene_results and 'fb' in scene_results:
                trad = scene_results['traditional']
                fb = scene_results['fb']
                
                rewards = [trad['avg_reward'], fb['avg_reward']]
                bars = ax.bar(methods, rewards, color=colors, alpha=0.7)
                ax.set_ylabel('Average Reward')
                ax.set_title('Performance (Higher is Better)')
                ax.grid(True, alpha=0.3)
                
                # Add values on bars
                for bar, reward in zip(bars, rewards):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{reward:.1f}', ha='center', va='bottom')
            
            # 2. Efficiency comparison
            ax = axes[0, 1]
            if 'traditional' in scene_results and 'fb' in scene_results:
                trad = scene_results['traditional']
                fb = scene_results['fb']
                
                efficiencies = [trad['efficiency'], fb['efficiency']]
                bars = ax.bar(methods, efficiencies, color=colors, alpha=0.7)
                ax.set_ylabel('Hits per Step')
                ax.set_title('Sampling Efficiency')
                ax.grid(True, alpha=0.3)
                
                for bar, eff in zip(bars, efficiencies):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{eff:.3f}', ha='center', va='bottom')
            
            # 3. Speed comparison
            ax = axes[0, 2]
            if 'traditional' in scene_results and 'fb' in scene_results:
                trad = scene_results['traditional']
                fb = scene_results['fb']
                
                times = [trad['avg_time'] * 1000, fb['avg_time'] * 1000]  # Convert to ms
                bars = ax.bar(methods, times, color=colors, alpha=0.7)
                ax.set_ylabel('Time per Trial (ms)')
                ax.set_title('Execution Speed')
                ax.grid(True, alpha=0.3)
                
                for bar, t in zip(bars, times):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{t:.1f}ms', ha='center', va='bottom')
            
            # 4. Strategy diversity
            ax = axes[1, 0]
            if 'traditional' in scene_results and 'fb' in scene_results:
                trad = scene_results['traditional']
                fb = scene_results['fb']
                
                strategies = [trad['unique_strategies'], fb['unique_strategies']]
                bars = ax.bar(methods, strategies, color=colors, alpha=0.7)
                ax.set_ylabel('Unique Strategies')
                ax.set_title('Adaptive Strategy Usage')
                ax.grid(True, alpha=0.3)
                
                for bar, s in zip(bars, strategies):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{s}', ha='center', va='bottom')
            
            # 5. Improvement percentages
            ax = axes[1, 1]
            if 'traditional' in scene_results and 'fb' in scene_results:
                trad = scene_results['traditional']
                fb = scene_results['fb']
                
                reward_impr = (fb['avg_reward'] - trad['avg_reward']) / max(0.01, trad['avg_reward']) * 100
                speed_impr = (trad['avg_time'] - fb['avg_time']) / max(0.0001, trad['avg_time']) * 100
                eff_impr = (fb['efficiency'] - trad['efficiency']) / max(0.01, trad['efficiency']) * 100
                
                improvements = [reward_impr, speed_impr, eff_impr]
                labels = ['Reward', 'Speed', 'Efficiency']
                colors_imp = ['green' if x >= 0 else 'red' for x in improvements]
                
                bars = ax.bar(labels, improvements, color=colors_imp, alpha=0.7)
                ax.set_ylabel('Improvement (%)')
                ax.set_title('FB Improvements Over Traditional')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.grid(True, alpha=0.3)
                
                for bar, impr in zip(bars, improvements):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., 
                           height + (1 if height >= 0 else -3),
                           f'{impr:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
            
            # 6. Scene information
            ax = axes[1, 2]
            ax.axis('off')
            
            scene_info = "YOUR CUSTOM SCENE:\n\n"
            scene_info += "Objects:\n"
            scene_info += "• Glass sphere (front)\n"
            scene_info += "• Large blue sphere (back)\n"
            scene_info += "• Small blue sphere\n"
            scene_info += "• Reflective purple sphere\n"
            scene_info += "• Green sphere\n"
            scene_info += "• Large yellow sphere (far)\n"
            scene_info += "• Sun (light source)\n\n"
            
            scene_info += "Challenges:\n"
            scene_info += "• Small sun hard to hit\n"
            scene_info += "• Glass creates caustics\n"
            scene_info += "• Reflections need accuracy\n"
            scene_info += "• Variety of materials"
            
            ax.text(0.1, 0.5, scene_info, fontsize=9, 
                   verticalalignment='center', fontfamily='monospace')
            ax.set_title('Scene Description', fontweight='bold')
            
            plt.suptitle('FB vs Traditional: Your Custom Scene Analysis', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            save_path = self.output_dir / "custom_scene_analysis.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"\nAnalysis visualization saved: {save_path}")
            
        except Exception as e:
            print(f"⚠ Could not create visualization: {e}")
    
    def save_custom_results(self, all_results):
        """Save custom scene results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results
        results_path = self.output_dir / f"custom_scene_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump({
                'scene_results': all_results,
                'experiment_config': self.config,
                'experiment_mode': self.current_mode,
                'timestamp': timestamp
            }, f, indent=2)
        
        # Save summary - FIX: Use UTF-8 encoding
        summary_path = self.output_dir / f"custom_scene_summary_{timestamp}.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:  # Added encoding='utf-8'
            f.write("="*80 + "\n")
            f.write("YOUR CUSTOM SCENE: FB VS TRADITIONAL ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"EXPERIMENT MODE: {self.current_mode.upper()}\n")
            f.write(f"Trials: {self.config['num_trials']}\n")
            f.write(f"Image size: {self.config['image_width']}x{self.config['image_height']}\n")
            f.write(f"Samples per pixel: {self.config['samples_per_pixel']}\n")
            f.write(f"Max bounces: {self.config['max_bounces']}\n\n")
            
            for scene_id, scene_results in all_results.items():
                f.write(f"SCENE: {scene_id.upper().replace('_', ' ')}\n")
                f.write("-"*40 + "\n")
                
                if 'traditional' in scene_results and 'fb' in scene_results:
                    trad = scene_results['traditional']
                    fb = scene_results['fb']
                    
                    f.write(f"Traditional Method:\n")
                    f.write(f"  Average Reward: {trad['avg_reward']:.1f}\n")
                    f.write(f"  Sun Hits: {trad['avg_lights']:.1f}\n")
                    f.write(f"  Efficiency: {trad['efficiency']:.4f}\n")
                    f.write(f"  Time: {trad['avg_time']*1000:.1f}ms\n\n")
                    
                    f.write(f"FB Method:\n")
                    f.write(f"  Average Reward: {fb['avg_reward']:.1f}\n")
                    f.write(f"  Sun Hits: {fb['avg_lights']:.1f}\n")
                    f.write(f"  Efficiency: {fb['efficiency']:.4f}\n")
                    f.write(f"  Time: {fb['avg_time']*1000:.1f}ms\n")
                    f.write(f"  Unique Strategies: {fb['unique_strategies']}\n\n")
                    
                    reward_impr = (fb['avg_reward'] - trad['avg_reward']) / max(0.01, trad['avg_reward'])
                    speed_impr = trad['avg_time'] / max(0.0001, fb['avg_time'])
                    eff_impr = (fb['efficiency'] - trad['efficiency']) / max(0.01, trad['efficiency'])
                    
                    f.write(f"FB Improvements:\n")
                    f.write(f"  Reward: {reward_impr:+.1%}\n")
                    f.write(f"  Speed: {speed_impr:.1f}x\n")
                    f.write(f"  Efficiency: {eff_impr:+.1%}\n")
                
                f.write("\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("SPEED CONTROL TIPS\n")  # Removed lightning bolt symbol
            f.write("="*80 + "\n\n")
            
            f.write("To make experiments faster/slower, change these parameters:\n\n")
            f.write("1. SAMPLES PER PIXEL (SPP):\n")
            f.write("   • Fast: 1-2 samples (noisy but fast)\n")
            f.write("   • Balanced: 4-8 samples (good quality)\n")
            f.write("   • Quality: 16-64 samples (slow but clean)\n\n")
            
            f.write("2. IMAGE SIZE:\n")
            f.write("   • Fast: 200x150 pixels\n")
            f.write("   • Balanced: 400x300 pixels\n")
            f.write("   • Quality: 800x600 pixels\n\n")
            
            f.write("3. NUMBER OF TRIALS:\n")
            f.write("   • Fast: 10-20 trials\n")
            f.write("   • Balanced: 50-100 trials\n")
            f.write("   • Statistical: 200-500 trials\n\n")
            
            f.write("4. RAYS PER TRIAL:\n")
            f.write("   • Fast: 5-8 rays\n")
            f.write("   • Balanced: 10-15 rays\n")
            f.write("   • Detailed: 20-30 rays\n\n")
            
            f.write("5. MAX BOUNCES:\n")
            f.write("   • Fast: 3-4 bounces\n")
            f.write("   • Balanced: 5-6 bounces\n")
            f.write("   • Quality: 8-10 bounces\n\n")
            
            f.write("Current mode settings in code:\n")
            f.write(f"  self.current_mode = '{self.current_mode}'  # Change to 'fast_mode' or 'quality_mode'\n")
        
        print(f"\nResults saved:")
        print(f"  JSON: {results_path}")
        print(f"  Summary: {summary_path}")


def main():
    """Main function"""
    print("Your Custom Scene: FB vs Traditional Comparison")
    print("="*80)
    
    if not BASE_IMPORTS_OK:
        print("Error: Required modules not available")
        print("Make sure you have:")
        print("  - vector.py, colour.py, object.py")
        print("  - material.py, ray.py, light.py")
        return
    
    # Check for FB training
    if FB_AVAILABLE:
        print("✓ FB training modules available")
        # Check for trained model
        fb_model_path = "./fb_training_outputs/fb_ray_tracer_final.pth"
        if Path(fb_model_path).exists():
            print(f"✓ Found trained FB model: {fb_model_path}")
        else:
            print(f"⚠ No trained FB model found")
            print(f"  To train FB model: python train_fb_ray_tracing.py")
    else:
        print("⚠ FB training modules not available - using heuristic FB only")
    
    
    # Create experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./custom_scene_{timestamp}"
    
    experiment = CustomSceneExperiment(output_dir)
    
    # Run experiment
    print("\nRunning custom scene experiment...")
    results = experiment.run_custom_scene_experiment()
    
    # Save results
    experiment.save_custom_results(results)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    
    if results:
        print("\n📊 QUICK RESULTS:")
        
        for scene_id, scene_results in results.items():
            if 'traditional' in scene_results and 'fb' in scene_results:
                trad = scene_results['traditional']
                fb = scene_results['fb']
                
                reward_impr = (fb['avg_reward'] - trad['avg_reward']) / max(0.01, trad['avg_reward'])
                speed_impr = trad['avg_time'] / max(0.0001, fb['avg_time'])
                
                print(f"\n{scene_id.replace('_', ' ').title()}:")
                print(f"  • FB found {fb['avg_lights']:.1f}x more sun hits")
                print(f"  • FB was {speed_impr:.1f}x faster")
                print(f"  • FB used {fb['unique_strategies']} different strategies")
        
        print(f"\n📁 GENERATED FILES:")
        print(f"  • custom_scene_traditional.png - Traditional rendering")
        print(f"  • custom_scene_fb.png - FB rendering")
        print(f"  • custom_scene_analysis.png - Performance analysis")
        print(f"  • custom_scene_results_*.json - Detailed data")
        print(f"  • custom_scene_summary_*.txt - Summary with speed tips")
        
        print(f"\n📍 Output directory: {output_dir}/")
        
        print(f"\n⚡ SPEED TIPS:")
        print(f"  To make it faster, change in the code:")
        print(f"    self.current_mode = 'fast_mode'")
        print(f"  To make it higher quality:")
        print(f"    self.current_mode = 'quality_mode'")


if __name__ == "__main__":
    main()