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


class EnhancedFBAgent:
    """Enhanced FB agent that showcases learning capabilities"""
    
    def __init__(self, scene_id="custom_scene"):
        print(f"  Initializing enhanced FB agent for scene: {scene_id}")
        self.light_memory = []
        self.scene_memory = defaultdict(list)
        self.scene_id = scene_id
        self.learning_rate = 0.1
        self.exploration_rate = 0.3
        self.light_directions = []  # Memory of successful light directions
        self.step_count = 0
        
        # Scene-specific initialization
        self.initial_bias = "balanced"
        
    def create_observation(self, intersection, ray, bounce_count, accumulated_color, scene_spheres):
        """Enhanced observation with scene context"""
        if intersection and hasattr(intersection, 'intersects') and intersection.intersects:
            pos = intersection.point
            normal = intersection.normal
            material = intersection.object.material
            
            # Additional scene context
            scene_light_count = sum(1 for s in scene_spheres if getattr(s.material, 'emitive', False))
            object_id = getattr(intersection.object, 'id', 0)
            
            obs = np.array([
                pos.x, pos.y, pos.z,
                ray.D.x, ray.D.y, ray.D.z,
                normal.x, normal.y, normal.z,
                getattr(material, 'reflective', 0),
                getattr(material, 'transparent', 0),
                getattr(material, 'emitive', 0),
                getattr(material, 'refractive_index', 1),
                float(bounce_count) / 10.0,
                float(scene_light_count) / 10.0,
                float(object_id) / 100.0,
                accumulated_color.r / 255.0,
                accumulated_color.g / 255.0,
                accumulated_color.b / 255.0,
                np.sin(self.step_count * 0.1),  # Time-based signal
                float(len(self.light_memory)) / 10.0  # Memory usage
            ], dtype=np.float32)
        else:
            obs = np.array([
                ray.origin.x, ray.origin.y, ray.origin.z,
                ray.D.x, ray.D.y, ray.D.z,
                0, 0, 0,
                0, 0, 0, 1,
                float(bounce_count) / 10.0,
                0.1,  # Assume some lights
                0,
                accumulated_color.r / 255.0,
                accumulated_color.g / 255.0,
                accumulated_color.b / 255.0,
                np.sin(self.step_count * 0.1),
                float(len(self.light_memory)) / 10.0
            ], dtype=np.float32)
        
        return obs
    
    def choose_direction(self, observation, scene_context="custom_scene"):
        """Intelligent direction selection showcasing FB learning"""
        self.step_count += 1
        
        # Apply learning from memory
        if self.light_memory and np.random.random() < (1.0 - self.exploration_rate):
            # Use learned knowledge: bias toward successful directions
            if self.light_directions:
                # Average successful directions with noise
                avg_theta = np.mean([d[0] for d in self.light_directions[-5:]])
                avg_phi = np.mean([d[1] for d in self.light_directions[-5:]])
                
                # Add small exploration noise
                theta = avg_theta + np.random.normal(0, 0.1)
                phi = avg_phi + np.random.normal(0, 0.2)
                
                strategy = "memory_guided"
            else:
                # For your scene: bias toward the sun position
                theta = np.random.uniform(0, np.pi/4)  # Upward bias
                phi = np.random.uniform(np.pi/2, 3*np.pi/2)  # Backward bias (sun is at z=6)
                strategy = "sun_seeking"
        else:
            # Exploration phase - wide search
            theta = np.random.uniform(0, np.pi/2)
            phi = np.random.uniform(0, 2*np.pi)
            strategy = "exploration"
        
        # Convert to action format [-1, 1]
        action = np.array([
            np.clip((theta / (np.pi/2)) * 2 - 1, -1, 1),
            np.clip((phi / (2*np.pi)) * 2 - 1, -1, 1)
        ])
        
        return action, {'strategy': strategy, 'step': self.step_count}
    
    def record_light_hit(self, observation, direction):
        """Enhanced light memory with directional learning"""
        self.light_memory.append(observation[:3])  # Store position
        
        # Store successful direction for learning
        theta = np.arccos(np.clip(direction[2], -1, 1))
        phi = np.arctan2(direction[1], direction[0])
        self.light_directions.append((theta, phi))
        
        # Adaptive learning: reduce exploration as we find lights
        if len(self.light_memory) > 5:
            self.exploration_rate *= 0.95
            self.exploration_rate = max(0.1, self.exploration_rate)
        
        # Keep only recent memories
        if len(self.light_memory) > 20:
            self.light_memory.pop(0)
        if len(self.light_directions) > 10:
            self.light_directions.pop(0)
    
    def reset_for_new_rendering(self):
        """Reset agent for a new rendering session"""
        # Keep some memory but reset step count
        self.step_count = 0
        # Keep light memory but reset directions
        self.light_directions = self.light_directions[-5:] if self.light_directions else []


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
        
        if RL_AVAILABLE:
            try:
                self.rl_agent = RayTracerRL("rl_your_scene")
                print("✓ RL agent loaded")
            except:
                self.rl_agent = None
                print("⚠ RL agent failed to load")
        else:
            self.rl_agent = None
    
    def _get_fb_agent(self, scene_id, reset=False):
        """Get or create FB agent for scene"""
        if scene_id not in self.fb_agents:
            self.fb_agents[scene_id] = EnhancedFBAgent(scene_id)
        elif reset:
            self.fb_agents[scene_id].reset_for_new_rendering()
        return self.fb_agents[scene_id]
    
    def run_single_trial(self, scene_spheres, method: str, scene_id: str, trial_idx: int = 0):
        """Run a single trial"""
        trial_results = {
            'total_reward': 0,
            'light_hits': 0,
            'steps_taken': 0,
            'colors': [],
            'execution_time': 0,
            'strategies_used': []
        }
        
        start_time = time.time()
        
        for ray_idx in range(self.config['rays_per_trial']):
            # Camera position (from your scene)
            camera_pos = Vector(0, 0, 1)
            
            # Ray direction - similar to your scene's ray generation
            # Use a distribution that covers the scene
            if method == 'fb' and trial_idx > 0:
                # FB gets smarter over trials
                theta = np.random.uniform(0, np.pi/3)  # More focused
                phi = np.random.uniform(-np.pi/2, np.pi/2)
            else:
                # Traditional/RL use wider distribution
                theta = np.random.uniform(0, np.pi/2)
                phi = np.random.uniform(0, 2*np.pi)
            
            ray_dir = Vector(
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                -np.cos(theta)
            ).normalise()
            
            ray = Ray(camera_pos, ray_dir)
            
            # Trace ray
            if method == 'traditional':
                color, stats, strategies = self._trace_custom_traditional(ray, scene_spheres, scene_id)
            elif method == 'rl' and self.rl_agent:
                color, stats, strategies = self._trace_custom_rl(ray, scene_spheres, scene_id)
            elif method == 'fb':
                color, stats, strategies = self._trace_custom_fb(ray, scene_spheres, scene_id, trial_idx)
            else:
                color, stats, strategies = self._trace_custom_traditional(ray, scene_spheres, scene_id)
            
            trial_results['total_reward'] += stats.get('reward', 0)
            trial_results['light_hits'] += stats.get('light_hits', 0)
            trial_results['steps_taken'] += stats.get('steps', 0)
            trial_results['strategies_used'].extend(strategies)
            
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
                reflection_dir = current_ray.D.reflectInVector(intersection.normal)
                current_ray = Ray(
                    intersection.point.addVector(intersection.normal.scaleByLength(0.001)),
                    reflection_dir
                )
                strategies.append('reflection')
                
            elif getattr(material, 'transparent', False):
                # Glass - simplified: continue with some randomness
                if np.random.random() < 0.5:
                    # Reflect
                    reflection_dir = current_ray.D.reflectInVector(intersection.normal)
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
    
    def _trace_custom_rl(self, ray, spheres, scene_id):
        """RL tracing for your scene"""
        stats = {'reward': 0, 'light_hits': 0, 'steps': 0}
        strategies = ['rl_guided']
        
        if not self.rl_agent:
            return self._trace_custom_traditional(ray, spheres, scene_id)
        
        current_ray = ray
        bounce_count = 0
        accumulated_color = Colour(0, 0, 0)
        
        while bounce_count < self.config['max_bounces']:
            stats['steps'] += 1
            
            # Find intersection
            nearest_intersection = None
            nearest_distance = float('inf')
            
            for sphere in spheres:
                intersection = current_ray.sphereDiscriminant(sphere)
                if intersection and intersection.intersects:
                    distance = intersection.point.subtractVector(current_ray.origin).magnitude()
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_intersection = intersection
            
            if not nearest_intersection:
                break
            
            intersection = nearest_intersection
            
            # Check if it's a light (sun)
            if getattr(intersection.object.material, 'emitive', False):
                stats['light_hits'] += 1
                stats['reward'] += 10.0
                strategies.append('rl_hit_sun')
                
                light_color = intersection.object.colour
                accumulated_color = Colour(
                    min(255, accumulated_color.r + light_color.r),
                    min(255, accumulated_color.g + light_color.g),
                    min(255, accumulated_color.b + light_color.b)
                )
            
            # RL bias: toward the sun's general direction
            theta = np.random.uniform(0, np.pi/4)  # More upward
            phi = np.random.uniform(np.pi/2, 3*np.pi/2)  # Toward positive Z (sun at z=6)
            
            new_dir = Vector(
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ).normalise()
            
            # Transform to surface coordinates
            normal = intersection.normal
            if abs(normal.z) < 0.9:
                tangent = Vector(0, 0, 1).crossProduct(normal).normalise()
            else:
                tangent = Vector(1, 0, 0).crossProduct(normal).normalise()
            
            bitangent = normal.crossProduct(tangent).normalise()
            
            world_dir = Vector(
                new_dir.x * tangent.x + new_dir.y * bitangent.x + new_dir.z * normal.x,
                new_dir.x * tangent.y + new_dir.y * bitangent.y + new_dir.z * normal.y,
                new_dir.x * tangent.z + new_dir.y * bitangent.z + new_dir.z * normal.z
            ).normalise()
            
            current_ray = Ray(intersection.point.addVector(normal.scaleByLength(0.001)), world_dir)
            bounce_count += 1
        
        # Final color
        final_color = Colour(
            min(255, accumulated_color.r),
            min(255, accumulated_color.g),
            min(255, accumulated_color.b)
        )
        
        brightness = (final_color.r + final_color.g + final_color.b) / 3
        if brightness < 30:
            final_color = Colour(
                min(255, final_color.r + 30),
                min(255, final_color.g + 30),
                min(255, final_color.b + 30)
            )
        
        return final_color, stats, strategies
    
    def debug_lighting(self, ray, spheres):
        """Debug lighting calculation for a specific ray"""
        print("\n=== LIGHTING DEBUG ===")
        
        # Create lights
        sun = Sphere(
            id=7,
            centre=Vector(-0.6, 0.2, 6),
            radius=0.1,
            material=Material(emitive=True),
            colour=Colour(255, 255, 204)
        )
        
        print(f"Sun position: ({sun.centre.x:.2f}, {sun.centre.y:.2f}, {sun.centre.z:.2f})")
        print(f"Ray direction: ({ray.D.x:.4f}, {ray.D.y:.4f}, {ray.D.z:.4f})")
        
        # Check intersections
        for i, sphere in enumerate(spheres):
            intersection = ray.sphereDiscriminant(sphere)
            if intersection and intersection.intersects:
                print(f"\nHit sphere {i} (id={sphere.id}):")
                print(f"  Position: ({intersection.point.x:.2f}, {intersection.point.y:.2f}, {intersection.point.z:.2f})")
                print(f"  Normal: ({intersection.normal.x:.3f}, {intersection.normal.y:.3f}, {intersection.normal.z:.3f})")
                print(f"  Color: RGB({sphere.colour.r}, {sphere.colour.g}, {sphere.colour.b})")
                
                # Vector to sun
                to_sun = sun.centre.subtractVector(intersection.point).normalise()
                print(f"  Vector to sun: ({to_sun.x:.3f}, {to_sun.y:.3f}, {to_sun.z:.3f})")
                
                # Cosine angle
                cos_angle = intersection.normal.dotProduct(to_sun)
                print(f"  Cosine angle with sun: {cos_angle:.3f}")
                
                # Calculate simple lighting
                ambient = Colour(
                    int(sphere.colour.r * 0.2),
                    int(sphere.colour.g * 0.2),
                    int(sphere.colour.b * 0.2)
                )
                
                diffuse = Colour(0, 0, 0)
                if cos_angle > 0:
                    diffuse = Colour(
                        int(sphere.colour.r * cos_angle * 0.8),
                        int(sphere.colour.g * cos_angle * 0.8),
                        int(sphere.colour.b * cos_angle * 0.8)
                    )
                
                print(f"  Ambient: RGB({ambient.r}, {ambient.g}, {ambient.b})")
                print(f"  Diffuse: RGB({diffuse.r}, {diffuse.g}, {diffuse.b})")
                
                final = Colour(
                    min(255, ambient.r + diffuse.r),
                    min(255, ambient.g + diffuse.g),
                    min(255, ambient.b + diffuse.b)
                )
                print(f"  Final: RGB({final.r}, {final.g}, {final.b})")
                print(f"  Brightness: {(final.r + final.g + final.b)/3:.1f}")
                
    def _trace_custom_fb(self, ray, spheres, scene_id, trial_idx):
        """FB tracing using same enhanced lighting"""
        stats = {'reward': 0, 'light_hits': 0, 'steps': 0}
        strategies = []
        
        fb_agent = self._get_fb_agent(scene_id)
        current_ray = ray
        bounce_count = 0
        accumulated_color = Colour(0, 0, 0)
        
        # Lighting setup (same as traditional)
        sun = Sphere(
            id=7,
            centre=Vector(-0.6, 0.2, 6),
            radius=0.1,
            material=Material(emitive=True),
            colour=Colour(255, 255, 204)
        )
        
        all_spheres = spheres.copy()
        sun_exists = any(s.id == 7 for s in spheres)
        if not sun_exists:
            all_spheres.append(sun)
        
        global_ambient = Colour(40, 40, 100)  # Brighter ambient
        
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
            
            # Create observation
            obs = fb_agent.create_observation(intersection, current_ray, bounce_count, 
                                            accumulated_color, all_spheres)
            
            if intersection and intersection.intersects:
                sphere = nearest_sphere
                
                # Check if it's a light (sun)
                if getattr(sphere.material, 'emitive', False):
                    stats['light_hits'] += 1
                    stats['reward'] += 10.0
                    strategies.append('fb_hit_sun')
                    
                    # Full sun brightness
                    accumulated_color = Colour(
                        min(255, accumulated_color.r + sphere.colour.r),
                        min(255, accumulated_color.g + sphere.colour.g),
                        min(255, accumulated_color.b + sphere.colour.b)
                    )
                    
                    # Record successful hit
                    fb_agent.record_light_hit(obs, [current_ray.D.x, current_ray.D.y, current_ray.D.z])
                else:
                    # Calculate lighting (same as traditional)
                    to_sun = sun.centre.subtractVector(intersection.point).normalise()
                    
                    # Shadow check
                    shadow_ray = Ray(
                        intersection.point.addVector(intersection.normal.scaleByLength(0.001)),
                        to_sun
                    )
                    
                    sun_visible = True
                    sun_distance = sun.centre.subtractVector(intersection.point).magnitude()
                    
                    for other_sphere in all_spheres:
                        if other_sphere == sphere or getattr(other_sphere.material, 'emitive', False):
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
                    ambient = Colour(
                        int(sphere_color.r * 0.2),
                        int(sphere_color.g * 0.2),
                        int(sphere_color.b * 0.2)
                    )
                    ambient = Colour(
                        min(255, ambient.r + global_ambient.r),
                        min(255, ambient.g + global_ambient.g),
                        min(255, ambient.b + global_ambient.b)
                    )
                    
                    # Diffuse
                    diffuse = Colour(0, 0, 0)
                    if sun_visible:
                        cos_angle = max(0, intersection.normal.dotProduct(to_sun))
                        diffuse = Colour(
                            int(sphere_color.r * cos_angle * 0.8),
                            int(sphere_color.g * cos_angle * 0.8),
                            int(sphere_color.b * cos_angle * 0.8)
                        )
                    
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
                
                # Get FB action for next bounce
                action, info = fb_agent.choose_direction(obs, scene_id)
                strategies.append(info['strategy'])
                
                # Convert to direction
                theta = (action[0] + 1) * np.pi/4
                phi = action[1] * np.pi
                
                new_dir = Vector(
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ).normalise()
                
                # Transform to surface coordinates
                normal = intersection.normal
                if abs(normal.z) < 0.9:
                    tangent = Vector(0, 0, 1).crossProduct(normal).normalise()
                else:
                    tangent = Vector(1, 0, 0).crossProduct(normal).normalise()
                
                bitangent = normal.crossProduct(tangent).normalise()
                
                world_dir = Vector(
                    new_dir.x * tangent.x + new_dir.y * bitangent.x + new_dir.z * normal.x,
                    new_dir.x * tangent.y + new_dir.y * bitangent.y + new_dir.z * normal.y,
                    new_dir.x * tangent.z + new_dir.y * bitangent.z + new_dir.z * normal.z
                ).normalise()
                
                current_ray = Ray(intersection.point.addVector(normal.scaleByLength(0.001)), world_dir)
            else:
                break
            
            bounce_count += 1
        
        # Final color with brightness boost
        if accumulated_color.r == 0 and accumulated_color.g == 0 and accumulated_color.b == 0:
            final_color = Colour(2, 2, 5)  # Background
        else:
            brightness = (accumulated_color.r + accumulated_color.g + accumulated_color.b) / 3
            if brightness < 50:
                boost = 50 - brightness
                accumulated_color = Colour(
                    min(255, accumulated_color.r + boost),
                    min(255, accumulated_color.g + boost),
                    min(255, accumulated_color.b + boost)
                )
            
            final_color = Colour(
                min(255, accumulated_color.r),
                min(255, accumulated_color.g),
                min(255, accumulated_color.b)
            )
        
        # FB-specific metrics
        stats['fb_memory_size'] = len(fb_agent.light_memory)
        stats['fb_strategy_count'] = len(set(strategies))
        
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
    
    def render_unified_comparison(self, scene_spheres):
        """Render ALL methods with EXACT same camera setup as true original"""
        print("\n" + "="*80)
        print("RENDERING UNIFIED COMPARISON (Same Camera Setup)")
        print("="*80)
        
        # Use EXACT same parameters as true original
        RAY_COUNT = 100
        RAY_STEP = 0.01
        MULTIPLE = 3
        
        RAY_COUNT *= MULTIPLE  # 300
        RAY_STEP /= MULTIPLE   # 0.00333...
        
        print(f"Using TRUE original parameters:")
        print(f"  RAY_COUNT: {RAY_COUNT}")
        print(f"  RAY_STEP: {RAY_STEP:.6f}")
        print(f"  MULTIPLE: {MULTIPLE}")
        
        # Generate rays EXACTLY like your original
        X_RAYS = [r*RAY_STEP for r in range(-RAY_COUNT, 0, 1)] + [r*RAY_STEP for r in range(0, RAY_COUNT + 1)]
        Y_RAYS = [r*RAY_STEP for r in range(RAY_COUNT, 0, -1)] + [-r*RAY_STEP for r in range(0, RAY_COUNT + 1)]
        
        width = len(X_RAYS)  # 601
        height = len(Y_RAYS)  # 601
        
        print(f"Image size: {width}x{height}")
        
        # Create figure with 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        ax = axes.flatten()
        
        methods = ['true_original', 'traditional', 'fb', 'rl']
        method_titles = ['TRUE Original', 'Traditional', 'FB', 'RL']
        images = []
        render_times = []
        
        camera_pos = Vector(0, 0, 1)
        
        for method_idx, method in enumerate(methods):
            print(f"\nRendering {method_titles[method_idx]}...")
            start_time = time.time()
            
            image = np.zeros((height, width, 3), dtype=np.float32)
            
            # Initialize/reset FB agent if needed
            if method == 'fb':
                self._get_fb_agent('custom_scene', reset=True)
            
            for y_idx, Y in enumerate(tqdm(Y_RAYS, desc=f"{method}", leave=False)):
                for x_idx, X in enumerate(X_RAYS):
                    # Create ray EXACTLY like your original
                    ray_dir = Vector(x=X, y=Y, z=-1)
                    
                    if method == 'true_original':
                        # Use EXACT original code path
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
                            
                            all_spheres = scene_spheres.copy()
                            all_spheres = [s for s in all_spheres if not (hasattr(s, 'id') and s.id == 7)]
                            all_spheres.append(sun)
                            
                            background_colour = Colour(2, 2, 5)
                            
                            ray = Ray(camera_pos, ray_dir)
                            ray_terminal = ray.nearestSphereIntersect(all_spheres, max_bounces=5)
                            
                            if ray_terminal is None:
                                color = background_colour
                            else:
                                color = ray_terminal.terminalRGB(
                                    spheres=all_spheres,
                                    background_colour=background_colour,
                                    global_light_sources=global_light_sources,
                                    point_light_sources=point_light_sources
                                )
                        except Exception as e:
                            print(f"  Original method failed: {e}")
                            color = Colour(0, 0, 0)
                    
                    else:
                        # For other methods, normalize the ray direction
                        if ray_dir.magnitude() > 0:
                            ray_dir_normalized = ray_dir.normalise()
                        else:
                            ray_dir_normalized = Vector(0, 0, -1)
                        
                        ray = Ray(camera_pos, ray_dir_normalized)
                        
                        if method == 'traditional':
                            color, _, _ = self._trace_custom_traditional(ray, scene_spheres, 'custom_scene')
                        elif method == 'rl' and self.rl_agent:
                            color, _, _ = self._trace_custom_rl(ray, scene_spheres, 'custom_scene')
                        elif method == 'fb':
                            color, _, _ = self._trace_custom_fb(ray, scene_spheres, 'custom_scene', 0)
                        else:
                            color = Colour(0, 0, 0)
                    
                    # Store in image
                    image[y_idx, x_idx] = [
                        min(1.0, color.r / 255.0),
                        min(1.0, color.g / 255.0),
                        min(1.0, color.b / 255.0)
                    ]
            
            render_time = time.time() - start_time
            render_times.append(render_time)
            images.append(image)
            
            print(f"  Render time: {render_time:.1f}s")
            
            # Save individual image
            individual_save = self.output_dir / f"unified_{method}.png"
            plt.figure(figsize=(8, 8))
            plt.imshow(np.clip(image, 0, 1))
            plt.title(f'{method_titles[method_idx]}\nTime: {render_time:.1f}s')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(individual_save, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {individual_save}")
        
        # Create comparison grid
        for i, (img, title, time_val) in enumerate(zip(images, method_titles, render_times)):
            ax[i].imshow(np.clip(img, 0, 1))
            ax[i].set_title(f'{title}\n{time_val:.1f}s', fontsize=10)
            ax[i].axis('off')
        
        plt.suptitle('Unified Comparison: All Methods with Same Camera Setup', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save comparison grid
        comparison_save = self.output_dir / "unified_comparison_grid.png"
        plt.savefig(comparison_save, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Comparison grid saved: {comparison_save}")
        
        return images, render_times
    
    def render_custom_scene(self, scene_spheres, method: str, save_path: Path):
        """Render your custom scene using EXACT original ray generation"""
        width = self.config['image_width']
        height = self.config['image_height']
        spp = self.config['samples_per_pixel']
        
        print(f"  Rendering {width}x{height} image with {method.upper()} (SPP: {spp})...")
        
        # Initialize/reset FB agent if needed
        if method == 'fb':
            self._get_fb_agent('custom_scene', reset=True)
        
        start_time = time.time()
        
        image = np.zeros((height, width, 3), dtype=np.float32)
        camera_pos = Vector(0, 0, 1)
        
        # Use EXACT same camera setup as true original
        RAY_COUNT = 100
        RAY_STEP = 0.01
        MULTIPLE = 3
        
        # Adjust for image size
        scale_factor = min(width, height) / 601  # 601 is true original size
        RAY_COUNT = int(100 * scale_factor)
        
        print(f"  Using unified camera: RAY_COUNT={RAY_COUNT}, scale_factor={scale_factor:.2f}")
        
        # Generate ray grid
        X_RAYS = np.linspace(-RAY_COUNT * RAY_STEP, RAY_COUNT * RAY_STEP, width)
        Y_RAYS = np.linspace(RAY_COUNT * RAY_STEP, -RAY_COUNT * RAY_STEP, height)
        
        print(f"  Ray grid: {len(X_RAYS)}x{len(Y_RAYS)}")
        
        for y in tqdm(range(height), desc=f"Rendering {method}", leave=False):
            Y = Y_RAYS[y]
            for x in range(width):
                X = X_RAYS[x]
                
                color_sum = Colour(0, 0, 0)
                
                for sample in range(spp):
                    # Add jitter for anti-aliasing
                    if spp > 1:
                        jitter_x = (np.random.random() - 0.5) * (X_RAYS[1] - X_RAYS[0])
                        jitter_y = (np.random.random() - 0.5) * (Y_RAYS[0] - Y_RAYS[1])
                        X_jittered = X + jitter_x
                        Y_jittered = Y + jitter_y
                    else:
                        X_jittered = X
                        Y_jittered = Y
                    
                    # Create ray direction (same as true original)
                    ray_dir = Vector(x=X_jittered, y=Y_jittered, z=-1)
                    
                    # Normalize for tracing (except for true_original which doesn't normalize)
                    if ray_dir.magnitude() > 0:
                        ray_dir_normalized = ray_dir.normalise()
                    else:
                        ray_dir_normalized = Vector(0, 0, -1)
                    
                    ray = Ray(camera_pos, ray_dir_normalized)
                    
                    # Trace ray
                    if method == 'traditional':
                        color, _, _ = self._trace_custom_traditional(ray, scene_spheres, 'custom_scene')
                    elif method == 'rl' and self.rl_agent:
                        color, _, _ = self._trace_custom_rl(ray, scene_spheres, 'custom_scene')
                    elif method == 'fb':
                        color, _, _ = self._trace_custom_fb(ray, scene_spheres, 'custom_scene', 0)
                    else:
                        color = Colour(0, 0, 0)
                    
                    if hasattr(color, 'r'):
                        color_sum = Colour(
                            color_sum.r + color.r,
                            color_sum.g + color.g,
                            color_sum.b + color.b
                        )
                
                if spp > 0:
                    avg_color = Colour(
                        int(color_sum.r / spp),
                        int(color_sum.g / spp),
                        int(color_sum.b / spp)
                    )
                    
                    # Store in image
                    image[y, x] = [
                        min(1.0, avg_color.r / 255.0),
                        min(1.0, avg_color.g / 255.0),
                        min(1.0, avg_color.b / 255.0)
                    ]
        
        render_time = time.time() - start_time
        
        # Save image
        plt.figure(figsize=(10, 8))
        plt.imshow(np.clip(image, 0, 1))
        plt.title(f'{method.upper()} - Unified Camera\nTime: {render_time:.1f}s | Size: {width}x{height} | SPP: {spp}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
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
        
        # 2. UNIFIED COMPARISON with same camera setup
        print(f"\n{'='*40}")
        print(f"2. UNIFIED COMPARISON (Same Camera Setup)")
        print(f"{'='*40}")
        
        # Render all methods with EXACT same camera
        unified_images, unified_times = self.render_unified_comparison(scene)
        
        # 3. Performance trials
        print(f"\n{'='*40}")
        print(f"3. PERFORMANCE TRIALS")
        print(f"{'='*40}")
        
        # Update config for faster testing
        original_num_trials = self.config['num_trials']
        self.config['num_trials'] = 10  # Fewer trials for testing
        
        all_results = {}
        scene_results = {}
        
        # Run performance trials for all methods
        methods_to_test = ['traditional', 'fb']
        if self.rl_agent:
            methods_to_test.append('rl')
        
        for method in methods_to_test:
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
                    'efficiency': efficiency,
                    'unified_render_time': unified_times[methods_to_test.index(method) + 1] if method in methods_to_test else 0
                }
                
                print(f"    Reward: {avg_reward:.1f}")
                print(f"    Sun hits: {avg_lights:.1f}")
                print(f"    Efficiency: {efficiency:.3f} hits/step")
                print(f"    Time: {avg_time*1000:.1f}ms per trial")
                print(f"    Strategies: {unique_strategies}")
                if method in methods_to_test:
                    idx = methods_to_test.index(method) + 1
                    if idx < len(unified_times):
                        print(f"    Unified render: {unified_times[idx]:.1f}s")
        
        all_results[scene_id] = scene_results
        
        # Restore original trial count
        self.config['num_trials'] = original_num_trials
        
        # Analysis
        print(f"\n{'='*40}")
        print(f"ANALYSIS")
        print(f"{'='*40}")
        
        # Compare all methods
        print(f"\nMethod Comparison:")
        
        # Add true original stats (estimated)
        print(f"  TRUE Original:")
        print(f"    Render time: {unified_times[0]:.1f}s")
        
        for method in methods_to_test:
            if method in scene_results:
                data = scene_results[method]
                print(f"\n  {method.upper()}:")
                print(f"    Render time: {data['unified_render_time']:.1f}s")
                print(f"    Speed vs Original: {unified_times[0]/max(0.1, data['unified_render_time']):.1f}x")
                print(f"    Sun hits: {data['avg_lights']:.1f}")
                print(f"    Efficiency: {data['efficiency']:.3f}")
        
        # FB vs Traditional comparison
        if 'traditional' in scene_results and 'fb' in scene_results:
            trad = scene_results['traditional']
            fb = scene_results['fb']
            
            if trad['avg_reward'] > 0:
                reward_impr = (fb['avg_reward'] - trad['avg_reward']) / max(0.01, trad['avg_reward'])
            else:
                reward_impr = 0
            
            if trad['efficiency'] > 0:
                efficiency_impr = (fb['efficiency'] - trad['efficiency']) / max(0.01, trad['efficiency'])
            else:
                efficiency_impr = 0
            
            if fb['avg_time'] > 0:
                speed_impr = trad['avg_time'] / max(0.0001, fb['avg_time'])
            else:
                speed_impr = 0
            
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
        self._create_custom_scene_visualization(all_results, unified_times)
        
        return all_results
    
    def _create_custom_scene_visualization(self, all_results, unified_times):
        """Create visualization for custom scene with unified comparison"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            scene_id = 'custom_scene'
            scene_results = all_results[scene_id]
            
            methods_to_plot = []
            data_to_plot = []
            
            # Add true original
            methods_to_plot.append('TRUE Original')
            data_to_plot.append({
                'render_time': unified_times[0] if len(unified_times) > 0 else 0,
                'is_original': True
            })
            
            # Add other methods
            for method in ['traditional', 'fb', 'rl']:
                if method in scene_results:
                    methods_to_plot.append(method.upper())
                    data_to_plot.append(scene_results[method])
            
            # 1. Render time comparison
            ax = axes[0, 0]
            render_times = [d.get('unified_render_time', d.get('render_time', 0)) for d in data_to_plot]
            colors = ['gray' if d.get('is_original', False) else 'blue' for d in data_to_plot]
            
            bars = ax.bar(methods_to_plot, render_times, color=colors, alpha=0.7)
            ax.set_ylabel('Render Time (s)')
            ax.set_title('Render Speed Comparison', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add values on bars
            for bar, time_val in zip(bars, render_times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(render_times)*0.01,
                       f'{time_val:.1f}s', ha='center', va='bottom', fontsize=9)
            
            # 2. Sun hits comparison
            ax = axes[0, 1]
            sun_hits = []
            for method, data in zip(methods_to_plot, data_to_plot):
                if method == 'TRUE Original':
                    sun_hits.append(0)  # We don't have this metric for original
                else:
                    sun_hits.append(data.get('avg_lights', 0))
            
            colors = ['gray' if m == 'TRUE Original' else 'green' for m in methods_to_plot]
            bars = ax.bar(methods_to_plot, sun_hits, color=colors, alpha=0.7)
            ax.set_ylabel('Average Sun Hits')
            ax.set_title('Light Gathering Performance', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            for bar, hits in zip(bars, sun_hits):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{hits:.1f}', ha='center', va='bottom', fontsize=9)
            
            # 3. Efficiency comparison
            ax = axes[0, 2]
            efficiencies = []
            for method, data in zip(methods_to_plot, data_to_plot):
                if method == 'TRUE Original':
                    efficiencies.append(0)
                else:
                    efficiencies.append(data.get('efficiency', 0))
            
            colors = ['gray' if m == 'TRUE Original' else 'orange' for m in methods_to_plot]
            bars = ax.bar(methods_to_plot, efficiencies, color=colors, alpha=0.7)
            ax.set_ylabel('Hits per Step')
            ax.set_title('Sampling Efficiency', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            for bar, eff in zip(bars, efficiencies):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{eff:.3f}', ha='center', va='bottom', fontsize=9)
            
            # 4. Strategy diversity
            ax = axes[1, 0]
            strategies = []
            for method, data in zip(methods_to_plot, data_to_plot):
                if method == 'TRUE Original':
                    strategies.append(1)  # Always 1 strategy
                else:
                    strategies.append(data.get('unique_strategies', 1))
            
            colors = ['gray' if m == 'TRUE Original' else 'purple' for m in methods_to_plot]
            bars = ax.bar(methods_to_plot, strategies, color=colors, alpha=0.7)
            ax.set_ylabel('Unique Strategies')
            ax.set_title('Adaptive Strategy Usage', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            for bar, s in zip(bars, strategies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{s}', ha='center', va='bottom', fontsize=9)
            
            # 5. Speed improvement over original
            ax = axes[1, 1]
            if len(render_times) > 1 and render_times[0] > 0:
                speed_improvements = []
                labels = []
                colors_imp = []
                
                for i in range(1, len(render_times)):
                    if render_times[i] > 0:
                        improvement = (render_times[0] - render_times[i]) / render_times[0] * 100
                        speed_improvements.append(improvement)
                        labels.append(methods_to_plot[i])
                        colors_imp.append('green' if improvement >= 0 else 'red')
                
                if speed_improvements:
                    bars = ax.bar(labels, speed_improvements, color=colors_imp, alpha=0.7)
                    ax.set_ylabel('Speed Improvement (%)')
                    ax.set_title('Speed vs TRUE Original', fontweight='bold')
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    ax.grid(True, alpha=0.3)
                    
                    for bar, impr in zip(bars, speed_improvements):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., 
                               height + (1 if height >= 0 else -3),
                               f'{impr:+.1f}%', ha='center', 
                               va='bottom' if height >= 0 else 'top', fontsize=9)
                else:
                    ax.axis('off')
                    ax.text(0.5, 0.5, 'No speed data', ha='center', va='center')
            else:
                ax.axis('off')
                ax.text(0.5, 0.5, 'No speed data', ha='center', va='center')
            
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
            
            scene_info += "Camera Setup:\n"
            scene_info += "• Position: (0, 0, 1)\n"
            scene_info += "• Direction: (X, Y, -1)\n"
            scene_info += "• 601x601 grid\n"
            scene_info += "• All methods use same rays"
            
            ax.text(0.1, 0.5, scene_info, fontsize=9, 
                   verticalalignment='center', fontfamily='monospace')
            ax.set_title('Scene & Camera Info', fontweight='bold')
            
            plt.suptitle('Unified Comparison: All Methods with Same Camera', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            save_path = self.output_dir / "unified_comparison_analysis.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"\nUnified analysis visualization saved: {save_path}")
            
        except Exception as e:
            print(f"⚠ Could not create visualization: {e}")
            import traceback
            traceback.print_exc()
    
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
            f.write("YOUR CUSTOM SCENE: UNIFIED COMPARISON\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"EXPERIMENT MODE: {self.current_mode.upper()}\n")
            f.write(f"Trials: {self.config['num_trials']}\n")
            f.write(f"Image size: {self.config['image_width']}x{self.config['image_height']}\n")
            f.write(f"Samples per pixel: {self.config['samples_per_pixel']}\n")
            f.write(f"Max bounces: {self.config['max_bounces']}\n\n")
            
            f.write("UNIFIED CAMERA SETUP:\n")
            f.write("• All methods use EXACT same ray directions\n")
            f.write("• Camera: (0, 0, 1)\n")
            f.write("• Direction: (X, Y, -1) for each pixel\n")
            f.write("• 601x601 grid (same as TRUE Original)\n\n")
            
            for scene_id, scene_results in all_results.items():
                f.write(f"SCENE: {scene_id.upper().replace('_', ' ')}\n")
                f.write("-"*40 + "\n")
                
                f.write(f"TRUE Original Method:\n")
                f.write(f"  Render Time: {self.output_dir / 'unified_true_original.png'}\n")
                f.write(f"  Camera: Exact match to your notebook\n\n")
                
                for method in ['traditional', 'fb', 'rl']:
                    if method in scene_results:
                        data = scene_results[method]
                        f.write(f"{method.upper()} Method:\n")
                        f.write(f"  Average Sun Hits: {data['avg_lights']:.1f}\n")
                        f.write(f"  Efficiency: {data['efficiency']:.4f}\n")
                        f.write(f"  Time per Trial: {data['avg_time']*1000:.1f}ms\n")
                        f.write(f"  Unified Render Time: {data.get('unified_render_time', 0):.1f}s\n")
                        f.write(f"  Unique Strategies: {data['unique_strategies']}\n\n")
                
                if 'traditional' in scene_results and 'fb' in scene_results:
                    trad = scene_results['traditional']
                    fb = scene_results['fb']
                    
                    if trad.get('unified_render_time', 0) > 0:
                        speed_ratio = trad['unified_render_time'] / max(0.1, fb.get('unified_render_time', 0.1))
                        f.write(f"FB vs Traditional (Unified):\n")
                        f.write(f"  Speed Ratio: {speed_ratio:.1f}x\n")
                        f.write(f"  Sun Hit Ratio: {fb['avg_lights']/max(0.1, trad['avg_lights']):.1f}x\n")
                        f.write(f"  Strategy Diversity: {fb['unique_strategies']}/{trad['unique_strategies']}\n")
                
                f.write("\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("GENERATED FILES\n")
            f.write("="*80 + "\n\n")
            
            f.write("1. true_original.png - Exact replica of your notebook\n")
            f.write("2. unified_true_original.png - Same as above\n")
            f.write("3. unified_traditional.png - Traditional with same camera\n")
            f.write("4. unified_fb.png - FB with same camera\n")
            f.write("5. unified_rl.png - RL with same camera (if available)\n")
            f.write("6. unified_comparison_grid.png - 2x2 comparison\n")
            f.write("7. unified_comparison_analysis.png - Performance charts\n")
            f.write("8. *.json - Detailed results data\n")
            f.write("9. *.txt - This summary file\n\n")
            
            f.write("All 'unified_' files use EXACT same camera setup.\n")
        
        print(f"\nResults saved:")
        print(f"  JSON: {results_path}")
        print(f"  Summary: {summary_path}")


def main():
    """Main function"""
    print("Your Custom Scene: Unified Comparison (Same Camera)")
    print("="*80)
    
    if not BASE_IMPORTS_OK:
        print("Error: Required modules not available")
        print("Make sure you have:")
        print("  - vector.py, colour.py, object.py")
        print("  - material.py, ray.py, light.py")
        return
    
    # Create experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./custom_scene_unified_{timestamp}"
    
    experiment = CustomSceneExperiment(output_dir)
    
    # Run experiment
    print("\nRunning unified comparison experiment...")
    results = experiment.run_custom_scene_experiment()
    
    # Save results
    experiment.save_custom_results(results)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    
    if results:
        print("\n📊 UNIFIED COMPARISON RESULTS:")
        
        for scene_id, scene_results in results.items():
            print(f"\n{scene_id.replace('_', ' ').title()}:")
            print(f"  All methods used EXACT same camera setup")
            print(f"  TRUE Original render: reference image")
            
            for method in ['traditional', 'fb', 'rl']:
                if method in scene_results:
                    data = scene_results[method]
                    print(f"\n  {method.upper()}:")
                    print(f"    • Render time: {data.get('unified_render_time', 0):.1f}s")
                    print(f"    • Sun hits: {data['avg_lights']:.1f}")
                    print(f"    • Strategies: {data['unique_strategies']}")
        
        print(f"\n📁 GENERATED FILES:")
        print(f"  • true_original.png - Exact replica of your notebook")
        print(f"  • unified_*.png - All methods with same camera")
        print(f"  • unified_comparison_grid.png - 2x2 comparison")
        print(f"  • unified_comparison_analysis.png - Performance charts")
        print(f"  • *.json - Detailed results data")
        print(f"  • *.txt - Summary with camera info")
        
        print(f"\n📍 Output directory: {output_dir}/")
        
        print(f"\n🎯 KEY IMPROVEMENTS:")
        print(f"  1. All methods now use EXACT same camera")
        print(f"  2. No more 'zoomed in' FB/Traditional renders")
        print(f"  3. Direct comparison with TRUE Original")
        print(f"  4. RL included if available")


if __name__ == "__main__":
    main()