"""
Improved FB Renderer - More efficient with sun finding
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import time

# Import your scene modules
try:
    from vector import Vector
    from colour import Colour
    from object import Sphere
    from material import Material
    from ray import Ray
    from light import GlobalLight, PointLight
    BASE_IMPORTS_OK = True
except ImportError as e:
    print(f"Error: Could not import base modules: {e}")
    print("Make sure vector.py, colour.py, object.py, material.py, ray.py, light.py are available")
    BASE_IMPORTS_OK = False

# Import FB model
try:
    from fb_ray_tracing import FBResearchAgent, FBConfig
    FB_AVAILABLE = True
except ImportError as e:
    print(f"Error: Could not import FB modules: {e}")
    print("Make sure fb_ray_tracing.py is available")
    FB_AVAILABLE = False

def create_your_custom_scene():
    """Create your exact custom scene"""
    print("Creating your custom scene...")
    
    # Materials
    base_material = Material(reflective=False)
    reflective_material = Material(reflective=True)
    glass = Material(reflective=False, transparent=True, refractive_index=1.52)
    emitive_material = Material(emitive=True)
    
    # Your spheres
    spheres = [
        # Glass sphere (red)
        Sphere(id=1, centre=Vector(-0.8, 0.6, 0), radius=0.3, 
               material=glass, colour=Colour(255, 100, 100)),
        # Large blue sphere
        Sphere(id=2, centre=Vector(0.8, -0.8, -10), radius=2.2,
               material=base_material, colour=Colour(204, 204, 255)),
        # Small blue sphere
        Sphere(id=3, centre=Vector(0.3, 0.34, 0.1), radius=0.2,
               material=base_material, colour=Colour(0, 51, 204)),
        # Reflective purple sphere
        Sphere(id=4, centre=Vector(5.6, 3, -2), radius=5,
               material=reflective_material, colour=Colour(153, 51, 153)),
        # Green sphere
        Sphere(id=5, centre=Vector(-0.8, -0.8, -0.2), radius=0.25,
               material=base_material, colour=Colour(153, 204, 0)),
        # Background yellow sphere
        Sphere(id=6, centre=Vector(-3, 10, -75), radius=30,
               material=base_material, colour=Colour(255, 204, 102)),
        # SUN - the light source
        Sphere(id=7, centre=Vector(-0.6, 0.2, 6), radius=0.1,
               material=emitive_material, colour=Colour(255, 255, 204))
    ]
    
    print(f"Scene created with {len(spheres)} spheres")
    print(f"Sun position: ({spheres[-1].centre.x:.1f}, {spheres[-1].centre.y:.1f}, {spheres[-1].centre.z:.1f})")
    
    return spheres

class ImprovedFBRenderer:
    """Improved renderer with better sun finding and efficiency"""
    
    def __init__(self, model_path=None):
        if not BASE_IMPORTS_OK:
            raise ImportError("Required scene modules not available")
        
        if not FB_AVAILABLE:
            raise ImportError("FB modules not available")
        
        # Load scene
        self.scene = create_your_custom_scene()
        
        # Sun position
        self.sun_position = Vector(-0.6, 0.2, 6)
        self.sun_radius = 0.1
        self.sun_color = Colour(255, 255, 204)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load FB model
        self.agent = None
        self.fb_model_loaded = False
        self.load_fb_model(model_path)
        
        # Rendering parameters
        self.max_bounces = 6
        self.samples_per_pixel = 4  # Reduced for speed
        self.direct_sun_sampling_prob = 0.3  # 30% chance to sample sun directly
        self.min_light_threshold = 0.1  # Minimum brightness to consider
        
        # Statistics
        self.stats = {
            'total_rays': 0,
            'sun_hits': 0,
            'direct_sun_hits': 0,
            'indirect_sun_hits': 0,
            'avg_bounces': 0,
            'render_time': 0,
            'fb_decisions': 0,
            'fallback_decisions': 0
        }
        
        # Light direction cache
        self.last_sun_direction = None
        self.sun_hit_positions = []
        
    def load_fb_model(self, model_path=None):
        """Load trained FB model"""
        if model_path is None:
            # Look for the most recent model
            model_dir = Path("./fb_training_outputs")
            if model_dir.exists():
                model_files = list(model_dir.glob("fb_your_scene_*.pth"))
                if model_files:
                    # Get most recent
                    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    model_path = model_files[0]
                    print(f"Found model: {model_path}")
        
        if model_path is None or not Path(model_path).exists():
            print("Warning: No FB model found. Using heuristic rendering.")
            return
        
        try:
            print(f"Loading FB model from {model_path}...")
            
            # Create config (must match training config)
            config = FBConfig(
                z_dim=32,
                f_hidden_dim=256,
                b_hidden_dim=128,
                num_forward_heads=2,
                num_layers=2,
                learning_rate=1e-3,
                batch_size=64,
                buffer_capacity=20000,
                fb_weight=1.0,
                contrastive_weight=0.3,
                predictive_weight=0.2,
                max_bounces=6
            )
            
            # Create agent
            self.agent = FBResearchAgent(config, device=self.device)
            
            # Load model
            self.agent.load(model_path)
            
            self.fb_model_loaded = True
            print("✓ FB model loaded successfully")
            
            # Check if agent has learned any light directions
            if hasattr(self.agent, 'light_memory') and len(self.agent.light_memory.get('encodings', [])) > 0:
                print(f"  Agent knows about {len(self.agent.light_memory['encodings'])} light sources")
            else:
                print("  Agent has no memory of light sources")
            
        except Exception as e:
            print(f"✗ Failed to load FB model: {e}")
            import traceback
            traceback.print_exc()
            self.agent = None
    
    def create_observation(self, intersection, ray, bounce_count, accumulated_color):
        """Create observation for FB model"""
        if intersection is None or not intersection.intersects:
            # Create default observation
            obs = np.zeros(22, dtype=np.float32)
            obs[0:3] = [ray.origin.x, ray.origin.y, ray.origin.z]
            obs[3:6] = [ray.D.x, ray.D.y, ray.D.z]
            obs[16] = float(bounce_count) / self.max_bounces
            return obs
        
        pos = intersection.point
        direction = ray.D
        normal = intersection.normal
        material = intersection.object.material
        
        # Sun direction (important for learning!)
        sun_dir = self.sun_position.subtractVector(pos).normalise()
        
        # Normalize color
        color_norm = np.array([
            accumulated_color.r / 255.0,
            accumulated_color.g / 255.0,
            accumulated_color.b / 255.0
        ], dtype=np.float32)
        
        # Material type
        is_reflective = float(material.reflective)
        is_transparent = float(material.transparent)
        is_emitive = float(material.emitive)
        
        # Create observation
        obs = np.array([
            # Position (3)
            pos.x, pos.y, pos.z,
            # Direction (3)
            direction.x, direction.y, direction.z,
            # Normal (3)
            normal.x, normal.y, normal.z,
            # Material properties (4)
            is_reflective, is_transparent, is_emitive, material.refractive_index,
            # Current color (3)
            color_norm[0], color_norm[1], color_norm[2],
            # Bounce and history (3)
            float(bounce_count) / self.max_bounces,
            0.0,  # through_count
            float(intersection.object.id) / 100.0,
            # Sun direction (3) - CRITICAL for learning!
            sun_dir.x, sun_dir.y, sun_dir.z
        ], dtype=np.float32)
        
        return obs
    
    def sample_sun_direction(self, intersection_point, strategy="smart"):
        """Sample direction toward the sun with different strategies"""
        
        if strategy == "direct":
            # Direct line to sun
            to_sun = self.sun_position.subtractVector(intersection_point)
            return to_sun.normalise()
        
        elif strategy == "importance":
            # Importance sampling based on sun solid angle
            # The sun is small (radius 0.1) and far (distance ~6), so solid angle is small
            to_sun_center = self.sun_position.subtractVector(intersection_point)
            distance = to_sun_center.magnitude()
            
            # Sun angular radius
            angular_radius = np.arctan(self.sun_radius / distance)
            
            # Sample within sun cone
            r1 = np.random.random()
            r2 = np.random.random()
            
            theta = angular_radius * np.sqrt(r1)  # Cosine weighted
            phi = 2 * np.pi * r2
            
            # Perturb from center direction
            if abs(to_sun_center.z) > 0.9:
                tangent = Vector(1, 0, 0)
            else:
                tangent = Vector(0, 0, 1).crossProduct(to_sun_center)
            tangent = tangent.normalise()
            bitangent = to_sun_center.crossProduct(tangent).normalise()
            
            # Local perturbation
            local_dir = Vector(
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            )
            
            # Transform to world
            world_dir = Vector(
                local_dir.x * tangent.x + local_dir.y * bitangent.x + local_dir.z * to_sun_center.x,
                local_dir.x * tangent.y + local_dir.y * bitangent.y + local_dir.z * to_sun_center.y,
                local_dir.x * tangent.z + local_dir.y * bitangent.z + local_dir.z * to_sun_center.z
            ).normalise()
            
            return world_dir
        
        else:  # "smart" - use FB if available, otherwise importance sampling
            if self.fb_model_loaded and self.agent:
                try:
                    # Create a dummy observation for FB
                    dummy_obs = np.zeros(22, dtype=np.float32)
                    dummy_obs[0:3] = [intersection_point.x, intersection_point.y, intersection_point.z]
                    dummy_obs[-3:] = [self.sun_position.x, self.sun_position.y, self.sun_position.z]  # Sun direction
                    
                    # Get action from FB
                    action, info = self.agent.choose_direction_research(
                        dummy_obs, 
                        scene_context="custom_scene", 
                        exploration_phase="exploit"
                    )
                    
                    # Convert action to direction
                    theta = (action[0] + 1) * np.pi/4
                    phi = action[1] * np.pi
                    
                    # Create direction from angles
                    local_dir = Vector(
                        np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta)
                    )
                    
                    # Transform based on surface normal (we'll use up vector as default)
                    normal = Vector(0, 0, 1)  # Default upward
                    if abs(normal.z) > 0.9:
                        tangent = Vector(1, 0, 0)
                    else:
                        tangent = Vector(0, 0, 1).crossProduct(normal)
                    tangent = tangent.normalise()
                    bitangent = normal.crossProduct(tangent).normalise()
                    
                    world_dir = Vector(
                        local_dir.x * tangent.x + local_dir.y * bitangent.x + local_dir.z * normal.x,
                        local_dir.x * tangent.y + local_dir.y * bitangent.y + local_dir.z * normal.y,
                        local_dir.x * tangent.z + local_dir.y * bitangent.z + local_dir.z * normal.z
                    ).normalise()
                    
                    return world_dir
                    
                except Exception as e:
                    print(f"FB direction failed: {e}, falling back to importance sampling")
                    return self.sample_sun_direction(intersection_point, "importance")
            else:
                return self.sample_sun_direction(intersection_point, "importance")
    
    def calculate_lighting_original_style(self, intersection, ray, sun_visible=True):
        """Calculate lighting in the style of your original renderer"""
        
        sphere_color = intersection.object.colour
        
        if intersection.object.id == 7:  # Sun
            self.stats['sun_hits'] += 1
            if hasattr(ray, 'origin') and ray.origin.z > 0:  # Coming from camera
                self.stats['direct_sun_hits'] += 1
            else:
                self.stats['indirect_sun_hits'] += 1
            return self.sun_color
        
        # Your original uses: GlobalLight + PointLight
        # Let's approximate it
        
        # 1. GLOBAL LIGHT (blue ambient from direction (3, 1, -0.75))
        global_light_dir = Vector(3, 1, -0.75).normalise()
        global_cos = max(0, intersection.normal.dotProduct(global_light_dir))
        global_strength = 0.3  # 30% of global light
        
        global_contrib = Colour(
            int(20 * global_cos * global_strength),  # Blue ambient: Colour(20, 20, 255)
            int(20 * global_cos * global_strength),
            int(255 * global_cos * global_strength)
        )
        
        # 2. SUN LIGHT (if visible)
        sun_contrib = Colour(0, 0, 0)
        if sun_visible:
            to_sun = self.sun_position.subtractVector(intersection.point).normalise()
            distance = intersection.point.distanceFrom(self.sun_position)
            
            # Distance attenuation (like your original PointLight with func=-1)
            attenuation = 1.0 / (distance ** 2) if distance > 0 else 1.0
            attenuation = min(1.0, attenuation * 100)  # Scale factor
            
            # Diffuse component
            cos_angle = max(0, intersection.normal.dotProduct(to_sun))
            
            # BRIGHTER like your original
            sun_strength = 0.9 * attenuation
            
            sun_contrib = Colour(
                int(self.sun_color.r * cos_angle * sun_strength),
                int(self.sun_color.g * cos_angle * sun_strength),
                int(self.sun_color.b * cos_angle * sun_strength)
            )
        
        # 3. Combine: (global + sun) * sphere_color
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
        
        return final_light
    
    def is_sun_visible(self, intersection_point, normal):
        """Check if sun is visible from intersection point"""
        # Create shadow ray
        to_sun = self.sun_position.subtractVector(intersection_point)
        sun_distance = to_sun.magnitude()
        to_sun_dir = to_sun.normalise()
        
        shadow_ray = Ray(
            intersection_point.addVector(normal.scaleByLength(0.001)),
            to_sun_dir
        )
        
        # Check for occlusions
        for sphere in self.scene:
            if sphere.id == 7:  # Skip sun itself
                continue
            
            shadow_intersect = shadow_ray.sphereDiscriminant(sphere)
            if shadow_intersect and shadow_intersect.intersects:
                shadow_dist = shadow_intersect.point.distanceFrom(intersection_point)
                if shadow_dist < sun_distance:
                    return False
        
        return True
    
    def trace_ray_improved(self, ray, max_bounces=6):
        """Improved ray tracing with multiple strategies"""
        current_ray = ray
        bounce_count = 0
        accumulated_color = Colour(0, 0, 0)
        total_light_contrib = 0
        
        # Background color (like your original)
        background_color = Colour(2, 2, 5)
        
        while bounce_count < max_bounces:
            self.stats['total_rays'] += 1
            
            # Find nearest intersection
            nearest_intersection = None
            nearest_distance = float('inf')
            nearest_sphere = None
            
            for sphere in self.scene:
                intersection = current_ray.sphereDiscriminant(sphere)
                if intersection and intersection.intersects:
                    distance = intersection.point.distanceFrom(current_ray.origin)
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_intersection = intersection
                        nearest_sphere = sphere
            
            if not nearest_intersection:
                # Hit background
                if bounce_count == 0:
                    accumulated_color = background_color
                break
            
            intersection = nearest_intersection
            sphere = nearest_sphere
            
            # Check if it's the sun
            if sphere.id == 7:
                # Direct sun hit - full brightness
                sun_color = self.calculate_lighting_original_style(intersection, current_ray, sun_visible=True)
                accumulated_color = sun_color
                self.stats['sun_hits'] += 1
                break
            
            # Check sun visibility for lighting
            sun_visible = self.is_sun_visible(intersection.point, intersection.normal)
            
            # Calculate lighting
            lighting = self.calculate_lighting_original_style(intersection, current_ray, sun_visible)
            
            # Add to accumulated color with Russian Roulette termination
            brightness = (lighting.r + lighting.g + lighting.b) / 3 / 255.0
            
            # Russian Roulette: continue with probability based on brightness
            if bounce_count > 2:
                continue_prob = min(0.95, brightness * 2)
                if np.random.random() > continue_prob:
                    break
            
            accumulated_color = Colour(
                min(255, accumulated_color.r + lighting.r),
                min(255, accumulated_color.g + lighting.g),
                min(255, accumulated_color.b + lighting.b)
            )
            
            total_light_contrib += brightness
            
            # Choose next direction with multiple strategies
            material = sphere.material
            
            # Strategy 1: Direct sun sampling (for efficiency)
            if np.random.random() < self.direct_sun_sampling_prob and sun_visible:
                # Try to hit sun directly
                next_dir = self.sample_sun_direction(intersection.point, "direct")
                strategy = "direct_sun"
                self.stats['fallback_decisions'] += 1
                
            # Strategy 2: Material-based
            elif getattr(material, 'reflective', False):
                # Mirror reflection - FIXED: use reflectInVector instead of reflect
                next_dir = current_ray.D.reflectInVector(intersection.normal)
                strategy = "mirror"
                self.stats['fallback_decisions'] += 1
                
            elif getattr(material, 'transparent', False):
                # Glass - mix of reflection and transmission
                if np.random.random() < 0.5:
                    next_dir = current_ray.D.reflectInVector(intersection.normal)  # FIXED here too
                else:
                    next_dir = current_ray.D  # Continue straight
                strategy = "glass"
                self.stats['fallback_decisions'] += 1
                
            else:
                # Strategy 3: Use FB model or importance sampling toward sun
                if self.fb_model_loaded and total_light_contrib < self.min_light_threshold:
                    # Use FB to find light
                    obs = self.create_observation(intersection, current_ray, bounce_count, accumulated_color)
                    try:
                        action, info = self.agent.choose_direction_research(
                            obs, 
                            scene_context="custom_scene", 
                            exploration_phase="exploit"
                        )
                        
                        # Convert action to direction
                        theta = (action[0] + 1) * np.pi/4
                        phi = action[1] * np.pi
                        
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
                        next_dir = Vector(
                            local_dir.x * tangent.x + local_dir.y * bitangent.x + local_dir.z * normal.x,
                            local_dir.x * tangent.y + local_dir.y * bitangent.y + local_dir.z * normal.y,
                            local_dir.x * tangent.z + local_dir.y * bitangent.z + local_dir.z * normal.z
                        ).normalise()
                        
                        strategy = "fb_guided"
                        self.stats['fb_decisions'] += 1
                        
                    except Exception as e:
                        # FB failed, fallback to importance sampling toward sun
                        next_dir = self.sample_sun_direction(intersection.point, "importance")
                        strategy = "importance_fallback"
                        self.stats['fallback_decisions'] += 1
                else:
                    # Diffuse bounce with sun bias
                    if sun_visible and np.random.random() < 0.7:
                        # Bias toward sun
                        next_dir = self.sample_sun_direction(intersection.point, "importance")
                        strategy = "sun_biased_diffuse"
                    else:
                        # Pure diffuse
                        # Cosine-weighted hemisphere sampling
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
                        
                        next_dir = Vector(
                            local_dir.x * tangent.x + local_dir.y * bitangent.x + local_dir.z * normal.x,
                            local_dir.x * tangent.y + local_dir.y * bitangent.y + local_dir.z * normal.y,
                            local_dir.x * tangent.z + local_dir.y * bitangent.z + local_dir.z * normal.z
                        ).normalise()
                        strategy = "pure_diffuse"
                    
                    self.stats['fallback_decisions'] += 1
            
            # Create new ray
            current_ray = Ray(
                intersection.point.addVector(intersection.normal.scaleByLength(0.001)),
                next_dir
            )
            
            bounce_count += 1
        
        self.stats['avg_bounces'] += bounce_count
        
        # If accumulated color is too dark, add some ambient
        final_brightness = (accumulated_color.r + accumulated_color.g + accumulated_color.b) / 3
        if final_brightness < 30 and self.stats['sun_hits'] == 0:
            # Add ambient boost
            boost = 50 - final_brightness
            accumulated_color = Colour(
                min(255, accumulated_color.r + boost),
                min(255, accumulated_color.g + boost),
                min(255, accumulated_color.b + boost)
            )
        
        return accumulated_color
    
    def render_image_fast(self, width=400, height=300, output_path=None):
        """Fast rendering with adaptive sampling"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"./fb_render_fast_{timestamp}.png"
        
        print(f"\nRendering image: {width}x{height}")
        print(f"Max bounces: {self.max_bounces}")
        print(f"Direct sun sampling: {self.direct_sun_sampling_prob*100:.0f}%")
        
        # Reset stats
        self.stats = {
            'total_rays': 0,
            'sun_hits': 0,
            'direct_sun_hits': 0,
            'indirect_sun_hits': 0,
            'avg_bounces': 0,
            'render_time': 0,
            'fb_decisions': 0,
            'fallback_decisions': 0
        }
        
        start_time = time.time()
        
        # Create image buffer
        image = np.zeros((height, width, 3), dtype=np.float32)
        
        # Camera position
        camera_pos = Vector(0, 0, 1)
        
        # Adaptive sampling: fewer samples in dark areas
        max_samples = 4
        min_samples = 1
        
        for y in tqdm(range(height), desc="Rendering"):
            for x in range(width):
                pixel_samples = min_samples
                pixel_color = Colour(0, 0, 0)
                
                # Simple camera rays (no antialiasing for speed)
                u = (x / width - 0.5) * 2.0
                v = (y / height - 0.5) * -2.0  # Flip Y
                
                # Adjust for aspect ratio
                aspect_ratio = width / height
                u *= aspect_ratio
                
                # Create ray direction
                fov = np.pi / 3  # 60 degrees
                ray_dir = Vector(u * np.tan(fov/2), v * np.tan(fov/2), -1).normalise()
                
                # Trace primary ray
                primary_ray = Ray(camera_pos, ray_dir)
                color = self.trace_ray_improved(primary_ray, self.max_bounces)
                
                # Store pixel
                image[y, x] = [
                    min(1.0, color.r / 255.0),
                    min(1.0, color.g / 255.0),
                    min(1.0, color.b / 255.0)
                ]
        
        render_time = time.time() - start_time
        self.stats['render_time'] = render_time
        
        # Calculate averages
        if self.stats['total_rays'] > 0:
            self.stats['avg_bounces'] /= self.stats['total_rays']
        
        # Save image
        plt.figure(figsize=(10, 8))
        plt.imshow(np.clip(image, 0, 1))
        plt.title(f'FB Render - Fast Mode\n{width}x{height}, {render_time:.1f}s')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nRender complete!")
        print(f"Image saved: {output_path}")
        print(f"Render time: {render_time:.1f} seconds")
        
        return image, output_path
    
    def print_statistics(self):
        """Print detailed statistics"""
        print("\n" + "="*60)
        print("DETAILED RENDERING STATISTICS")
        print("="*60)
        print(f"Total rays traced: {self.stats['total_rays']:,}")
        print(f"Sun hits: {self.stats['sun_hits']:,}")
        print(f"  Direct sun hits: {self.stats['direct_sun_hits']:,}")
        print(f"  Indirect sun hits: {self.stats['indirect_sun_hits']:,}")
        print(f"Sun hit rate: {(self.stats['sun_hits'] / max(1, self.stats['total_rays'])) * 100:.2f}%")
        print(f"Average bounces per ray: {self.stats['avg_bounces']:.2f}")
        print(f"Render time: {self.stats['render_time']:.1f} seconds")
        
        if self.stats['total_rays'] > 0:
            rays_per_second = self.stats['total_rays'] / self.stats['render_time']
            print(f"Performance: {rays_per_second:,.0f} rays/second")
        
        total_decisions = self.stats['fb_decisions'] + self.stats['fallback_decisions']
        if total_decisions > 0:
            fb_percent = (self.stats['fb_decisions'] / total_decisions) * 100
            print(f"FB decisions: {self.stats['fb_decisions']:,} ({fb_percent:.1f}%)")
            print(f"Fallback decisions: {self.stats['fallback_decisions']:,} ({100-fb_percent:.1f}%)")
        
        if self.fb_model_loaded:
            print(f"Model used: Trained FB model")
            if hasattr(self.agent, 'research_stats'):
                light_hit_rate = self.agent.research_stats.get('light_hits', 0) / max(1, self.agent.research_stats.get('total_rays', 1))
                print(f"Agent light hit rate: {light_hit_rate*100:.1f}%")
        else:
            print(f"Model used: Heuristic only")

def main():
    """Main function"""
    print("="*80)
    print("IMPROVED FB RENDERER - Faster with Better Sun Finding")
    print("="*80)
    
    if not BASE_IMPORTS_OK or not FB_AVAILABLE:
        print("Error: Required modules not available")
        return
    
    # Path to your trained FB model
    model_path = "./fb_training_outputs/fb_your_scene_final.pth"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Warning: Model not found at {model_path}")
        print("Looking for alternative models...")
        
        # Try to find any fb_your_scene model
        import glob
        model_files = glob.glob("fb_your_scene_*/fb_your_scene_*.pth") + \
                     glob.glob("./fb_your_scene_*/fb_your_scene_*.pth")
        
        if model_files:
            model_path = sorted(model_files, key=lambda x: Path(x).stat().st_mtime, reverse=True)[0]
            print(f"Found model: {model_path}")
        else:
            print("No FB model found. Will use heuristic rendering.")
            model_path = None
    
    # Create renderer
    try:
        renderer = ImprovedFBRenderer(model_path)
    except Exception as e:
        print(f"Failed to create renderer: {e}")
        return
    
    # Rendering settings
    width = 400    # Image width
    height = 300   # Image height
    
    # Adjust parameters for better performance
    renderer.max_bounces = 6
    renderer.direct_sun_sampling_prob = 0.3  # 30% chance to try hitting sun directly
    
    print("\nRendering strategy:")
    print("1. Direct sun sampling for efficiency")
    print("2. FB-guided decisions when needed")
    print("3. Material-based reflections/refractions")
    print("4. Sun-biased diffuse sampling")
    
    # Render image
    try:
        image, output_path = renderer.render_image_fast(width, height)
        
        # Print statistics
        renderer.print_statistics()
        
        print(f"\nTips for better results:")
        print(f"1. To make it faster: Reduce image size (width={width//2}, height={height//2})")
        print(f"2. To make it brighter: Increase direct_sun_sampling_prob to 0.5")
        print(f"3. For more detail: Increase max_bounces to 8")
        print(f"4. Train FB model more: Run train_fb_custom_scene.py with more steps")
        
    except Exception as e:
        print(f"Error during rendering: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()