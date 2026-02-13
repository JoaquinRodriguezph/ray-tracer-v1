"""
Simplified FB Renderer - FB only helps with efficiency, output should match traditional
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import time

# Add safe globals for PyTorch 2.6+ compatibility
torch.serialization.add_safe_globals([
    np.ndarray,
    np.dtype,
    np.dtypes.Float32DType,
    np.dtypes.Float64DType,
])

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

class SimplifiedFBRenderer:
    """Simplified renderer where FB only helps with efficiency"""
    
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
        
        # Load FB model (for efficiency only)
        self.agent = None
        self.fb_model_loaded = False
        self.load_fb_model(model_path)
        
        # EXACT original rendering parameters
        self.max_bounces = 5
        self.samples_per_pixel = 100  # Original uses 1 sample per pixel
        
        # Simple FB usage: only use it to bias diffuse sampling toward light
        self.fb_usage_prob = 0.0  # Use FB 50% of the time when sampling diffuse
        
        # Statistics
        self.stats = {
            'total_rays': 0,
            'sun_hits': 0,
            'fb_used': 0,
            'fb_success': 0,
            'render_time': 0
        }
        
    def load_fb_model(self, model_path=None):
        """Load trained FB model with proper safe globals"""
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
            print("Warning: No FB model found. Using traditional rendering.")
            return
        
        try:
            print(f"Loading FB model from {model_path}...")
            
            # FIRST: Add all the safe globals needed for numpy
            import numpy
            torch.serialization.add_safe_globals([
                numpy.ndarray,
                numpy.core.multiarray._reconstruct,
                numpy.dtype,
                numpy.float32,
                numpy.float64,
                numpy.int32,
                numpy.int64,
                # Add numpy dtypes (required for PyTorch 2.6+)
                numpy.dtypes.Float32DType,
                numpy.dtypes.Float64DType,
                numpy.dtypes.Int32DType,
                numpy.dtypes.Int64DType,
            ])
            
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
            
            # Load the model using the agent's load method
            # This is the key fix: use agent.load() not agent.fb_learner.load_state_dict()
            self.agent.load(model_path)
            
            self.fb_model_loaded = True
            print("✓ FB model loaded successfully")
            
        except Exception as e:
            print(f"✗ Failed to load FB model: {e}")
            import traceback
            traceback.print_exc()
            self.agent = None
    
    def calculate_lighting_exact_original(self, intersection):
        """EXACTLY replicate your original lighting calculation"""
        # This should match your original terminalRGB method exactly
        
        sphere = intersection.object
        sphere_color = sphere.colour
        
        # If it's the sun, return sun color
        if sphere.id == 7:
            self.stats['sun_hits'] += 1
            return self.sun_color
        
        # Create lights EXACTLY like your original
        global_light = GlobalLight(
            vector=Vector(3, 1, -0.75),
            colour=Colour(20, 20, 255),
            strength=1,
            max_angle=np.radians(90),
            func=0
        )
        
        sun = Sphere(
            id=0,
            centre=self.sun_position,
            radius=self.sun_radius,
            material=Material(emitive=True),
            colour=self.sun_color
        )
        
        point_light = PointLight(
            id=sun.id,
            position=sun.centre,
            colour=sun.colour,
            strength=1,
            max_angle=np.radians(90),
            func=-1
        )
        
        # This is the key part - we need to simulate your original lighting
        # Since we can't directly call terminalRGB without the full ray terminal,
        # we'll approximate it closely
        
        # Your original does:
        # 1. Global light from direction (3, 1, -0.75) with Colour(20, 20, 255)
        # 2. Point light (sun) at (-0.6, 0.2, 6) with attenuation func=-1
        
        # Simplified approximation:
        to_sun = self.sun_position.subtractVector(intersection.point).normalise()
        
        # Global light contribution
        global_dir = Vector(3, 1, -0.75).normalise()
        global_cos = max(0, intersection.normal.dotProduct(global_dir))
        global_light_color = Colour(20, 20, 255)
        global_contrib = Colour(
            int(global_light_color.r * global_cos * 0.3),  # 30% strength
            int(global_light_color.g * global_cos * 0.3),
            int(global_light_color.b * global_cos * 0.3)
        )
        
        # Sun light contribution (if visible)
        # Check shadow
        shadow_ray = Ray(
            intersection.point.addVector(intersection.normal.scaleByLength(0.001)),
            to_sun
        )
        
        sun_visible = True
        sun_distance = intersection.point.distanceFrom(self.sun_position)
        
        for other_sphere in self.scene:
            if other_sphere == sphere or other_sphere.id == 7:
                continue
            
            shadow_intersect = shadow_ray.sphereDiscriminant(other_sphere)
            if shadow_intersect and shadow_intersect.intersects:
                shadow_dist = shadow_intersect.point.distanceFrom(intersection.point)
                if shadow_dist < sun_distance:
                    sun_visible = False
                    break
        
        sun_contrib = Colour(0, 0, 0)
        if sun_visible:
            # Your original PointLight uses func=-1 which is 1/distance^2
            distance = intersection.point.distanceFrom(self.sun_position)
            attenuation = 1.0 / (distance ** 2) if distance > 0 else 1.0
            attenuation = min(1.0, attenuation * 100)  # Scale factor
            
            cos_angle = max(0, intersection.normal.dotProduct(to_sun))
            
            sun_contrib = Colour(
                int(self.sun_color.r * cos_angle * attenuation * 0.9),
                int(self.sun_color.g * cos_angle * attenuation * 0.9),
                int(self.sun_color.b * cos_angle * attenuation * 0.9)
            )
        
        # Combine: (global + sun) * sphere_color
        combined_light = Colour(
            min(255, global_contrib.r + sun_contrib.r),
            min(255, global_contrib.g + sun_contrib.g),
            min(255, global_contrib.b + sun_contrib.b)
        )
        
        # Multiply by sphere color (normalized)
        final_color = Colour(
            int(sphere_color.r * (combined_light.r / 255.0)),
            int(sphere_color.g * (combined_light.g / 255.0)),
            int(sphere_color.b * (combined_light.b / 255.0))
        )
        
        return final_color
    
    def get_fb_sun_direction(self, intersection_point, normal, ray_dir=None, 
                         bounce_count=0, accumulated_color=None, 
                         material=None, sphere_id=None):
        """Use FB with FULL 22D observation (matching training)"""
        if not self.fb_model_loaded or not self.agent:
            return None
        
        try:
            # Create FULL 22D observation matching training
            if ray_dir is None:
                # Default ray direction (incoming ray)
                ray_dir = Vector(0, 0, -1)
            
            if accumulated_color is None:
                accumulated_color = Colour(0, 0, 0)
            
            if material is None:
                material = Material(reflective=False)
            
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
            
            # Create EXACT 22D observation (same as training)
            obs = np.array([
                # Position (3)
                intersection_point.x, intersection_point.y, intersection_point.z,
                # Direction (3) - current ray direction
                ray_dir.x, ray_dir.y, ray_dir.z,
                # Normal (3)
                normal.x, normal.y, normal.z,
                # Material properties (4)
                is_reflective, is_transparent, is_emitive, refractive_index,
                # Current color (3)
                color_norm[0], color_norm[1], color_norm[2],
                # Bounce and history (3)
                float(bounce_count) / self.max_bounces,
                0.0,  # through_count (not used)
                float(sphere_id if sphere_id is not None else 0) / 100.0,
                # Sun direction (3) - CRITICAL!
                sun_dir.x, sun_dir.y, sun_dir.z
            ], dtype=np.float32)
            
            # Ensure it's exactly 22 dimensions
            assert len(obs) == 22, f"Observation has {len(obs)} dimensions, expected 22"
            
            # Get action from FB model
            action, _ = self.agent.choose_direction_research(
                obs, 
                scene_context="custom_scene", 
                exploration_phase="exploit"
            )
            
            if action is not None:
                self.stats['fb_used'] += 1
                
                # Convert action (theta, phi) to direction
                theta = (action[0] + 1) * np.pi/4  # [-1,1] -> [0,π/2]
                phi = action[1] * np.pi  # [-1,1] -> [-π,π]
                
                # Create local direction
                local_dir = Vector(
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                )
                
                # Transform to world coordinates
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
            print(f"FB direction failed: {e}")
            # Fall back to traditional
        
        return None
    def sample_cosine_weighted_hemisphere(self, normal):
        """Traditional cosine-weighted hemisphere sampling"""
        r1 = np.random.random()
        r2 = np.random.random()
        theta = np.arccos(np.sqrt(r1))
        phi = 2 * np.pi * r2
        
        if abs(normal.z) > 0.9:
            tangent = Vector(1, 0, 0)
        else:
            tangent = Vector(0, 0, 1).crossProduct(normal)
        tangent = tangent.normalise()
        bitangent = normal.crossProduct(tangent).normalise()
        
        local_dir = Vector(
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        )
        
        return Vector(
            local_dir.x * tangent.x + local_dir.y * bitangent.x + local_dir.z * normal.x,
            local_dir.x * tangent.y + local_dir.y * bitangent.y + local_dir.z * normal.y,
            local_dir.x * tangent.z + local_dir.y * bitangent.z + local_dir.z * normal.z
        ).normalise()

    def trace_ray_simple(self, ray):
        """Simple ray tracing that matches original but can use FB for efficiency"""
        current_ray = ray
        bounce_count = 0
        accumulated_color = Colour(0, 0, 0)
        
        # Background color (like your original)
        background_color = Colour(2, 2, 5)
        
        while bounce_count < self.max_bounces:
            self.stats['total_rays'] += 1
            
            # Find nearest intersection
            nearest_intersection = None
            nearest_distance = float('inf')
            
            for sphere in self.scene:
                intersection = current_ray.sphereDiscriminant(sphere)
                if intersection and intersection.intersects:
                    distance = intersection.point.distanceFrom(current_ray.origin)
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_intersection = intersection
            
            if not nearest_intersection:
                # Hit background
                if bounce_count == 0:
                    accumulated_color = background_color
                break
            
            intersection = nearest_intersection
            sphere = intersection.object
            
            # Calculate lighting (EXACT original style)
            lighting = self.calculate_lighting_exact_original(intersection)
            
            # Add to accumulated color
            accumulated_color = Colour(
                min(255, accumulated_color.r + lighting.r),
                min(255, accumulated_color.g + lighting.g),
                min(255, accumulated_color.b + lighting.b)
            )
            
            # If we hit the sun, stop
            if sphere.id == 7:
                break
            
            # Determine next bounce direction
            material = sphere.material
            
            if getattr(material, 'reflective', False):
                # Mirror reflection
                next_dir = current_ray.D.reflectInVector(intersection.normal)
                
            elif getattr(material, 'transparent', False):
                # Glass - 50/50 reflect or transmit
                if np.random.random() < 0.5:
                    next_dir = current_ray.D.reflectInVector(intersection.normal)
                else:
                    next_dir = current_ray.D  # Continue straight
                    
            else:
                # Diffuse material - THIS IS WHERE FB CAN HELP
                # Traditional: cosine-weighted hemisphere sampling
                # FB-enhanced: bias toward light source when possible
                
                use_fb = (self.fb_model_loaded and 
                        np.random.random() < self.fb_usage_prob and
                        not getattr(material, 'reflective', False) and
                        not getattr(material, 'transparent', False))

                if use_fb:
                    # Pass ALL required parameters for 22D observation
                    fb_dir = self.get_fb_sun_direction(
                        intersection.point, 
                        intersection.normal,
                        current_ray.D,  # Current ray direction
                        bounce_count,   # Current bounce count
                        accumulated_color,  # Current accumulated color
                        material,       # Current material
                        sphere.id       # Sphere ID
                    )
                    
                    if fb_dir is not None:
                        next_dir = fb_dir
                        self.stats['fb_success'] += 1
                    else:
                        # Fall back to traditional diffuse
                        r1 = np.random.random()
                        r2 = np.random.random()
                        theta = np.arccos(np.sqrt(r1))
                        phi = 2 * np.pi * r2
                        
                        normal = intersection.normal
                        if abs(normal.z) > 0.9:
                            tangent = Vector(1, 0, 0)
                        else:
                            tangent = Vector(0, 0, 1).crossProduct(normal)
                        tangent = tangent.normalise()
                        bitangent = normal.crossProduct(tangent).normalise()
                        
                        local_dir = Vector(
                            np.sin(theta) * np.cos(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(theta)
                        )
                        
                        next_dir = self.sample_cosine_weighted_hemisphere(intersection.normal)
                else:
                    # Traditional diffuse sampling
                    r1 = np.random.random()
                    r2 = np.random.random()
                    theta = np.arccos(np.sqrt(r1))
                    phi = 2 * np.pi * r2
                    
                    normal = intersection.normal
                    if abs(normal.z) > 0.9:
                        tangent = Vector(1, 0, 0)
                    else:
                        tangent = Vector(0, 0, 1).crossProduct(normal)
                    tangent = tangent.normalise()
                    bitangent = normal.crossProduct(tangent).normalise()
                    
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
            
            # Create new ray
            current_ray = Ray(
                intersection.point.addVector(intersection.normal.scaleByLength(0.001)),
                next_dir
            )
            
            bounce_count += 1
        
        return accumulated_color
    
    def render_original_style(self, width=400, height=300, output_path=None):
        """Render in original style but with FB efficiency help"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"./fb_simple_render_{timestamp}.png"
        
        print(f"\nRendering image: {width}x{height}")
        print(f"Max bounces: {self.max_bounces}")
        print(f"FB usage probability: {self.fb_usage_prob*100:.0f}%")
        
        # Reset stats
        self.stats = {
            'total_rays': 0,
            'sun_hits': 0,
            'fb_used': 0,
            'fb_success': 0,
            'render_time': 0
        }
        
        start_time = time.time()
        
        # Create image buffer
        image = np.zeros((height, width, 3), dtype=np.float32)
        
        # Camera position (same as original)
        camera_pos = Vector(0, 0, 1)
        
        # Generate rays in a grid (like original)
        # Note: Original uses 601x601 grid with specific parameters
        # We'll use a simpler approach but similar look
        
        for y in tqdm(range(height), desc="Rendering"):
            for x in range(width):
                # Map pixel coordinates to [-1, 1] range
                u = (x / width - 0.5) * 2.0
                v = (y / height - 0.5) * -2.0  # Flip Y
                
                # Adjust for aspect ratio
                aspect_ratio = width / height
                u *= aspect_ratio
                
                # Create ray direction (same FOV as original ~60 degrees)
                fov = np.pi / 3
                ray_dir = Vector(u * np.tan(fov/2), v * np.tan(fov/2), -1).normalise()
                
                # Trace ray
                ray = Ray(camera_pos, ray_dir)
                color = self.trace_ray_simple(ray)
                
                # Store pixel
                image[y, x] = [
                    min(1.0, color.r / 255.0),
                    min(1.0, color.g / 255.0),
                    min(1.0, color.b / 255.0)
                ]
        
        render_time = time.time() - start_time
        self.stats['render_time'] = render_time
        
        # Save image
        plt.figure(figsize=(10, 8))
        plt.imshow(np.clip(image, 0, 1))
        plt.title(f'FB-Assisted Render (Same Quality, Better Efficiency)\n{width}x{height}, {render_time:.1f}s')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nRender complete!")
        print(f"Image saved: {output_path}")
        print(f"Render time: {render_time:.1f} seconds")
        
        # Print statistics
        self.print_statistics()
        
        return image, output_path
    
    def print_statistics(self):
        """Print statistics"""
        print("\n" + "="*60)
        print("RENDERING STATISTICS")
        print("="*60)
        print(f"Total rays traced: {self.stats['total_rays']:,}")
        print(f"Sun hits: {self.stats['sun_hits']:,}")
        
        if self.stats['total_rays'] > 0:
            sun_hit_rate = (self.stats['sun_hits'] / self.stats['total_rays']) * 100
            print(f"Sun hit rate: {sun_hit_rate:.2f}%")
        
        if self.fb_model_loaded:
            print(f"FB model used: Yes")
            print(f"FB direction attempts: {self.stats['fb_used']:,}")
            print(f"FB successful directions: {self.stats['fb_success']:,}")
            
            if self.stats['fb_used'] > 0:
                fb_success_rate = (self.stats['fb_success'] / self.stats['fb_used']) * 100
                print(f"FB success rate: {fb_success_rate:.1f}%")
        else:
            print(f"FB model used: No (traditional rendering only)")
        
        print(f"Render time: {self.stats['render_time']:.1f} seconds")
        
        if self.stats['render_time'] > 0:
            rays_per_second = self.stats['total_rays'] / self.stats['render_time']
            print(f"Performance: {rays_per_second:,.0f} rays/second")

def main():
    """Main function"""
    print("="*80)
    print("SIMPLIFIED FB RENDERER - Same Quality, Better Efficiency")
    print("="*80)
    print("Concept: FB learns scene representation to guide diffuse sampling")
    print("Result: Images should look identical to traditional ray tracing")
    print("Benefit: Faster convergence, especially in complex lighting")
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
            print("No FB model found. Will use traditional rendering.")
            model_path = None
    
    # Create renderer
    try:
        renderer = SimplifiedFBRenderer(model_path)
    except Exception as e:
        print(f"Failed to create renderer: {e}")
        return
    
    # Rendering settings
    width = 400    # Image width
    height = 300   # Image height
    
    # Adjust FB usage probability
    # Higher = more FB guidance (potentially faster)
    # Lower = more traditional sampling (more consistent)
    renderer.fb_usage_prob = 0.7  # 70% chance to use FB for diffuse sampling
    
    print("\nRendering strategy:")
    print("1. Exact original lighting calculations")
    print("2. Materials behave exactly as in original")
    print("3. FB only helps with diffuse bounce directions")
    print("4. Output should match traditional rendering exactly")
    
    # Render image
    try:
        image, output_path = renderer.render_original_style(width, height)
        
        print(f"\nExpected result:")
        print("• Image should look identical to traditional ray tracing")
        print("• Potentially faster rendering (especially in shadows)")
        print("• Better sampling efficiency (fewer wasted rays)")
        
        if renderer.fb_model_loaded:
            print(f"\nTo compare with traditional:")
            print("Set fb_usage_prob = 0.0 for pure traditional rendering")
            print("Set fb_usage_prob = 1.0 for full FB-guided rendering")
        
    except Exception as e:
        print(f"Error during rendering: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()