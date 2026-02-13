import numpy as np
from stable_baselines3 import SAC
from ray_tracer_env import RayTracerEnv
from vector import Vector
from colour import Colour
from object import Sphere
from material import Material

class RayTracerRL:
    def __init__(self, model_path="raytracer_final"):
        # Load RL model
        try:
            self.rl_model = SAC.load(model_path)
            self.use_rl = True
            print(f"✓ RL model loaded from {model_path}")
        except:
            self.rl_model = None
            self.use_rl = False
            print("⚠️ RL model not found, using traditional sampling")
        
        # Create test scene (same as in use_trained_model.py)
        self.create_test_scene()
    
    def create_test_scene(self):
        """Create the same scene you trained on"""
        matte = Material(reflective=0, transparent=0, emitive=0.1, refractive_index=1)
        reflective = Material(reflective=1, transparent=0, emitive=0, refractive_index=1)
        light_mat = Material(reflective=0, transparent=0, emitive=1, refractive_index=1)
        
        # Adjusted positions - move everything further from camera
        self.spheres = [
            # Ground (much larger to act as floor)
            Sphere(Vector(0, -5, -15), 10, matte, Colour(180, 180, 180), id=1),
            # Center sphere (reflective)
            Sphere(Vector(0, 0, -10), 1.5, reflective, Colour(255, 255, 255), id=2),
            # Left sphere
            Sphere(Vector(-3, 0.5, -10), 1.2, reflective, Colour(180, 180, 255), id=3),
            # Right sphere
            Sphere(Vector(3, -0.3, -10), 1.2, reflective, Colour(255, 180, 180), id=4),
            # Lights - positioned above the scene
            Sphere(Vector(0, 6, -8), 1.0, light_mat, Colour(255, 255, 200), id=99),
            Sphere(Vector(-4, 5, -8), 0.8, light_mat, Colour(200, 255, 200), id=100),
        ]
    
    def _create_observation(self, intersection, ray, bounce_count):
        """Create observation for RL model"""
        pos = intersection.point
        normal = intersection.normal
        material = intersection.object.material
        
        obs = np.array([
            pos.x, pos.y, pos.z,
            ray.D.x, ray.D.y, ray.D.z,
            normal.x, normal.y, normal.z,
            material.reflective,
            material.transparent,
            material.emitive,
            material.refractive_index,
            0, 0, 0,  # Accumulated color
            float(bounce_count),
            0.0  # Through count
        ], dtype=np.float32)
        
        return obs
    
    def _action_to_direction(self, action, normal):
        """Convert RL action [theta, phi] to 3D direction"""
        theta, phi = action[0], action[1]
        
        # Convert spherical to Cartesian (local space)
        local_x = np.sin(theta) * np.cos(phi)
        local_y = np.sin(theta) * np.sin(phi)
        local_z = np.cos(theta)
        
        # Create coordinate frame around normal
        if abs(normal.z) < 0.9:
            tangent = Vector(0, 0, 1).crossProduct(normal).normalise()
        else:
            tangent = Vector(1, 0, 0).crossProduct(normal).normalise()
        
        bitangent = normal.crossProduct(tangent).normalise()
        
        # Transform to world space
        world_dir = Vector(
            local_x * tangent.x + local_y * bitangent.x + local_z * normal.x,
            local_x * tangent.y + local_y * bitangent.y + local_z * normal.y,
            local_x * tangent.z + local_y * bitangent.z + local_z * normal.z
        )
        
        return world_dir.normalise()
    
    def _sample_hemisphere_traditional(self, normal):
        """Traditional random sampling"""
        # Uniform sampling on hemisphere
        theta = np.random.uniform(0, np.pi/2)
        phi = np.random.uniform(0, 2*np.pi)
        
        # Convert to direction
        local_x = np.sin(theta) * np.cos(phi)
        local_y = np.sin(theta) * np.sin(phi)
        local_z = np.cos(theta)
        
        # Create coordinate frame
        if abs(normal.z) < 0.9:
            tangent = Vector(0, 0, 1).crossProduct(normal).normalise()
        else:
            tangent = Vector(1, 0, 0).crossProduct(normal).normalise()
        
        bitangent = normal.crossProduct(tangent).normalise()
        
        world_dir = Vector(
            local_x * tangent.x + local_y * bitangent.x + local_z * normal.x,
            local_x * tangent.y + local_y * bitangent.y + local_z * normal.y,
            local_x * tangent.z + local_y * bitangent.z + local_z * normal.z
        )
        
        return world_dir.normalise()
    
    def trace_ray(self, ray, use_rl=True, max_bounces=5, samples_per_pixel=1):
        """Trace a ray with optional RL guidance"""
        from ray import Ray as RayClass
        
        total_color = Colour(0, 0, 0)
        
        # Multiple samples per pixel for better quality
        for _ in range(samples_per_pixel):
            color = Colour(0, 0, 0)
            current_ray = ray
            bounce_count = 0
            
            while bounce_count < max_bounces:
                # Find intersection
                intersection = current_ray.nearestSphereIntersect(
                    spheres=self.spheres,
                    max_bounces=max_bounces
                )
                
                if not intersection or not intersection.intersects:
                    break
                
                # Get object color
                obj = intersection.object
                obj_color = obj.colour
                
                # Simple lighting calculation
                # Light 1 direction
                light1_pos = Vector(0, 6, -8)
                light1_dir = light1_pos.subtractVector(intersection.point).normalise()
                
                # Light 2 direction  
                light2_pos = Vector(-4, 5, -8)
                light2_dir = light2_pos.subtractVector(intersection.point).normalise()
                
                # Check if light is visible
                def is_light_visible(light_pos):
                    to_light = light_pos.subtractVector(intersection.point)
                    to_light_dist = to_light.magnitude()
                    to_light_dir = to_light.normalise()
                    
                    shadow_ray = RayClass(intersection.point.addVector(intersection.normal.scaleByLength(0.001)), to_light_dir)
                    shadow_test = shadow_ray.nearestSphereIntersect(
                        spheres=self.spheres,
                        suppress_ids=[obj.id],
                        max_bounces=1
                    )
                    
                    if shadow_test and shadow_test.intersects and shadow_test.distance < to_light_dist:
                        return False  # In shadow
                    return True  # Visible
                
                # Calculate lighting
                ambient = 0.1
                diffuse = 0
                
                if is_light_visible(light1_pos):
                    dot1 = max(0, intersection.normal.dotProduct(light1_dir))
                    diffuse += 0.4 * dot1
                
                if is_light_visible(light2_pos):
                    dot2 = max(0, intersection.normal.dotProduct(light2_dir))
                    diffuse += 0.3 * dot2
                
                # Emissive materials (lights)
                if obj.material.emitive > 0:
                    brightness = 1.0
                else:
                    brightness = ambient + diffuse
                
                # Scale color by brightness
                scaled_color = obj_color.scaleRGB(brightness, return_type='Colour')
                color = color.addColour(scaled_color)
                
                # Choose next direction (for next bounce)
                if use_rl and self.rl_model is not None and bounce_count >= 0:
                    # RL-guided sampling
                    obs = self._create_observation(intersection, current_ray, bounce_count)
                    action, _ = self.rl_model.predict(obs, deterministic=True)
                    new_direction = self._action_to_direction(action, intersection.normal)
                else:
                    # Traditional random sampling
                    new_direction = self._sample_hemisphere_traditional(intersection.normal)
                
                # Create new ray for next bounce
                current_ray = RayClass(intersection.point.addVector(intersection.normal.scaleByLength(0.001)), new_direction)
                bounce_count += 1
            
            total_color = total_color.addColour(color)
        
        # Average multiple samples
        if samples_per_pixel > 1:
            avg_r = total_color.r / samples_per_pixel
            avg_g = total_color.g / samples_per_pixel
            avg_b = total_color.b / samples_per_pixel
            total_color = Colour(int(avg_r), int(avg_g), int(avg_b))
        
        return total_color
    
    def color_to_rgb(self, color):
        """Safely convert Colour to RGB tuple with integers"""
        if not hasattr(color, 'r'):
            return (0, 0, 0)
        
        r = int(round(min(255, max(0, color.r))))
        g = int(round(min(255, max(0, color.g))))
        b = int(round(min(255, max(0, color.b))))
        return (r, g, b)
    
    def render_comparison(self, width=400, height=300):
        """Render side-by-side comparison: RL vs Traditional"""
        try:
            from PIL import Image
        except ImportError:
            print("Install PIL: pip install pillow")
            return
        
        # Create two images side-by-side
        comparison = Image.new('RGB', (width*2, height))
        pixels_rl = Image.new('RGB', (width, height))
        pixels_trad = Image.new('RGB', (width, height))
        
        print(f"Rendering comparison {width}x{height}...")
        print("Left: RL-guided | Right: Traditional")
        print("Scene: 3 spheres on a ground plane with 2 lights above")
        
        # Camera setup - further back
        camera_pos = Vector(0, 2, 5)  # Camera at (0, 2, 5) looking toward (0, 0, -10)
        
        for y in range(height):
            for x in range(width):
                # Convert pixel coordinates to ray direction
                # Simple perspective projection
                aspect_ratio = width / height
                ndc_x = (x + 0.5) / width
                ndc_y = (y + 0.5) / height
                
                # Convert to screen space
                screen_x = (2.0 * ndc_x - 1.0) * aspect_ratio
                screen_y = 1.0 - 2.0 * ndc_y  # Flip y
                
                # Create ray from camera through pixel
                ray_origin = camera_pos
                # Ray direction points toward scene
                ray_dir = Vector(screen_x * 0.5, screen_y * 0.5, -1).normalise()
                
                from ray import Ray
                ray = Ray(ray_origin, ray_dir)
                
                # Render with RL
                rl_color = self.trace_ray(ray, use_rl=True, max_bounces=3, samples_per_pixel=1)
                # Render traditional
                trad_color = self.trace_ray(ray, use_rl=False, max_bounces=3, samples_per_pixel=1)
                
                # Convert to RGB tuples
                rl_rgb = self.color_to_rgb(rl_color)
                trad_rgb = self.color_to_rgb(trad_color)
                
                pixels_rl.putpixel((x, y), rl_rgb)
                pixels_trad.putpixel((x, y), trad_rgb)
            
            if y % 30 == 0:
                print(f"  Progress: {y}/{height} rows ({y/height*100:.0f}%)")
        
        # Combine images
        comparison.paste(pixels_rl, (0, 0))
        comparison.paste(pixels_trad, (width, 0))
        
        # Add labels
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(comparison)
            
            # Simple labels
            draw.text((10, 10), "RL-Guided", fill=(255, 255, 255))
            draw.text((width + 10, 10), "Traditional", fill=(255, 255, 255))
        except:
            pass  # Labels are optional
        
        # Save
        comparison.save('rl_vs_traditional_comparison.png')
        print("✓ Comparison saved as 'rl_vs_traditional_comparison.png'")
        
        # Also save individual images
        pixels_rl.save('rl_only_render.png')
        pixels_trad.save('traditional_only_render.png')
        print("✓ Individual renders saved as 'rl_only_render.png' and 'traditional_only_render.png'")
        
        comparison.show()


# Test it
    def render_single_view(self, use_rl=True, width=400, height=300, filename="render.png"):
        """Render a single view (RL or traditional)"""
        try:
            from PIL import Image
        except ImportError:
            print("Install PIL: pip install pillow")
            return
        
        image = Image.new('RGB', (width, height))
        
        method = "RL-guided" if use_rl else "Traditional"
        print(f"Rendering {method} view {width}x{height}...")
        
        # Camera setup
        camera_pos = Vector(0, 2, 5)
        
        for y in range(height):
            for x in range(width):
                # Convert pixel coordinates to ray direction
                aspect_ratio = width / height
                ndc_x = (x + 0.5) / width
                ndc_y = (y + 0.5) / height
                
                screen_x = (2.0 * ndc_x - 1.0) * aspect_ratio
                screen_y = 1.0 - 2.0 * ndc_y
                
                ray_origin = camera_pos
                ray_dir = Vector(screen_x * 0.5, screen_y * 0.5, -1).normalise()
                
                from ray import Ray
                ray = Ray(ray_origin, ray_dir)
                
                color = self.trace_ray(ray, use_rl=use_rl, max_bounces=4, samples_per_pixel=4)
                rgb = self.color_to_rgb(color)
                image.putpixel((x, y), rgb)
            
            if y % 30 == 0:
                print(f"  Progress: {y}/{height} rows ({y/height*100:.0f}%)")
        
        image.save(filename)
        print(f"✓ {method} render saved as '{filename}'")
        image.show()

# Test both individually
if __name__ == "__main__":
    tracer = RayTracerRL("raytracer_final")
    
    # Render comparison
    tracer.render_comparison(width=400, height=300)
    
    # Also render individual high-quality versions
    print("\n" + "="*60)
    print("Rendering individual high-quality views...")
    print("="*60)
    
    tracer.render_single_view(use_rl=True, width=400, height=300, filename="rl_high_quality.png")
    tracer.render_single_view(use_rl=False, width=400, height=300, filename="traditional_high_quality.png")