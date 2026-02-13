"""
Traditional Ray Tracing Test for Complex Scene
Tests traditional (non-FB) ray tracing on the complex scene to establish baseline performance
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime

# Import scene modules
from vector import Vector
from colour import Colour
from object import Sphere
from material import Material
from ray import Ray

# Import complex scene
from complex_scene import create_complex_scene, create_camera_for_scene, create_lights_for_scene
from light import GlobalLight, PointLight

class TraditionalComplexSceneRenderer:
    """Traditional ray tracer for the complex scene"""
    
    def __init__(self):
        # Load complex scene
        self.scene = create_complex_scene()
        self.camera_position, self.camera_angle = create_camera_for_scene()
        self.global_lights, self.point_lights = create_lights_for_scene()
        
        # Render settings
        self.image_width = 800
        self.image_height = 600
        self.max_bounces = 5
        self.samples_per_pixel = 16  # Start with low samples for testing
        
        # Camera settings
        self.fov = 60  # degrees
        self.aspect_ratio = self.image_width / self.image_height
        
        # Statistics
        self.stats = {
            'total_rays': 0,
            'total_intersections': 0,
            'light_hits': 0,
            'render_time': 0,
            'rays_per_second': 0
        }
        
        # Cache for emitive spheres (lights)
        self.light_sources = [s for s in self.scene if s.material.emitive]
        
        print(f"Traditional Ray Tracer initialized:")
        print(f"  Scene objects: {len(self.scene)}")
        print(f"  Light sources: {len(self.light_sources)}")
        print(f"  Image size: {self.image_width}x{self.image_height}")
        print(f"  Max bounces: {self.max_bounces}")
        print(f"  Samples per pixel: {self.samples_per_pixel}")
    
    def generate_camera_ray(self, pixel_x, pixel_y, sample_x=0.5, sample_y=0.5):
        """Generate camera ray for a pixel with jittered sampling"""
        # Convert pixel coordinates to NDC [0, 1]
        ndc_x = (pixel_x + sample_x) / self.image_width
        ndc_y = (pixel_y + sample_y) / self.image_height
        
        # Convert to screen space [-1, 1]
        screen_x = 2.0 * ndc_x - 1.0
        screen_y = 1.0 - 2.0 * ndc_y  # Flip Y
        
        # Apply aspect ratio
        screen_x *= self.aspect_ratio
        
        # Apply FOV
        fov_rad = np.radians(self.fov)
        half_height = np.tan(fov_rad / 2)
        half_width = half_height * self.aspect_ratio
        
        screen_x *= half_width
        screen_y *= half_height
        
        # Create ray direction
        ray_dir = Vector(screen_x, screen_y, -1).normalise()
        
        return Ray(self.camera_position, ray_dir)
    
    def trace_ray_traditional(self, ray, bounce_count=0):
        """Traditional ray tracing with Monte Carlo sampling"""
        self.stats['total_rays'] += 1
        
        # Base case: max bounces reached or ray escaped
        if bounce_count >= self.max_bounces:
            return Colour(2, 2, 5)  # Background color
        
        # Find nearest intersection
        nearest_intersection = None
        nearest_distance = float('inf')
        
        for sphere in self.scene:
            intersection = ray.sphereDiscriminant(sphere)
            if intersection and intersection.intersects:
                # Calculate distance from ray origin
                distance = intersection.point.distanceFrom(ray.origin)
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_intersection = intersection
        
        # If no intersection, return background
        if not nearest_intersection:
            return Colour(2, 2, 5)
        
        self.stats['total_intersections'] += 1
        
        # Get intersection data
        intersection = nearest_intersection
        sphere = intersection.object
        point = intersection.point
        normal = intersection.normal
        material = sphere.material
        
        # Check if we hit a light source
        if material.emitive:
            self.stats['light_hits'] += 1
            return sphere.colour  # Direct light hit
        
        # ===== TRADITIONAL LIGHTING CALCULATION =====
        # This replicates standard Monte Carlo ray tracing
        
        # 1. Direct lighting from global lights
        direct_light = Colour(0, 0, 0)
        
        for global_light in self.global_lights:
            # Global lights are always visible (directional)
            cos_angle = max(0, normal.dotProduct(global_light.vector.normalise()))
            if cos_angle > 0:
                # Get light color based on angle
                angle = normal.angleBetween(global_light.vector.normalise())
                light_color = global_light.relativeStrength(angle)
                direct_light = direct_light.addColour(light_color)
        
        # 2. Direct lighting from point lights (our emitive spheres)
        for light_sphere in self.light_sources:
            if light_sphere == sphere:
                continue  # Skip self
            
            # Vector to light
            to_light = light_sphere.centre.subtractVector(point)
            distance_to_light = to_light.magnitude()
            to_light_normalized = to_light.normalise()
            
            # Check if light is above surface
            cos_angle = max(0, normal.dotProduct(to_light_normalized))
            if cos_angle <= 0:
                continue
            
            # Shadow check - shoot shadow ray
            shadow_ray = Ray(
                point.addVector(normal.scaleByLength(0.001)),  # Offset to avoid self-intersection
                to_light_normalized
            )
            
            in_shadow = False
            for other_sphere in self.scene:
                if other_sphere == sphere or other_sphere == light_sphere:
                    continue
                
                shadow_intersect = shadow_ray.sphereDiscriminant(other_sphere)
                if shadow_intersect and shadow_intersect.intersects:
                    shadow_distance = shadow_intersect.point.distanceFrom(point)
                    if shadow_distance < distance_to_light:
                        in_shadow = True
                        break
            
            if not in_shadow:
                # Light is visible - calculate contribution
                # Solid angle approximation
                light_radius = light_sphere.radius
                solid_angle = 2 * np.pi * (1 - np.sqrt(1 - (light_radius/distance_to_light)**2))
                
                # Inverse square law
                attenuation = 1.0 / (distance_to_light ** 2)
                attenuation = min(1.0, attenuation * 100)  # Scale factor
                
                # Light contribution
                light_contrib = Colour(
                    int(light_sphere.colour.r * cos_angle * solid_angle * attenuation * 0.5),
                    int(light_sphere.colour.g * cos_angle * solid_angle * attenuation * 0.5),
                    int(light_sphere.colour.b * cos_angle * solid_angle * attenuation * 0.5)
                )
                direct_light = direct_light.addColour(light_contrib)
        
        # 3. Indirect lighting (next bounce)
        indirect_light = Colour(0, 0, 0)
        
        # Choose next bounce direction based on material
        if material.reflective > 0:
            # Mirror reflection - deterministic
            reflect_dir = ray.D.reflectInVector(normal)
            reflect_ray = Ray(point.addVector(normal.scaleByLength(0.001)), reflect_dir)
            indirect_light = self.trace_ray_traditional(reflect_ray, bounce_count + 1)
            
            # Blend based on reflectivity
            indirect_light = indirect_light.scaleRGB(material.reflective)
            
        elif material.transparent > 0:
            # Glass - 50% reflection, 50% transmission
            if np.random.random() < 0.5:
                # Reflection
                reflect_dir = ray.D.reflectInVector(normal)
                reflect_ray = Ray(point.addVector(normal.scaleByLength(0.001)), reflect_dir)
                indirect_reflect = self.trace_ray_traditional(reflect_ray, bounce_count + 1)
                indirect_light = indirect_light.addColour(indirect_reflect.scaleRGB(0.5))
            else:
                # Transmission (continue straight, simplified)
                transmit_ray = Ray(point.addVector(normal.scaleByLength(0.001)), ray.D)
                indirect_transmit = self.trace_ray_traditional(transmit_ray, bounce_count + 1)
                indirect_light = indirect_light.addColour(indirect_transmit.scaleRGB(0.5))
                
        else:
            # Diffuse surface - Monte Carlo hemisphere sampling
            # Cosine-weighted importance sampling
            
            # Generate random direction on hemisphere
            r1 = np.random.random()
            r2 = np.random.random()
            
            theta = np.arccos(np.sqrt(r1))
            phi = 2 * np.pi * r2
            
            # Create local coordinate system around normal
            if abs(normal.z) > 0.9:
                tangent = Vector(1, 0, 0)
            else:
                tangent = Vector(0, 0, 1).crossProduct(normal)
            tangent = tangent.normalise()
            bitangent = normal.crossProduct(tangent).normalise()
            
            # Convert spherical to cartesian
            local_dir = Vector(
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            )
            
            # Transform to world coordinates
            bounce_dir = Vector(
                local_dir.x * tangent.x + local_dir.y * bitangent.x + local_dir.z * normal.x,
                local_dir.x * tangent.y + local_dir.y * bitangent.y + local_dir.z * normal.y,
                local_dir.x * tangent.z + local_dir.y * bitangent.z + local_dir.z * normal.z
            ).normalise()
            
            # Cast bounce ray
            bounce_ray = Ray(point.addVector(normal.scaleByLength(0.001)), bounce_dir)
            indirect_sample = self.trace_ray_traditional(bounce_ray, bounce_count + 1)
            
            # Cosine-weighted importance sampling correction
            pdf = np.cos(theta) / np.pi  # PDF for cosine-weighted sampling
            if pdf > 0:
                indirect_sample = indirect_sample.scaleRGB(1.0 / pdf)
            
            indirect_light = indirect_sample
        
        # 4. Combine lighting
        # Object color * (direct + indirect)
        object_color = sphere.colour
        
        # Normalize colors for multiplication
        direct_norm = Colour(
            min(255, direct_light.r),
            min(255, direct_light.g),
            min(255, direct_light.b)
        )
        
        indirect_norm = Colour(
            min(255, indirect_light.r),
            min(255, indirect_light.g),
            min(255, indirect_light.b)
        )
        
        # Combine: object_color * (direct + indirect)
        total_light = Colour(
            min(255, direct_norm.r + indirect_norm.r),
            min(255, direct_norm.g + indirect_norm.g),
            min(255, direct_norm.b + indirect_norm.b)
        )
        
        final_color = Colour(
            int(object_color.r * (total_light.r / 255.0)),
            int(object_color.g * (total_light.g / 255.0)),
            int(object_color.b * (total_light.b / 255.0))
        )
        
        return final_color
    
    def render_scene(self, samples_per_pixel=None):
        """Render the complex scene with traditional ray tracing"""
        if samples_per_pixel:
            self.samples_per_pixel = samples_per_pixel
        
        print(f"\nStarting traditional ray tracing render...")
        print(f"Samples per pixel: {self.samples_per_pixel}")
        print(f"Expected total rays: ~{self.image_width * self.image_height * self.samples_per_pixel * (self.max_bounces + 1):,}")
        
        # Reset stats
        self.stats = {
            'total_rays': 0,
            'total_intersections': 0,
            'light_hits': 0,
            'render_time': 0,
            'rays_per_second': 0
        }
        
        start_time = time.time()
        
        # Create image buffer
        image = np.zeros((self.image_height, self.image_width, 3), dtype=np.float32)
        
        # Progress bar
        pbar = tqdm(total=self.image_height * self.image_width, desc="Rendering")
        
        # Render each pixel
        for y in range(self.image_height):
            for x in range(self.image_width):
                pixel_color = Colour(0, 0, 0)
                
                # Monte Carlo sampling
                for sample in range(self.samples_per_pixel):
                    # Jitter for anti-aliasing
                    jitter_x = np.random.random() - 0.5
                    jitter_y = np.random.random() - 0.5
                    
                    # Generate camera ray
                    ray = self.generate_camera_ray(x, y, 0.5 + jitter_x, 0.5 + jitter_y)
                    
                    # Trace ray
                    sample_color = self.trace_ray_traditional(ray)
                    pixel_color = pixel_color.addColour(sample_color)
                
                # Average samples
                pixel_color = Colour(
                    pixel_color.r // self.samples_per_pixel,
                    pixel_color.g // self.samples_per_pixel,
                    pixel_color.b // self.samples_per_pixel
                )
                
                # Store in image (normalize to [0, 1])
                image[y, x] = [
                    min(1.0, pixel_color.r / 255.0),
                    min(1.0, pixel_color.g / 255.0),
                    min(1.0, pixel_color.b / 255.0)
                ]
                
                pbar.update(1)
        
        pbar.close()
        
        # Calculate statistics
        render_time = time.time() - start_time
        self.stats['render_time'] = render_time
        
        if render_time > 0:
            self.stats['rays_per_second'] = self.stats['total_rays'] / render_time
        
        return image
    
    def save_render(self, image, output_path=None):
        """Save rendered image and statistics"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("./traditional_renders")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"traditional_complex_{timestamp}.png"
        
        # Create figure with statistics overlay
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Render image
        ax1.imshow(np.clip(image, 0, 1))
        ax1.set_title(f'Traditional Ray Tracing - Complex Scene\n{self.image_width}x{self.image_height}, {self.samples_per_pixel} spp')
        ax1.axis('off')
        
        # Statistics text
        stats_text = f"""Traditional Ray Tracing Statistics:
        
        Render Settings:
        Image Size: {self.image_width}x{self.image_height}
        Samples per Pixel: {self.samples_per_pixel}
        Max Bounces: {self.max_bounces}
        
        Performance:
        Render Time: {self.stats['render_time']:.1f} seconds
        Total Rays: {self.stats['total_rays']:,}
        Rays per Second: {self.stats['rays_per_second']:,.0f}
        
        Scene Complexity:
        Scene Objects: {len(self.scene)}
        Light Sources: {len(self.light_sources)}
        Total Intersections: {self.stats['total_intersections']:,}
        Direct Light Hits: {self.stats['light_hits']:,}
        
        Light Hit Rate: {(self.stats['light_hits']/max(1, self.stats['total_rays'])*100):.4f}%
        
        Expected FB Advantage:
        • Small lights (0.08-0.15 radius) are hard to hit randomly
        • Mirror sphere causes many bounce rays
        • Glass spheres require precise sampling
        • Traditional MC has high variance in this scene
        """
        
        ax2.text(0.1, 0.95, stats_text, fontfamily='monospace', fontsize=10,
                verticalalignment='top', transform=ax2.transAxes)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save statistics to file
        stats_path = str(output_path).replace('.png', '_stats.txt')
        with open(stats_path, 'w') as f:
            f.write(stats_text)
        
        print(f"\nRender saved: {output_path}")
        print(f"Statistics saved: {stats_path}")
        
        return str(output_path), stats_text
    
    def run_performance_test(self, spp_values=[1, 4, 16, 64]):
        """Run performance test with different samples per pixel"""
        print("\n" + "="*80)
        print("TRADITIONAL RAY TRACING PERFORMANCE TEST")
        print("="*80)
        
        results = []
        
        for spp in spp_values:
            print(f"\nTesting with {spp} samples per pixel...")
            
            # Render with current spp
            image = self.render_scene(samples_per_pixel=spp)
            
            # Save render
            timestamp = datetime.now().strftime("%H%M%S")
            output_path = f"./traditional_renders/complex_spp_{spp}_{timestamp}.png"
            self.save_render(image, output_path)
            
            # Record results
            results.append({
                'spp': spp,
                'render_time': self.stats['render_time'],
                'total_rays': self.stats['total_rays'],
                'rays_per_second': self.stats['rays_per_second'],
                'light_hits': self.stats['light_hits'],
                'hit_rate': self.stats['light_hits'] / max(1, self.stats['total_rays'])
            })
            
            # Print summary
            print(f"  Render time: {self.stats['render_time']:.1f}s")
            print(f"  Total rays: {self.stats['total_rays']:,}")
            print(f"  Rays/sec: {self.stats['rays_per_second']:,.0f}")
            print(f"  Light hit rate: {results[-1]['hit_rate']*100:.4f}%")
        
        # Print comparison table
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON")
        print("="*80)
        print(f"{'SPP':>6} {'Time (s)':>10} {'Rays (M)':>10} {'Rays/s (M)':>12} {'Hit Rate %':>12}")
        print("-"*60)
        
        for r in results:
            print(f"{r['spp']:>6} {r['render_time']:>10.1f} {r['total_rays']/1e6:>10.1f} "
                  f"{r['rays_per_second']/1e6:>12.1f} {r['hit_rate']*100:>12.4f}")
        
        return results

def main():
    """Main function to test traditional ray tracing on complex scene"""
    print("="*80)
    print("TRADITIONAL RAY TRACING - COMPLEX SCENE TEST")
    print("="*80)
    print("Testing baseline performance before FB optimization")
    print("Scene features small, occluded lights that are difficult for MC sampling")
    print("="*80)
    
    # Create renderer
    renderer = TraditionalComplexSceneRenderer()
    
    # Quick test render (low samples for speed)
    print("\nRunning quick test render (4 spp) to verify scene...")
    image = renderer.render_scene(samples_per_pixel=4)
    output_path, stats = renderer.save_render(image)
    
    # Run performance test
    print("\n" + "="*80)
    print("STARTING PERFORMANCE TEST")
    print("="*80)
    
    results = renderer.run_performance_test(spp_values=[1, 2, 4, 8, 16])
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS & PREDICTIONS FOR FB")
    print("="*80)
    
    # Calculate expected FB improvement
    best_hit_rate = max(r['hit_rate'] for r in results)
    
    print(f"\nCurrent best light hit rate: {best_hit_rate*100:.4f}%")
    print(f"Traditional rays needed per light hit: ~{1/best_hit_rate:,.0f}")
    
    # Predict FB performance
    print("\nExpected FB performance (educated guess):")
    print("• FB should learn to aim at lights through reflections")
    print(f"• Could improve hit rate to ~{min(5.0, best_hit_rate*100):.1f}% (5-50x improvement)")
    print(f"• Equivalent to ~{1/0.05:,.0f} rays per hit (vs current {1/best_hit_rate:,.0f})")
    print(f"• Expected speedup: {1/best_hit_rate/20:,.1f}x to {1/best_hit_rate/5:,.1f}x")
    
    print("\nKey areas where FB should excel:")
    print("1. Hitting tiny lights (0.08 radius = ~0.0001% random hit chance)")
    print("2. Navigating through mirror reflections to lights")
    print("3. Finding light paths around occluding objects")
    print("4. Reducing noise in shadow regions")
    
    # Save final report
    report_path = "./traditional_renders/performance_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TRADITIONAL RAY TRACING BASELINE REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("Test Configuration:\n")
        f.write(f"  Scene: Complex scene with {len(renderer.scene)} objects\n")
        f.write(f"  Lights: {len(renderer.light_sources)} emitive spheres\n")
        f.write(f"  Image: {renderer.image_width}x{renderer.image_height}\n")
        f.write(f"  Max bounces: {renderer.max_bounces}\n\n")
        
        f.write("Performance Results:\n")
        f.write(f"{'SPP':>6} {'Time (s)':>10} {'Rays (M)':>10} {'Rays/s (M)':>12} {'Hit Rate %':>12}\n")
        f.write("-"*60 + "\n")
        
        for r in results:
            f.write(f"{r['spp']:>6} {r['render_time']:>10.1f} {r['total_rays']/1e6:>10.1f} "
                   f"{r['rays_per_second']/1e6:>12.1f} {r['hit_rate']*100:>12.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FB IMPROVEMENT PREDICTIONS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Current best hit rate: {best_hit_rate*100:.4f}%\n")
        f.write(f"Traditional rays per light hit: {1/best_hit_rate:,.0f}\n\n")
        
        f.write("Expected FB improvements:\n")
        f.write("1. Hit rate improvement: 5x to 50x\n")
        f.write(f"2. Expected FB hit rate: {min(5.0, best_hit_rate*100*10):.1f}% to {min(20.0, best_hit_rate*100*50):.1f}%\n")
        f.write(f"3. Expected speedup: {1/best_hit_rate/20:,.1f}x to {1/best_hit_rate/5:,.1f}x\n")
        f.write("4. Primary benefit: Variance reduction in difficult lighting\n")
    
    print(f"\nFull report saved: {report_path}")
    print("\n" + "="*80)
    print("NEXT STEP: Train FB on this scene and compare results!")
    print("="*80)

if __name__ == "__main__":
    main()
