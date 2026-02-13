"""
Quick test to verify the complex scene loads and renders
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')  # Ensure current directory is in path

try:
    from complex_scene import create_complex_scene, analyze_scene_difficulty
    print("✓ complex_scene.py loaded successfully")
    
    from traditional_complex_scene_test import TraditionalComplexSceneRenderer
    print("✓ TraditionalComplexSceneRenderer loaded successfully")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nMake sure all required files are in the same directory:")
    print("1. vector.py")
    print("2. colour.py") 
    print("3. object.py")
    print("4. material.py")
    print("5. ray.py")
    print("6. light.py")
    print("7. complex_scene.py (just created)")
    print("8. traditional_complex_scene_test.py (just created)")
    sys.exit(1)

def quick_scene_preview():
    """Show a quick preview of the scene setup"""
    print("\n" + "="*80)
    print("WELL-LIT COMPLEX SCENE PREVIEW")
    print("="*80)
    
    # Create scene
    from complex_scene import create_complex_scene, create_camera_for_scene
    from vector import Vector
    
    spheres = create_complex_scene()
    camera_pos, _ = create_camera_for_scene()
    
    # Get all lights
    all_lights = [s for s in spheres if s.material.emitive]
    
    # Analyze scene difficulty
    main_lights, small_lights = analyze_scene_difficulty(spheres, camera_pos)
    
    # Visualize scene layout
    fig = plt.figure(figsize=(16, 12))
    
    # Top view (XZ plane)
    ax1 = plt.subplot(231)
    for sphere in spheres:
        if sphere.material.emitive:
            color = 'gold' if sphere.radius >= 0.5 else 'red'
            alpha = 0.9
        elif sphere.material.reflective > 0.5:
            color = 'blue'
            alpha = 0.6
        elif sphere.material.transparent > 0.5:
            color = 'cyan'
            alpha = 0.6
        else:
            color = 'gray'
            alpha = 0.4
            
        circle = plt.Circle((sphere.centre.x, sphere.centre.z), 
                          sphere.radius, color=color, alpha=alpha, fill=True)
        ax1.add_patch(circle)
    
    # Mark camera position
    ax1.scatter(camera_pos.x, camera_pos.z, color='black', s=100, marker='^', label='Camera')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_title('Top View (XZ plane)\nGold=Main Lights, Red=Small Lights')
    ax1.set_aspect('equal', 'box')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-8, 8)
    ax1.set_ylim(-2, 15)
    ax1.legend()
    
    # Side view (YZ plane)
    ax2 = plt.subplot(232)
    for sphere in spheres:
        if sphere.material.emitive:
            color = 'gold' if sphere.radius >= 0.5 else 'red'
            alpha = 0.9
        elif sphere.material.reflective > 0.5:
            color = 'blue'
            alpha = 0.6
        elif sphere.material.transparent > 0.5:
            color = 'cyan'
            alpha = 0.6
        else:
            color = 'gray'
            alpha = 0.4
            
        circle = plt.Circle((sphere.centre.y, sphere.centre.z), 
                          sphere.radius, color=color, alpha=alpha, fill=True)
        ax2.add_patch(circle)
    
    # Mark camera position
    ax2.scatter(camera_pos.y, camera_pos.z, color='black', s=100, marker='^')
    
    ax2.set_xlabel('Y')
    ax2.set_ylabel('Z')
    ax2.set_title('Side View (YZ plane)')
    ax2.set_aspect('equal', 'box')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-6, 8)
    ax2.set_ylim(-2, 15)
    
    # Front view (XY plane)
    ax3 = plt.subplot(233)
    for sphere in spheres:
        if sphere.material.emitive:
            color = 'gold' if sphere.radius >= 0.5 else 'red'
            alpha = 0.9
        elif sphere.material.reflective > 0.5:
            color = 'blue'
            alpha = 0.6
        elif sphere.material.transparent > 0.5:
            color = 'cyan'
            alpha = 0.6
        else:
            color = 'gray'
            alpha = 0.4
            
        circle = plt.Circle((sphere.centre.x, sphere.centre.y), 
                          sphere.radius, color=color, alpha=alpha, fill=True)
        ax3.add_patch(circle)
    
    # Mark camera position
    ax3.scatter(camera_pos.x, camera_pos.y, color='black', s=100, marker='^')
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Front View (XY plane)')
    ax3.set_aspect('equal', 'box')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-8, 8)
    ax3.set_ylim(-6, 8)
    
    # Statistics panel
    ax4 = plt.subplot(212)
    
    stats_text = f"""SCENE STATISTICS:

Total Objects: {len(spheres)}
• Main Lights (Gold): {len(main_lights)} (radius >= 0.5)
• Small Lights (Red): {len(small_lights)} (radius < 0.5)
• Mirror Surfaces (Blue): {len([s for s in spheres if s.material.reflective > 0.5])}
• Glass Surfaces (Cyan): {len([s for s in spheres if s.material.transparent > 0.5])}
• Matte Surfaces (Gray): {len([s for s in spheres if not s.material.emitive and 
                               s.material.reflective < 0.3 and s.material.transparent < 0.3])}

CAMERA POSITION:
X: {camera_pos.x:.1f}, Y: {camera_pos.y:.1f}, Z: {camera_pos.z:.1f}

CHALLENGING ELEMENTS (for Traditional Ray Tracing):
1. Small Lights: 
"""
    
    # Add small light details
    for i, light in enumerate(small_lights[:3]):  # Show first 3
        distance = camera_pos.distanceFrom(light.centre)
        solid_angle = 2 * np.pi * (1 - np.sqrt(1 - (light.radius/distance)**2))
        probability = solid_angle / (4 * np.pi) * 100
        stats_text += f"   • ID {light.id}: Radius={light.radius:.3f}, Hit Prob={probability:.4f}%\n"
    
    if len(small_lights) > 3:
        stats_text += f"   • ... and {len(small_lights) - 3} more small lights\n"
    
    stats_text += f"""
2. Mirror Sphere: Causes many reflection bounces
3. Glass Objects: Create caustics (high Monte Carlo variance)
4. Object Cluster: {len([s for s in spheres if 20 <= s.id < 30])} interlocked spheres
5. Geometric Complexity: {len(spheres)} total intersection tests per ray

EXPECTED FB ADVANTAGE:
• 10-100x better hit rate on small lights
• More efficient reflection sampling
• Reduced variance in caustic regions
• Overall faster convergence to clean image
"""
    
    ax4.text(0.02, 0.98, stats_text, fontfamily='monospace', fontsize=9,
            verticalalignment='top', transform=ax4.transAxes)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.suptitle('Well-Lit Complex Scene Layout', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./complex_scene_layout_well_lit.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nScene layout visualization saved: complex_scene_layout_well_lit.png")
    
    # Test traditional renderer with better settings for visibility
    print("\n" + "="*80)
    print("TESTING TRADITIONAL RENDERER (LOW QUALITY BUT VISIBLE)")
    print("="*80)
    
    renderer = TraditionalComplexSceneRenderer()
    renderer.image_width = 400  # Decent size for preview
    renderer.image_height = 300
    renderer.samples_per_pixel = 4  # Low but shows scene
    renderer.max_bounces = 3  # Reduce for speed
    
    print(f"Rendering test image {renderer.image_width}x{renderer.image_height} with 4 spp...")
    image = renderer.render_scene()
    
    # Display with better color scaling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original image
    im1 = ax1.imshow(np.clip(image, 0, 1))
    ax1.set_title('Traditional Ray Tracing - 4 spp\n(Direct Display)')
    ax1.axis('off')
    
    # Enhanced brightness for preview
    # Apply gamma correction to make dark areas more visible
    image_enhanced = np.clip(image ** 0.7, 0, 1)  # Gamma 0.7 for brightness
    
    im2 = ax2.imshow(image_enhanced)
    ax2.set_title('Traditional Ray Tracing - 4 spp\n(Enhanced Brightness)')
    ax2.axis('off')
    
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('./complex_scene_quick_test_well_lit.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Quick test render saved: complex_scene_quick_test_well_lit.png")
    
    # Show statistics
    print("\nQuick Test Statistics:")
    print(f"  Total rays traced: {renderer.stats['total_rays']:,}")
    print(f"  Total intersections tested: {renderer.stats['total_intersections']:,}")
    print(f"  Light hits (all lights): {renderer.stats['light_hits']:,}")
    print(f"  Overall light hit rate: {renderer.stats['light_hits']/max(1, renderer.stats['total_rays'])*100:.4f}%")
    print(f"  Render time: {renderer.stats['render_time']:.1f}s")
    print(f"  Rays per second: {renderer.stats['rays_per_second']:,.0f}")
    
    # Estimate small light hits
    if small_lights and all_lights:
        small_light_ids = [light.id for light in small_lights]
        # This is an estimate - in reality we'd need to track per-light hits
        estimated_small_hits = renderer.stats['light_hits'] * len(small_lights) / len(all_lights)
        print(f"\nEstimated small light hits: ~{estimated_small_hits:.0f}")
        print(f"Estimated small light hit rate: ~{estimated_small_hits/max(1, renderer.stats['total_rays'])*100:.6f}%")
        
        # Calculate expected improvement
        current_small_hit_rate = estimated_small_hits / max(1, renderer.stats['total_rays'])
        print(f"\nFB TARGET: Improve small light hit rate by 10-100x")
        print(f"Current: 1 hit per {1/current_small_hit_rate:,.0f} rays")
        print(f"FB Goal: 1 hit per {1/(current_small_hit_rate*10):,.0f} to {1/(current_small_hit_rate*100):,.0f} rays")
    
    print("\n" + "="*80)
    print("SCENE VERIFICATION COMPLETE!")
    print("="*80)
    print("\nKey improvements in this version:")
    print("1. Scene is WELL-LIT and VISIBLE (3 large main lights)")
    print("2. Still has CHALLENGING elements (2 small, occluded lights)")
    print("3. Clear geometric structure (mirror, glass, cluster visible)")
    print("4. Good balance: Traditional can render it, but FB will be faster")
    
    print("\nNext steps:")
    print("1. Run traditional_complex_scene_test.py for full performance test")
    print("2. Note the DIFFERENCE between main vs small light hit rates")
    print("3. Train FB to specifically target the small lights")
    print("4. Compare FB's efficiency on the challenging parts")

if __name__ == "__main__":
    quick_scene_preview()
