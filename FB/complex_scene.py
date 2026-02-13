"""
Complex scene for testing FB vs Traditional Ray Tracing
This scene features:
1. Good overall lighting (visible scene)
2. Still has challenging indirect lighting scenarios
3. Mirrored surfaces causing multiple reflections
4. Glass surfaces with refraction
5. Small secondary lights requiring precise sampling
6. Clear visual structure to see what's happening
"""

import numpy as np
from vector import Vector
from colour import Colour
from object import Sphere
from material import Material
from light import GlobalLight, PointLight


def create_complex_scene():
    """Create a complex but well-lit scene with challenging indirect lighting"""
    
    print("Creating complex but well-lit scene...")
    
    # Define materials with better reflectivity for visibility
    matte_gray = Material(reflective=0.2, transparent=0, emitive=0, refractive_index=1)
    matte_red = Material(reflective=0.25, transparent=0, emitive=0, refractive_index=1)
    matte_green = Material(reflective=0.25, transparent=0, emitive=0, refractive_index=1)
    matte_blue = Material(reflective=0.25, transparent=0, emitive=0, refractive_index=1)
    matte_yellow = Material(reflective=0.25, transparent=0, emitive=0, refractive_index=1)
    
    mirror = Material(reflective=0.8, transparent=0, emitive=0, refractive_index=1)
    glass = Material(reflective=0.1, transparent=0.8, emitive=0, refractive_index=1.5)
    glossy = Material(reflective=0.4, transparent=0, emitive=0, refractive_index=1)
    emitive = Material(reflective=0, transparent=0, emitive=1, refractive_index=1)
    
    # Bright emitter materials (lights)
    bright_white = Material(reflective=0, transparent=0, emitive=1, refractive_index=1)
    warm_light = Material(reflective=0, transparent=0, emitive=1, refractive_index=1)
    cool_light = Material(reflective=0, transparent=0, emitive=1, refractive_index=1)
    
    spheres = []
    
    # ===== MAIN LIGHTING (MAKE SCENE VISIBLE) =====
    # Large ceiling light (primary illumination)
    spheres.append(Sphere(
        id=100,
        centre=Vector(0, 5, 3),
        radius=1.5,
        material=bright_white,
        colour=Colour(255, 255, 240)  # Bright warm white
    ))
    
    # Additional fill lights
    spheres.append(Sphere(
        id=101,
        centre=Vector(-3, 3, 6),
        radius=1.0,
        material=warm_light,
        colour=Colour(255, 230, 200)  # Warm light
    ))
    
    spheres.append(Sphere(
        id=102,
        centre=Vector(3, 3, 6),
        radius=1.0,
        material=cool_light,
        colour=Colour(200, 230, 255)  # Cool light
    ))
    
    # ===== FLOOR AND WALLS (BRIGHTER COLORS) =====
    # Floor (large sphere as ground plane)
    spheres.append(Sphere(
        id=1,
        centre=Vector(0, -100, 0),
        radius=99,
        material=matte_gray,
        colour=Colour(220, 220, 220)  # Bright gray floor
    ))
    
    # Back wall
    spheres.append(Sphere(
        id=2,
        centre=Vector(0, 0, -100),
        radius=99,
        material=matte_blue,
        colour=Colour(180, 200, 255)  # Light blue wall
    ))
    
    # Left wall
    spheres.append(Sphere(
        id=3,
        centre=Vector(-100, 0, 0),
        radius=99,
        material=matte_red,
        colour=Colour(255, 200, 200)  # Light red wall
    ))
    
    # Right wall
    spheres.append(Sphere(
        id=4,
        centre=Vector(100, 0, 0),
        radius=99,
        material=matte_green,
        colour=Colour(200, 255, 200)  # Light green wall
    ))
    
    # ===== VISIBLE COMPLEX OBJECTS =====
    
    # 1. Large mirrored sphere (centerpiece - creates interesting reflections)
    spheres.append(Sphere(
        id=10,
        centre=Vector(0, 0, 3),
        radius=1.2,
        material=mirror,
        colour=Colour(255, 255, 255)
    ))
    
    # 2. Glass sphere (refracts and shows caustics)
    spheres.append(Sphere(
        id=11,
        centre=Vector(-2.5, -0.5, 4),
        radius=1.0,
        material=glass,
        colour=Colour(240, 240, 255)
    ))
    
    # 3. Glossy colored spheres (for visual interest)
    spheres.append(Sphere(
        id=12,
        centre=Vector(2.5, -0.5, 4),
        radius=0.9,
        material=glossy,
        colour=Colour(255, 220, 180)  # Peach
    ))
    
    # 4. Cluster of smaller spheres (geometric complexity)
    cluster_center = Vector(0, 1.5, 6)
    colors = [
        Colour(255, 200, 200),  # Light red
        Colour(200, 255, 200),  # Light green
        Colour(200, 200, 255),  # Light blue
        Colour(255, 255, 200),  # Light yellow
        Colour(255, 200, 255),  # Light purple
        Colour(200, 255, 255),  # Light cyan
    ]
    
    for i in range(6):
        angle = i * 60 * np.pi / 180
        height = (i % 2) * 0.3
        x = cluster_center.x + 1.8 * np.cos(angle)
        z = cluster_center.z + 1.8 * np.sin(angle)
        
        # Vary materials in cluster
        if i % 3 == 0:
            cluster_mat = glossy
        elif i % 3 == 1:
            cluster_mat = matte_blue
        else:
            cluster_mat = Material(reflective=0.3, transparent=0, emitive=0, refractive_index=1)
        
        spheres.append(Sphere(
            id=20 + i,
            centre=Vector(x, cluster_center.y + height, z),
            radius=0.4 + 0.1 * (i % 3),
            material=cluster_mat,
            colour=colors[i]
        ))
    
    # 5. Glass pyramid (creates complex refractions)
    pyramid_center = Vector(-3, -0.8, 5)
    pyramid_glass = Material(reflective=0.1, transparent=0.85, emitive=0, refractive_index=1.52)
    
    for i in range(4):
        angle = i * 90 * np.pi / 180
        x = pyramid_center.x + 0.8 * np.cos(angle)
        z = pyramid_center.z + 0.8 * np.sin(angle)
        
        spheres.append(Sphere(
            id=30 + i,
            centre=Vector(x, pyramid_center.y + 0.3, z),
            radius=0.25,
            material=pyramid_glass,
            colour=Colour(220, 240, 255)
        ))
    
    # Center sphere of pyramid
    spheres.append(Sphere(
        id=34,
        centre=Vector(pyramid_center.x, pyramid_center.y + 1.0, pyramid_center.z),
        radius=0.35,
        material=pyramid_glass,
        colour=Colour(220, 240, 255)
    ))
    
    # 6. DIFFICULT-TO-REACH SMALL LIGHTS (the FB challenge)
    # These are small and partially occluded, but not the main light sources
    
    # Small light inside a "lantern" (occluded but visible)
    lantern_center = Vector(3.5, 1.0, 7)
    
    # Lantern structure (partially occludes the light)
    for i in range(4):
        angle = i * 90 * np.pi / 180
        x = lantern_center.x + 0.6 * np.cos(angle)
        z = lantern_center.z + 0.6 * np.sin(angle)
        
        spheres.append(Sphere(
            id=40 + i,
            centre=Vector(x, lantern_center.y, z),
            radius=0.2,
            material=matte_yellow,
            colour=Colour(220, 200, 150)
        ))
    
    # Small light inside lantern (FB's target)
    spheres.append(Sphere(
        id=44,
        centre=lantern_center,
        radius=0.12,  # Small - hard to hit randomly
        material=emitive,
        colour=Colour(255, 240, 200)  # Warm light
    ))
    
    # Another small light behind objects
    spheres.append(Sphere(
        id=45,
        centre=Vector(-4, 0.5, 8),
        radius=0.15,  # Small
        material=emitive,
        colour=Colour(200, 230, 255)  # Cool light
    ))
    
    # Objects partially occluding the small light
    for i in range(3):
        angle = i * 120 * np.pi / 180
        x = -4 + 0.7 * np.cos(angle)
        z = 8 + 0.7 * np.sin(angle)
        
        spheres.append(Sphere(
            id=50 + i,
            centre=Vector(x, 0.3, z),
            radius=0.3,
            material=glossy,
            colour=Colour(180, 180, 200)
        ))
    
    # 7. Reflective floor objects (create caustics with lights)
    for i in range(5):
        x = -2 + i * 1.0
        if i % 2 == 0:
            # Glass spheres on floor
            spheres.append(Sphere(
                id=60 + i,
                centre=Vector(x, -0.95, 2),
                radius=0.25,
                material=glass,
                colour=Colour(255, 255, 255)
            ))
        else:
            # Reflective spheres on floor
            spheres.append(Sphere(
                id=60 + i,
                centre=Vector(x, -0.95, 2),
                radius=0.25,
                material=Material(reflective=0.6, transparent=0, emitive=0, refractive_index=1),
                colour=Colour(220, 220, 220)
            ))
    
    # 8. Textured wall (array of small spheres - geometric complexity)
    for i in range(6):
        for j in range(4):
            x = -5 + i * 2.0
            z = 10 + j * 1.5
            
            # Create pattern
            if (i + j) % 4 == 0:
                mat = glossy
                color = Colour(255, 220, 220)
            elif (i + j) % 4 == 1:
                mat = matte_blue
                color = Colour(220, 220, 255)
            elif (i + j) % 4 == 2:
                mat = matte_green
                color = Colour(220, 255, 220)
            else:
                mat = Material(reflective=0.3, transparent=0, emitive=0, refractive_index=1)
                color = Colour(255, 255, 220)
            
            spheres.append(Sphere(
                id=100 + i*10 + j,
                centre=Vector(x, -0.5, z),
                radius=0.35,
                material=mat,
                colour=color
            ))
    
    print(f"Created well-lit complex scene with {len(spheres)} objects")
    print("Scene features:")
    print("  - 3 large main lights (make scene clearly visible)")
    print("  - 2 small challenging lights (FB's optimization target)")
    print("  - Mirror sphere (centerpiece for reflections)")
    print("  - Glass sphere and pyramid (refraction complexity)")
    print("  - Cluster of 6 colorful spheres")
    print("  - 5 floor spheres creating caustics")
    print("  - Textured wall with pattern")
    print("  - Good overall illumination with challenging elements")
    
    return spheres

def create_camera_for_scene():
    """Create camera setup that shows the interesting parts"""
    # Position camera to see the mirror sphere, glass sphere, and cluster
    camera_position = Vector(0, 1, 10)  # Slightly elevated for better view
    camera_angle = Vector(0, -0.1, 0)   # Looking slightly down
    
    return camera_position, camera_angle

def create_lights_for_scene():
    """Create additional lights for the scene"""
    # We're using emitive spheres as primary lights
    # Add some global fill lights for even illumination
    from light import GlobalLight
    
    global_lights = []
    
    # Soft top lighting (complements the ceiling light)
    global_lights.append(GlobalLight(
        vector=Vector(0, -1, -0.1).normalise(),
        colour=Colour(100, 100, 120),
        strength=0.3,
        max_angle=np.radians(80),
        func=0
    ))
    
    # Soft front lighting
    global_lights.append(GlobalLight(
        vector=Vector(0, 0, -1).normalise(),
        colour=Colour(80, 80, 100),
        strength=0.2,
        max_angle=np.radians(60),
        func=0
    ))
    
    return global_lights, []  # No point lights, using emitive spheres

def analyze_scene_difficulty(spheres, camera_position):
    """Analyze why this scene is still challenging despite being well-lit"""
    print("\n" + "="*80)
    print("SCENE DIFFICULTY ANALYSIS")
    print("="*80)
    
    # Categorize lights
    main_lights = [s for s in spheres if s.material.emitive and s.radius >= 0.5]
    small_lights = [s for s in spheres if s.material.emitive and s.radius < 0.5]
    
    print(f"\nLight sources:")
    print(f"  Main lights (easy to hit): {len(main_lights)}")
    print(f"  Small lights (FB challenge): {len(small_lights)}")
    
    print("\nMain lights (provide overall illumination):")
    for light in main_lights:
        distance = camera_position.distanceFrom(light.centre)
        solid_angle = 2 * np.pi * (1 - np.sqrt(1 - (light.radius/distance)**2))
        probability = solid_angle / (4 * np.pi) * 100
        print(f"  ID {light.id}: radius={light.radius:.2f}, hit probability={probability:.2f}%")
    
    print("\nSmall challenging lights (FB's optimization target):")
    for light in small_lights:
        distance = camera_position.distanceFrom(light.centre)
        solid_angle = 2 * np.pi * (1 - np.sqrt(1 - (light.radius/distance)**2))
        probability = solid_angle / (4 * np.pi) * 100
        
        # Check occlusion
        occluded_by = []
        for sphere in spheres:
            if sphere == light or sphere.material.emitive:
                continue
            
            # Simple line-of-sight check
            to_light = light.centre.subtractVector(camera_position).normalise()
            test_dist = light.centre.distanceFrom(camera_position)
            
            # Check if sphere is between camera and light
            to_sphere = sphere.centre.subtractVector(camera_position)
            proj_length = to_sphere.dotProduct(to_light)
            
            if 0 < proj_length < test_dist:
                distance_to_ray = (camera_position.addVector(to_light.scaleByLength(proj_length))
                                  .distanceFrom(sphere.centre))
                if distance_to_ray < sphere.radius * 1.2:
                    occluded_by.append(sphere.id)
        
        occlusion_text = f", occluded by {len(occluded_by)} objects" if occluded_by else ""
        
        print(f"  ID {light.id}: radius={light.radius:.3f}, "
              f"hit probability={probability:.4f}%{occlusion_text}")
        print(f"    Position: ({light.centre.x:.2f}, {light.centre.y:.2f}, {light.centre.z:.2f})")
        print(f"    Rays needed for 95% chance: {int(np.log(0.05)/np.log(1 - probability/100)):,}")
    
    # Count challenging materials
    mirrors = [s for s in spheres if s.material.reflective > 0.5]
    glass = [s for s in spheres if s.material.transparent > 0.5]
    
    print(f"\nChallenging materials:")
    print(f"  Mirror surfaces: {len(mirrors)} (cause many bounce rays)")
    print(f"  Glass surfaces: {len(glass)} (require precise sampling for caustics)")
    
    print("\n" + "="*80)
    print("WHY THIS SCENE IS STILL CHALLENGING FOR TRADITIONAL RAY TRACING:")
    print("="*80)
    print("1. Small lights (ID 44, 45) have ~0.01% random hit probability")
    print("2. Mirror sphere causes many reflection rays (computationally expensive)")
    print("3. Glass creates caustics (high variance in Monte Carlo)")
    print("4. Cluster of spheres requires many intersection tests")
    print("5. Despite good overall lighting, DETAILS require many samples")
    
    print("\nFB SHOULD EXCEL AT:")
    print("1. Learning to aim at small lights through reflections")
    print("2. Efficiently sampling the mirror sphere (not wasting rays)")
    print("3. Finding light paths to occluded regions")
    print("4. Reducing variance in caustic regions")
    
    return main_lights, small_lights

if __name__ == "__main__":
    
    # Test the scene creation
    spheres = create_complex_scene()
    camera_pos, camera_angle = create_camera_for_scene()
    global_lights, point_lights = create_lights_for_scene()
    
    print(f"\nCamera position: ({camera_pos.x:.1f}, {camera_pos.y:.1f}, {camera_pos.z:.1f})")
    print(f"Camera angle: ({camera_angle.x:.2f}, {camera_angle.y:.2f}, {camera_angle.z:.2f})")
    print(f"Number of global lights: {len(global_lights)}")
    
    # Count light sources
    all_lights = [s for s in spheres if s.material.emitive]
    print(f"Number of emitive spheres (light sources): {len(all_lights)}")
    
    # Analyze difficulty
    main_lights, small_lights = analyze_scene_difficulty(spheres, camera_pos)
    
    # Calculate expected performance
    print("\n" + "="*80)
    print("EXPECTED PERFORMANCE CHARACTERISTICS")
    print("="*80)
    
    # Calculate average hit probability for small lights
    if small_lights:
        total_prob = 0
        for light in small_lights:
            distance = camera_pos.distanceFrom(light.centre)
            solid_angle = 2 * np.pi * (1 - np.sqrt(1 - (light.radius/distance)**2))
            total_prob += solid_angle / (4 * np.pi) * 100
        
        avg_small_prob = total_prob / len(small_lights)
        
        print(f"\nSmall lights average hit probability: {avg_small_prob:.5f}%")
        print(f"Traditional rays per small light hit: ~{1/(avg_small_prob/100):,.0f}")
        print(f"FB target improvement: 10x to 100x better hit rate")
        print(f"Expected FB rays per hit: ~{1/(avg_small_prob/100)/50:,.0f} to {1/(avg_small_prob/100)/10:,.0f}")
    
    print("\nScene should render with:")
    print("• Good overall visibility (thanks to large lights)")
    print("• Visible geometric complexity (mirror, glass, cluster)")
    print("• Clear areas where FB should show improvement (small light contributions)")
    print("• Enough challenge to demonstrate FB's value")
