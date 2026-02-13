# train_rl_custom_scene.py
import numpy as np
from stable_baselines3 import SAC
from ray_tracer_env import RayTracerEnv
from vector import Vector, Angle
from colour import Colour
from object import Sphere
from material import Material
from light import GlobalLight, PointLight

def create_your_scene_with_original_lights():
    """Create YOUR exact scene for RL training with ORIGINAL lighting setup"""
    base_material = Material(reflective=False)
    reflective_material = Material(reflective=True)
    glass = Material(reflective=False, transparent=True, refractive_index=1.52)
    emitive_material = Material(emitive=True)
    
    spheres = [
        # Glass sphere (id=1 from original)
        Sphere(
            centre=Vector(-0.8, 0.6, 0),
            radius=0.3,
            material=glass,
            colour=Colour(255, 100, 100),
            id=1  # Moved to end
        ),
        # Large blue sphere (id=2 from original)
        Sphere(
            centre=Vector(0.8, -0.8, -10),
            radius=2.2,
            material=base_material,
            colour=Colour(204, 204, 255),
            id=2  # Moved to end
        ),
        # Small blue sphere (id=3 from original)
        Sphere(
            centre=Vector(0.3, 0.34, 0.1),
            radius=0.2,
            material=base_material,
            colour=Colour(0, 51, 204),
            id=3  # Moved to end
        ),
        # Reflective purple sphere (id=4 from original)
        Sphere(
            centre=Vector(5.6, 3, -2),
            radius=5,
            material=reflective_material,
            colour=Colour(153, 51, 153),
            id=4  # Moved to end
        ),
        # Green sphere (id=5 from original)
        Sphere(
            centre=Vector(-0.8, -0.8, -0.2),
            radius=0.25,
            material=base_material,
            colour=Colour(153, 204, 0),
            id=5  # Moved to end
        ),
        # Large yellow sphere (id=6 from original)
        Sphere(
            centre=Vector(-3, 10, -75),
            radius=30,
            material=base_material,
            colour=Colour(255, 204, 102),
            id=6  # Moved to end
        ),
        # SUN (LIGHT SOURCE)
        Sphere(
            centre=Vector(-0.6, 0.2, 6),
            radius=0.1,
            material=Material(emitive=True),
            colour=Colour(255, 255, 204),
            id=7  # Moved to end
        ),
    ]
    
    return spheres

def train_rl_on_your_scene():
    """Train RL agent on YOUR scene with EXACT original camera"""
    print("Training RL on YOUR custom scene with EXACT original camera...")
    
    spheres = create_your_scene_with_original_lights()
    
    # Create EXACT original lighting setup
    global_light_sources = [
        GlobalLight(
            vector=Vector(3, 1, -0.75).normalise(),
            colour=Colour(20, 20, 255),
            strength=1,
            max_angle=np.radians(90),
            func=0
        )
    ]
    
    # Original sun as PointLight
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
    
    # Add sun to spheres if needed
    # (Already added as emitive sphere with id=99)
    
    # Create environment with EXACT original camera setup
    # Your original camera parameters:
    # - Camera position: Vector(0, 0, 1)
    # - Image size: 601x601 (from 2*300 + 1)
    # - FOV: Approximately 90 degrees (based on ray grid calculation)
    # - No camera rotation
    
    env = RayTracerEnv(
        spheres=spheres,
        global_light_sources=global_light_sources,
        point_light_sources=point_light_sources,
        max_bounces=5,  # Original uses 5 bounces
        image_width=601,  # EXACT original: 601x601
        image_height=601,
        camera_position=Vector(0, 0, 1),  # EXACT original camera position
        camera_angle=Angle(0, 0, 0),  # No rotation
        fov=90,  # Approximately matches original ray grid
        background_colour=Colour(2, 2, 5),  # Original background
        render_mode=None
    )
    
    print(f"RL Environment Setup (matches original):")
    print(f"  • Camera position: ({env.camera_position.x}, {env.camera_position.y}, {env.camera_position.z})")
    print(f"  • Image size: {env.image_width}x{env.image_height}")
    print(f"  • FOV: {env.fov}°")
    print(f"  • Max bounces: {env.max_bounces}")
    print(f"  • Number of spheres: {len(env.spheres)}")
    print(f"  • Lights: {len(env.global_light_sources)} global, {len(env.point_light_sources)} point")
    
    # Verify the camera setup
    print(f"\nVerifying camera ray generation...")
    test_pixels = [
        (300, 300),  # Center pixel (should be (0, 0) in ray grid)
        (450, 300),  # Right of center
        (150, 300),  # Left of center
    ]
    
    for px, py in test_pixels:
        test_ray = env._get_initial_ray(px, py)
        print(f"  Pixel ({px}, {py}): ray direction = ({test_ray.D.x:.4f}, {test_ray.D.y:.4f}, {test_ray.D.z:.4f})")
    
    # Train SAC on YOUR scene
    print(f"\nStarting RL training for 50,000 steps...")
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=5000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        target_update_interval=1,
        target_entropy='auto',
    )
    
    model.learn(total_timesteps=50000, progress_bar=True)
    
    # Save the model
    model.save("rl_your_scene_exact_camera")
    print("✓ Model saved as 'rl_your_scene_exact_camera.zip'")
    
    # Test the trained model
    print(f"\nTesting trained model...")
    obs, info = env.reset(options={'pixel': (300, 300)})  # Center pixel
    done = False
    total_reward = 0
    
    print(f"Starting at pixel: {info['pixel']}")
    print(f"Initial ray: origin={info['initial_ray']['origin']}, direction={info['initial_ray']['direction']}")
    
    for step in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode ended at step {step+1}: {info.get('reason', 'unknown')}")
            break
            
        if step < 3:  # Show first 3 steps
            print(f"  Step {step+1}: reward={reward:.3f}, total={total_reward:.3f}")
    
    print(f"Final total reward: {total_reward:.3f}")
    
    return model

if __name__ == "__main__":
    train_rl_on_your_scene()