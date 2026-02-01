"""
Gymnasium Environment for Ray Tracing

This environment wraps the ray tracer to enable reinforcement learning
for light path sampling and rendering optimization.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

from vector import Vector, Angle
from ray import Ray, Intersection
from colour import Colour
from object import Sphere
from material import Material
from light import GlobalLight, PointLight


class RayTracerEnv(gym.Env):
    """
    A Gymnasium environment for ray tracing where an agent learns to
    sample light paths efficiently.
    
    The agent controls ray directions at each intersection point to
    maximize the gathered light/color information.
    
    Observation Space:
        - Current position (3D): x, y, z
        - Current direction (3D normalized): dx, dy, dz
        - Surface normal at hit point (3D normalized): nx, ny, nz
        - Material properties (4D): reflective, transparent, emitive, refractive_index
        - Current color accumulated (3D): r, g, b (normalized to 0-1)
        - Bounce count (1D): number of bounces so far
        - Through count (1D): number of transparent objects passed through
        Total: 18 dimensions
    
    Action Space:
        - Hemisphere direction sampling (2D continuous):
          - theta: polar angle [0, π/2] (0 = along normal, π/2 = tangent to surface)
          - phi: azimuthal angle [0, 2π] (rotation around normal)
        
    Reward:
        - Based on the illumination/color intensity gathered at each step
        - Terminal reward based on final color quality and path efficiency
    """
    
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        spheres=None,
        image_width=800,
        image_height=600,
        camera_position=Vector(0, 0, 0),
        camera_angle=Angle(0, 0, 0),
        fov=90,
        max_bounces=5,
        background_colour=Colour(0, 0, 0),
        global_light_sources=None,
        point_light_sources=None,
        render_mode=None
    ):
        super().__init__()
        
        # Scene configuration
        self.spheres = spheres if spheres is not None else []
        self.image_width = image_width
        self.image_height = image_height
        self.camera_position = camera_position
        self.camera_angle = camera_angle
        self.fov = fov
        self.max_bounces = max_bounces
        self.background_colour = background_colour
        self.global_light_sources = global_light_sources if global_light_sources is not None else []
        self.point_light_sources = point_light_sources if point_light_sources is not None else []
        self.render_mode = render_mode
        
        # Current episode state
        self.current_ray = None
        self.current_intersection = None
        self.current_pixel = None
        self.accumulated_color = Colour(0, 0, 0)
        self.bounce_count = 0
        self.through_count = 0
        self.total_reward = 0.0
        
        # Define observation space (18 dimensions)
        # Position (3) + Direction (3) + Normal (3) + Material (4) + Color (3) + Bounces (1) + Through (1)
        self.observation_space = spaces.Box(
            low=np.array([
                -np.inf, -np.inf, -np.inf,  # position (unbounded)
                -1, -1, -1,                   # direction (normalized)
                -1, -1, -1,                   # normal (normalized)
                0, 0, 0, 1,                   # material properties
                0, 0, 0,                      # accumulated color (0-1)
                0,                            # bounce count
                0                             # through count
            ], dtype=np.float32),
            high=np.array([
                np.inf, np.inf, np.inf,       # position (unbounded)
                1, 1, 1,                      # direction (normalized)
                1, 1, 1,                      # normal (normalized)
                1, 1, 1, 3,                   # material properties (refractive_index max ~3)
                1, 1, 1,                      # accumulated color (0-1)
                self.max_bounces,             # bounce count
                self.max_bounces              # through count
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define action space: hemisphere sampling
        # theta in [0, π/2], phi in [0, 2π]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([np.pi/2, 2*np.pi], dtype=np.float32),
            dtype=np.float32
        )
    
    def _get_initial_ray(self, pixel_x, pixel_y):
        """Generate initial camera ray for a given pixel"""
        # Calculate ray direction based on pixel position and camera settings
        aspect_ratio = self.image_width / self.image_height
        fov_radians = self.fov * np.pi / 180
        
        # Normalize pixel coordinates to [-1, 1]
        px = (2 * (pixel_x + 0.5) / self.image_width - 1) * aspect_ratio * np.tan(fov_radians / 2)
        py = (1 - 2 * (pixel_y + 0.5) / self.image_height) * np.tan(fov_radians / 2)
        
        # Create ray direction
        ray_direction = Vector(px, py, -1).normalise()
        
        # Debug print - uncomment to see ray directions
        if pixel_x == self.image_width // 2 and pixel_y == self.image_height // 2:
            print(f"  DEBUG: Center pixel ray direction: ({ray_direction.x:.3f}, {ray_direction.y:.3f}, {ray_direction.z:.3f})")
        
        # Apply camera rotation if needed
        if self.camera_angle.x != 0 or self.camera_angle.y != 0 or self.camera_angle.z != 0:
            ray_direction = ray_direction.rotate(self.camera_angle)
        
        return Ray(origin=self.camera_position, D=ray_direction)
    
    def _action_to_direction(self, action, normal):
        """
        Convert action (theta, phi) to a 3D direction vector in hemisphere around normal
        
        Args:
            action: [theta, phi] where theta in [0, π/2], phi in [0, 2π]
            normal: surface normal vector
        
        Returns:
            Vector: direction in world space
        """
        theta, phi = action[0], action[1]
        
        # Convert spherical coordinates to Cartesian (in local space)
        # Local space: normal is the "up" direction (0, 0, 1)
        local_x = np.sin(theta) * np.cos(phi)
        local_y = np.sin(theta) * np.sin(phi)
        local_z = np.cos(theta)
        
        # Create local direction vector
        local_dir = Vector(local_x, local_y, local_z)
        
        # Create coordinate frame around normal
        # Find a perpendicular vector to normal
        if abs(normal.z) < 0.9:
            tangent = Vector(0, 0, 1).crossProduct(normal).normalise()
        else:
            tangent = Vector(1, 0, 0).crossProduct(normal).normalise()
        
        bitangent = normal.crossProduct(tangent).normalise()
        
        # Transform local direction to world space
        world_dir = Vector(
            local_dir.x * tangent.x + local_dir.y * bitangent.x + local_dir.z * normal.x,
            local_dir.x * tangent.y + local_dir.y * bitangent.y + local_dir.z * normal.y,
            local_dir.x * tangent.z + local_dir.y * bitangent.z + local_dir.z * normal.z
        )
        
        return world_dir.normalise()
    
    def _get_observation(self):
        """Construct observation from current state"""
        if self.current_intersection is None or not self.current_intersection.intersects:
            # Ray didn't hit anything - return terminal observation
            return np.zeros(18, dtype=np.float32)
        
        pos = self.current_intersection.point
        direction = self.current_ray.D
        normal = self.current_intersection.normal
        material = self.current_intersection.object.material
        
        # Normalize color to [0, 1]
        color_norm = np.array([
            self.accumulated_color.r / 255.0,
            self.accumulated_color.g / 255.0,
            self.accumulated_color.b / 255.0
        ], dtype=np.float32)
        
        obs = np.array([
            # Position (3)
            pos.x, pos.y, pos.z,
            # Direction (3)
            direction.x, direction.y, direction.z,
            # Normal (3)
            normal.x, normal.y, normal.z,
            # Material (4)
            material.reflective,
            material.transparent,
            material.emitive,
            material.refractive_index,
            # Accumulated color (3)
            color_norm[0], color_norm[1], color_norm[2],
            # Bounce count (1)
            float(self.bounce_count),
            # Through count (1)
            float(self.through_count)
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_reward(self):
        """
        Calculate reward based on light gathered and path efficiency
        
        Returns:
            float: reward value
        """
        if self.current_intersection is None or not self.current_intersection.intersects:
            # Missed everything - small negative reward
            return -0.1
        
        # Get illumination at this point
        illumination = self.current_intersection.terminalRGB(
            spheres=self.spheres,
            background_colour=self.background_colour,
            global_light_sources=self.global_light_sources,
            point_light_sources=self.point_light_sources,
            max_bounces=0  # Don't recurse for reward calculation
        )
        
        # Reward based on brightness (normalized)
        brightness = (illumination.r + illumination.g + illumination.b) / (3 * 255)
        
        # Penalty for too many bounces (encourage efficiency)
        efficiency_penalty = -0.01 * self.bounce_count
        
        reward = brightness + efficiency_penalty
        
        return float(reward)
    
    def reset(self, seed=None, options=None):
    
        super().reset(seed=seed)
        
        # Reset state
        self.bounce_count = 0
        self.through_count = 0
        self.accumulated_color = Colour(0, 0, 0)
        self.total_reward = 0.0
        
        # Choose a random pixel or use provided pixel
        if options is not None and 'pixel' in options:
            self.current_pixel = options['pixel']
        else:
            self.current_pixel = (
                np.random.randint(0, self.image_width),
                np.random.randint(0, self.image_height)
            )
        
        # Generate initial camera ray
        self.current_ray = self._get_initial_ray(*self.current_pixel)
        
        # Find first intersection
        self.current_intersection = self.current_ray.nearestSphereIntersect(
            spheres=self.spheres,
            max_bounces=self.max_bounces
        )
        
        observation = self._get_observation()
        info = {
            'pixel': self.current_pixel,
            'bounce_count': self.bounce_count,
            'through_count': self.through_count,
            'initial_ray': {
                'origin': (self.current_ray.origin.x, self.current_ray.origin.y, self.current_ray.origin.z),
                'direction': (self.current_ray.D.x, self.current_ray.D.y, self.current_ray.D.z)
            }
        }
        
        return observation, info

    def step(self, action):
        """
        Take an action (sample new ray direction) and trace one step
        
        Args:
            action: [theta, phi] - hemisphere direction parameters
        
        Returns:
            observation: New observation after ray tracing step
            reward: Reward for this step
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Initialize info dictionary
        info = {'bounce_count': self.bounce_count, 'through_count': self.through_count}
        
        # Check if we can continue
        if self.current_intersection is None or not self.current_intersection.intersects:
            # Ray missed - episode terminates
            info['reason'] = 'ray_missed'
            info['total_reward'] = self.total_reward
            return (
                self._get_observation(),
                -1.0,  # Penalty for missing
                True,  # terminated
                False,  # truncated
                info
            )
        
        if self.bounce_count >= self.max_bounces:
            # Max bounces reached
            final_reward = self._calculate_reward()
            self.total_reward += final_reward
            info['reason'] = 'max_bounces'
            info['total_reward'] = self.total_reward
            return (
                self._get_observation(),
                final_reward,
                True,  # terminated
                True,   # truncated
                info
            )
        
        # Convert action to direction
        new_direction = self._action_to_direction(action, self.current_intersection.normal)
        
        # Create new ray from current intersection point
        new_ray = Ray(
            origin=self.current_intersection.point,
            D=new_direction
        )
        
        # Update bounce count
        self.bounce_count += 1
        
        # Trace to next intersection
        suppress_ids = [self.current_intersection.object.id]
        next_intersection = new_ray.nearestSphereIntersect(
            spheres=self.spheres,
            suppress_ids=suppress_ids,
            bounces=self.bounce_count,
            max_bounces=self.max_bounces,
            through_count=self.through_count
        )
        
        # Calculate reward for this step
        reward = self._calculate_reward()
        self.total_reward += reward
        
        # Update state
        self.current_ray = new_ray
        self.current_intersection = next_intersection
        
        # Update bounce_count in info (after increment)
        info['bounce_count'] = self.bounce_count
        
        # Accumulate color (simplified - you may want more sophisticated color accumulation)
        if next_intersection is not None and next_intersection.intersects:
            step_color = next_intersection.terminalRGB(
                spheres=self.spheres,
                background_colour=self.background_colour,
                global_light_sources=self.global_light_sources,
                point_light_sources=self.point_light_sources,
                max_bounces=0
            )
            self.accumulated_color = self.accumulated_color.addColour(step_color)
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        if next_intersection is None or not next_intersection.intersects:
            # Hit nothing - terminate
            terminated = True
            info['reason'] = 'ray_escaped'
        elif self.bounce_count >= self.max_bounces:
            # Max bounces
            terminated = True
            truncated = True
            info['reason'] = 'max_bounces'
        
        info['total_reward'] = self.total_reward
        
        observation = self._get_observation()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the current state
        
        Returns:
            RGB array if render_mode is 'rgb_array', otherwise None
        """
        if self.render_mode == "rgb_array":
            # Simple visualization: return current pixel color
            img = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
            if self.current_pixel is not None:
                px, py = self.current_pixel
                img[py, px] = [
                    min(255, max(0, self.accumulated_color.r)),
                    min(255, max(0, self.accumulated_color.g)),
                    min(255, max(0, self.accumulated_color.b))
                ]
            return img
        return None
    
    def close(self):
        """Clean up resources"""
        pass


# Example usage and testing
# Example usage and testing
if __name__ == "__main__":
    # Create a simple scene
    from material import Material
    
    # Define materials
    matte_material = Material(reflective=0, transparent=0, emitive=0.1, refractive_index=1)
    reflective_material = Material(reflective=1, transparent=0, emitive=0, refractive_index=1)
    glass_material = Material(reflective=0, transparent=1, emitive=0, refractive_index=1.5)
    
    # Create spheres - let's make them closer to the camera for testing
    scene_spheres = [
        Sphere(Vector(0, 0, -3), 1, matte_material, Colour(255, 0, 0), id=1),
        Sphere(Vector(2, 0, -3), 0.5, reflective_material, Colour(200, 200, 200), id=2),
        Sphere(Vector(-2, 0, -3), 0.5, glass_material, Colour(100, 100, 255), id=3),
    ]
    
    # Create lights
    global_lights = [
        GlobalLight(
            vector=Vector(0, -1, -1).normalise(),
            colour=Colour(255, 255, 255),
            strength=0.5,
            max_angle=np.pi/2
        )
    ]
    
    point_lights = [
        PointLight(
            id=99,
            position=Vector(0, 3, -3),
            colour=Colour(255, 255, 200),
            strength=10.0,
            max_angle=np.pi
        )
    ]
    
    # Create environment
    env = RayTracerEnv(
        spheres=scene_spheres,
        global_light_sources=global_lights,
        point_light_sources=point_lights,
        max_bounces=3
    )
    
    # Test the environment
    print("Testing RayTracerEnv...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Debug Information
    print("\n=== Debug Information ===")
    print(f"Camera position: ({env.camera_position.x}, {env.camera_position.y}, {env.camera_position.z})")
    print(f"Camera angle: ({env.camera_angle.x}, {env.camera_angle.y}, {env.camera_angle.z})")
    print(f"Field of View: {env.fov}")
    print(f"Image dimensions: {env.image_width}x{env.image_height}")
    print(f"Number of spheres: {len(env.spheres)}")
    
    # Correct attribute name is 'centre' not 'position'
    if len(env.spheres) > 0:
        print(f"Sphere centers: {[(s.centre.x, s.centre.y, s.centre.z) for s in env.spheres]}")
        print(f"Sphere radii: {[s.radius for s in env.spheres]}")
    
    # Test ray direction calculation for center pixel
    print("\n=== Testing Center Pixel ===")
    center_pixel = (env.image_width // 2, env.image_height // 2)
    center_ray = env._get_initial_ray(*center_pixel)
    print(f"Center pixel {center_pixel}:")
    print(f"  Ray origin: ({center_ray.origin.x}, {center_ray.origin.y}, {center_ray.origin.z})")
    print(f"  Ray direction: ({center_ray.D.x:.3f}, {center_ray.D.y:.3f}, {center_ray.D.z:.3f})")
    
    # Test a specific pixel that should hit the center sphere
    print("\n=== Running test episodes ===")
    num_episodes = 3
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        
        # Use center pixel for more reliable testing
        obs, info = env.reset(options={'pixel': center_pixel})
        
        # Debug initial ray
        init_ray = info['initial_ray']
        print(f"Initial ray:")
        print(f"  Origin: {init_ray['origin']}")
        print(f"  Direction: {init_ray['direction']}")
        
        # Check if we have an initial intersection
        if env.current_intersection and env.current_intersection.intersects:
            print(f"✓ Hit object ID: {env.current_intersection.object.id}")
            print(f"  Hit point: ({env.current_intersection.point.x:.2f}, {env.current_intersection.point.y:.2f}, {env.current_intersection.point.z:.2f})")
            print(f"  Surface normal: ({env.current_intersection.normal.x:.2f}, {env.current_intersection.normal.y:.2f}, {env.current_intersection.normal.z:.2f})")
            print(f"  Material: reflective={env.current_intersection.object.material.reflective}, "
                  f"transparent={env.current_intersection.object.material.transparent}")
        else:
            print("✗ No initial intersection - ray missed all objects")
            # Debug: test ray intersection manually
            test_intersection = center_ray.nearestSphereIntersect(
                spheres=env.spheres,
                max_bounces=env.max_bounces
            )
            if test_intersection and test_intersection.intersects:
                print(f"  But manual test shows intersection with object {test_intersection.object.id}!")
        
        # Try a few steps if we hit something
        if env.current_intersection and env.current_intersection.intersects:
            done = False
            step_count = 0
            total_episode_reward = 0
            
            while not done and step_count < 3:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_episode_reward += reward
                step_count += 1
                done = terminated or truncated
                
                print(f"Step {step_count}: reward={reward:.4f}, bounce_count={info['bounce_count']}, reason={info.get('reason', 'ongoing')}")
            
            print(f"Episode finished. Total reward: {total_episode_reward:.4f}")
        else:
            print("Skipping steps since no initial intersection.")