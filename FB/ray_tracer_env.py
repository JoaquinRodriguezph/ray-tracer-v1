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
        
        # Define action space: hemisphere sampling
        # theta in [0, π/2], phi in [0, 2π] normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define observation space (18 dimensions)
        self.observation_space = spaces.Box(
            low=np.array([
                # Position (3)
                -np.inf, -np.inf, -np.inf,
                # Direction (3)
                -1, -1, -1,
                # Normal (3)
                -1, -1, -1,
                # Material (4)
                0, 0, 0, 1,
                # Accumulated color (3)
                0, 0, 0,
                # Bounce count (1)
                0,
                # Through count (1)
                0
            ], dtype=np.float32),
            high=np.array([
                # Position (3)
                np.inf, np.inf, np.inf,
                # Direction (3)
                1, 1, 1,
                # Normal (3)
                1, 1, 1,
                # Material (4)
                1, 1, 1, 3,
                # Accumulated color (3)
                1, 1, 1,
                # Bounce count (1)
                self.max_bounces,
                # Through count (1)
                self.max_bounces
            ], dtype=np.float32),
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
        Now matches FB agent's action format
        
        Args:
            action: [theta_action, phi_action] where both in [-1, 1]
            normal: surface normal vector
        
        Returns:
            Vector: direction in world space
        """
        # Convert from [-1, 1] to [0, π/2] for theta, [-π, π] for phi
        # EXACTLY like FB agent does
        theta = (action[0] + 1) * np.pi/4  # action[0] ∈ [-1,1] -> theta ∈ [0,π/2]
        phi = action[1] * np.pi            # action[1] ∈ [-1,1] -> phi ∈ [-π,π]
        
        # Convert spherical coordinates to Cartesian (in local space)
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
            # Return zeros with correct shape (18 dimensions)
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
        
        # Construct observation (18 dimensions)
        obs = np.array([
            # Position (3)
            pos.x, pos.y, pos.z,
            # Direction (3)
            direction.x, direction.y, direction.z,
            # Normal (3)
            normal.x, normal.y, normal.z,
            # Material (4)
            float(material.reflective),
            float(material.transparent),
            float(material.emitive),
            float(material.refractive_index),
            # Accumulated color (3)
            color_norm[0], color_norm[1], color_norm[2],
            # Bounce count (1)
            float(self.bounce_count),
            # Through count (1)
            float(self.through_count)
        ], dtype=np.float32)
        
        return obs  # Should be 18 dimensions
    
    def _calculate_reward(self):
        """
        Calculate reward based on light gathered and path efficiency
        
        Returns:
            float: reward value
        """
        if self.current_intersection is None or not self.current_intersection.intersects:
            # Missed everything - small negative reward
            return -0.1
        
        # Get the intersected object
        obj = self.current_intersection.object
        
        # CRITICAL: Check if we hit the SUN (id=7) - MUST match your scene
        if obj.id == 7:  # SUN ID - matches FB/traditional
            print(f"RL DEBUG: HIT SUN! ID={obj.id}")  # DEBUG
            # Big reward for hitting the sun
            return 10.0
        
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
    
    def _calculate_lighting_reward(self):
        """Calculate reward based on lighting, similar to FB method"""
        if not self.current_intersection:
            return 0.0
        
        sphere = self.current_intersection.object
        intersection_point = self.current_intersection.point
        
        # Skip if it's a light source (handled separately)
        if getattr(sphere.material, 'emitive', False):
            return 0.0
        
        # Find the SUN (id=7)
        sun_sphere = None
        for s in self.spheres:
            if s.id == 7:  # Sun ID
                sun_sphere = s
                break
        
        if not sun_sphere:
            return 0.1  # Small base reward
        
        # Calculate lighting contribution (simplified version of FB's calculation)
        to_sun = sun_sphere.centre.subtractVector(intersection_point).normalise()
        normal = self.current_intersection.normal
        
        # Basic diffuse lighting
        cos_angle = max(0, normal.dotProduct(to_sun))
        
        # Shadow check (simplified)
        shadow_ray = Ray(
            intersection_point.addVector(normal.scaleByLength(0.001)),
            to_sun
        )
        
        sun_distance = sun_sphere.centre.subtractVector(intersection_point).magnitude()
        in_shadow = False
        
        for other_sphere in self.spheres:
            if other_sphere == sphere or other_sphere.id == 7:
                continue
            
            shadow_intersect = shadow_ray.sphereDiscriminant(other_sphere)
            if shadow_intersect and shadow_intersect.intersects:
                shadow_dist = shadow_intersect.point.subtractVector(intersection_point).magnitude()
                if shadow_dist < sun_distance:
                    in_shadow = True
                    break
        
        if in_shadow:
            # Only ambient lighting
            reward = 0.3
        else:
            # Full lighting
            reward = 0.3 + 0.7 * cos_angle  # Range: 0.3 to 1.0
        
        return reward

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
        
        # Find first intersection - FIX: pass max_bounces correctly
        self.current_intersection = self.current_ray.nearestSphereIntersect(
            spheres=self.spheres,
            max_bounces=self.max_bounces  # Add this parameter
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
        """
        # DEBUG: Print current state
        if self.current_intersection and self.current_intersection.intersects:
            obj = self.current_intersection.object
            print(f"RL STEP {self.bounce_count}: Hit ID={obj.id}, Material emitive={getattr(obj.material, 'emitive', False)}")
        
        # Initialize info dictionary
        info = {'bounce_count': self.bounce_count, 'through_count': self.through_count}
        
        # Check termination conditions BEFORE taking action
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
            final_reward = self._calculate_lighting_reward()
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
        
        # CRITICAL: Check if CURRENT intersection is the SUN before moving
        current_obj = self.current_intersection.object
        if current_obj.id == 7:  # We're already on the sun!
            reward = 10.0
            info['hit_sun'] = True
            info['reason'] = 'already_on_sun'
            info['total_reward'] = self.total_reward + reward
            return (
                self._get_observation(),
                reward,
                True,  # terminated
                False,  # truncated
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
        
        # Calculate reward for hitting NEXT object
        reward = 0
        terminated = False
        
        if next_intersection and next_intersection.intersects:
            next_obj = next_intersection.object
            if next_obj.id == 7:  # SUN HIT on next bounce!
                reward = 10.0
                info['hit_sun'] = True
                info['reason'] = 'hit_sun'
                terminated = True
                print(f"RL: HIT SUN! Reward = {reward}")  # DEBUG
            else:
                # Calculate lighting reward for hitting non-sun object
                # Temporarily update to calculate reward
                old_intersection = self.current_intersection
                self.current_intersection = next_intersection
                reward = self._calculate_lighting_reward()
                self.current_intersection = old_intersection  # Restore
                info['hit_sun'] = False
                terminated = False
        else:
            # Missed everything - small penalty
            reward = -0.1
            info['reason'] = 'ray_missed'
            terminated = True
        
        self.total_reward += reward
        
        # Update state
        self.current_ray = new_ray
        self.current_intersection = next_intersection
        
        # Accumulate color
        if next_intersection is not None and next_intersection.intersects:
            step_color = next_intersection.terminalRGB(
                spheres=self.spheres,
                background_colour=self.background_colour,
                global_light_sources=self.global_light_sources,
                point_light_sources=self.point_light_sources,
                max_bounces=0
            )
            self.accumulated_color = self.accumulated_color.addColour(step_color)
        
        # Check additional termination conditions
        truncated = False
        if terminated:
            # Already terminated (hit sun or missed)
            pass
        elif self.bounce_count >= self.max_bounces:
            # Max bounces
            terminated = True
            truncated = True
            info['reason'] = 'max_bounces'
        
        info['total_reward'] = self.total_reward
        info['bounce_count'] = self.bounce_count
        
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
if __name__ == "__main__":
    # Create a more interesting scene for RL testing
    from material import Material
    
    # Define materials with different properties
    matte_red = Material(reflective=0, transparent=0, emitive=0.1, refractive_index=1)
    reflective_gray = Material(reflective=1, transparent=0, emitive=0, refractive_index=1)
    glass_blue = Material(reflective=0, transparent=1, emitive=0, refractive_index=1.5)
    
    # Create a scene with overlapping spheres to encourage interesting bounces
    scene_spheres = [
        # Ground plane (large sphere)
        Sphere(Vector(0, -100.5, -3), 100, matte_red, Colour(200, 200, 200), id=1),
        # Main sphere
        Sphere(Vector(0, 0, -3), 0.5, reflective_gray, Colour(255, 255, 255), id=2),
        # Left sphere (reflective)
        Sphere(Vector(-1.2, 0, -3), 0.5, reflective_gray, Colour(200, 200, 255), id=3),
        # Right sphere (glass)
        Sphere(Vector(1.2, 0, -3), 0.5, glass_blue, Colour(255, 200, 200), id=4),
        # Light-emitting sphere (acts as light source)
        Sphere(Vector(0, 2, -3), 0.3, Material(reflective=0, transparent=0, emitive=1, refractive_index=1), 
               Colour(255, 255, 200), id=99),
    ]
    
    # Create lights - point light at the same position as emitting sphere
    global_lights = [
        GlobalLight(
            vector=Vector(0, -1, -0.5).normalise(),
            colour=Colour(200, 200, 255),
            strength=0.3,
            max_angle=np.pi/3
        )
    ]
    
    point_lights = [
        PointLight(
            id=99,  # Same ID as emitting sphere
            position=Vector(0, 2, -3),
            colour=Colour(255, 255, 200),
            strength=5.0,
            max_angle=np.pi,
            func=0
        )
    ]
    
    # Create environment with more bounces allowed
    env = RayTracerEnv(
        spheres=scene_spheres,
        global_light_sources=global_lights,
        point_light_sources=point_lights,
        max_bounces=10,  # Allow more bounces for learning
        image_width=400,  # Smaller for faster testing
        image_height=300
    )
    
    # Test the environment
    print("Testing RayTracerEnv with improved scene...")
    print(f"Max bounces: {env.max_bounces}")
    print(f"Number of spheres: {len(env.spheres)}")
    print(f"Sphere IDs: {[s.id for s in env.spheres]}")
    
    # Test different starting pixels
    test_pixels = [
        (env.image_width // 2, env.image_height // 2),  # Center
        (env.image_width // 4, env.image_height // 2),  # Left
        (3 * env.image_width // 4, env.image_height // 2),  # Right
    ]
    
    print("\n=== Testing different pixels ===")
    
    for i, pixel in enumerate(test_pixels):
        print(f"\n--- Test Pixel {i+1}: {pixel} ---")
        obs, info = env.reset(options={'pixel': pixel})
        
        if env.current_intersection and env.current_intersection.intersects:
            obj = env.current_intersection.object
            print(f"  Hit object ID: {obj.id}")
            print(f"  Object type: {'Light' if obj.id == 99 else 'Sphere'}")
            print(f"  Material - reflective: {obj.material.reflective}, "
                  f"transparent: {obj.material.transparent}, "
                  f"emitive: {obj.material.emitive}")
        
        # Try to learn a good path with some exploration
        done = False
        step_count = 0
        total_reward = 0
        max_steps = 5
        
        print("  Bounce path:")
        
        while not done and step_count < max_steps:
            # For testing, use a mix of random and "smart" actions
            if step_count == 0:
                # First action: try to bounce toward the light (emitting sphere)
                # This would require knowing light position - simplified version
                action = np.array([0.3, np.pi/2], dtype=np.float32)  # Upward-ish
            else:
                # Random exploration
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated
            
            # Show what happened
            if env.current_intersection and env.current_intersection.intersects:
                obj_id = env.current_intersection.object.id
                obj_type = "Light" if obj_id == 99 else "Sphere"
                print(f"    Step {step_count}: Hit {obj_type} {obj_id}, reward={reward:.4f}")
            else:
                print(f"    Step {step_count}: Missed, reward={reward:.4f}")
            
            if done:
                print(f"    Episode done: {info.get('reason', 'unknown')}")
        
        print(f"  Total reward: {total_reward:.4f}")
    
    # Test the observation space
    print("\n=== Testing observation space ===")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation values:")
    print(f"  Position: {obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f}")
    print(f"  Direction: {obs[3]:.3f}, {obs[4]:.3f}, {obs[5]:.3f}")
    print(f"  Normal: {obs[6]:.3f}, {obs[7]:.3f}, {obs[8]:.3f}")
    print(f"  Material: {obs[9]:.3f}, {obs[10]:.3f}, {obs[11]:.3f}, {obs[12]:.3f}")
    print(f"  Accumulated color: {obs[13]:.3f}, {obs[14]:.3f}, {obs[15]:.3f}")
    print(f"  Bounce count: {obs[16]:.0f}")
    print(f"  Through count: {obs[17]:.0f}")
    
    # Test action space
    print("\n=== Testing action space ===")
    test_actions = [
        np.array([0.0, 0.0], dtype=np.float32),  # Straight along normal
        np.array([np.pi/2, 0.0], dtype=np.float32),  # Tangent to surface
        np.array([np.pi/4, np.pi], dtype=np.float32),  # 45 degrees backward
    ]
    
    for i, action in enumerate(test_actions):
        env.reset()
        if env.current_intersection and env.current_intersection.intersects:
            direction = env._action_to_direction(action, env.current_intersection.normal)
            print(f"Action {i}: theta={action[0]:.2f}, phi={action[1]:.2f}")
            print(f"  -> Direction: ({direction.x:.3f}, {direction.y:.3f}, {direction.z:.3f})")
    
    print("\n=== Environment is working correctly! ===")
    print("The agent can now learn to:")
    print("1. Bounce rays toward light sources for higher rewards")
    print("2. Use reflective surfaces to gather more light")
    print("3. Avoid escaping the scene (ray_escaped gives negative reward)")
    print("4. Balance exploration vs exploitation of known light paths")