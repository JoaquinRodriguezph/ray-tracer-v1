#!/usr/bin/env python3
"""
train_complex_raytraced.py

Train FB agent on the complex scene using real ray‑traced experiences.
Generates paths by random sampling, records transitions, and trains
forward/backward models on actual next observations.
"""

import numpy as np
import torch
import math
import random
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import argparse
import sys

# Import base classes
from vector import Vector
from colour import Colour
from object import Sphere
from material import Material
from ray import Ray, Intersection
from light import GlobalLight, PointLight

# Import FB components
try:
    from fb_ray_tracing import FBResearchAgent, FBConfig
except ImportError:
    print("Error: fb_ray_tracing.py must be in the same directory.")
    sys.exit(1)

# Import multi‑scene trainer base
try:
    from fb_multi_scene_trainer import MultiSceneFBTrainer
except ImportError:
    print("Error: fb_multi_scene_trainer.py must be in the same directory.")
    sys.exit(1)

# Import the original complex scene creator and its variations
try:
    from complex_scene import create_complex_scene, create_camera_for_scene, create_lights_for_scene
except ImportError:
    print("Error: complex_scene.py must be in the same directory.")
    sys.exit(1)


# ----------------------------------------------------------------------
# Helper functions for ray tracing and observation creation
# ----------------------------------------------------------------------
def random_point_on_sphere(sphere):
    """Return a random point on sphere surface and the outward normal."""
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(0, math.pi)
    offset = Vector(
        math.sin(phi) * math.cos(theta),
        math.sin(phi) * math.sin(theta),
        math.cos(phi)
    ).scaleByLength(sphere.radius)
    point = sphere.centre.addVector(offset)
    normal = offset.normalise()
    return point, normal


def sample_cosine_weighted_direction(normal):
    """Sample a direction in the hemisphere oriented by normal, cosine‑weighted."""
    r1 = random.random()
    r2 = random.random()
    theta = math.acos(math.sqrt(r1))          # cos(theta) = sqrt(r1)
    phi = 2 * math.pi * r2

    # Local coordinates: z is along normal
    local_dir = Vector(
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta)
    )

    # Build orthonormal basis
    if abs(normal.z) < 0.999:
        tangent = Vector(0, 0, 1).crossProduct(normal).normalise()
    else:
        tangent = Vector(1, 0, 0).crossProduct(normal).normalise()
    bitangent = normal.crossProduct(tangent).normalise()

    # Transform to world space
    world_dir = Vector(
        local_dir.x * tangent.x + local_dir.y * bitangent.x + local_dir.z * normal.x,
        local_dir.x * tangent.y + local_dir.y * bitangent.y + local_dir.z * normal.y,
        local_dir.x * tangent.z + local_dir.y * bitangent.z + local_dir.z * normal.z
    ).normalise()
    return world_dir


def direction_to_action(direction, normal):
    """
    Convert world direction to action (theta, phi) in [-1,1]².
    Theta is angle from normal, phi is azimuth around normal.
    """
    # Build basis
    if abs(normal.z) < 0.999:
        tangent = Vector(0, 0, 1).crossProduct(normal).normalise()
    else:
        tangent = Vector(1, 0, 0).crossProduct(normal).normalise()
    bitangent = normal.crossProduct(tangent).normalise()

    # Project direction onto local coordinates
    local_x = direction.dotProduct(tangent)
    local_y = direction.dotProduct(bitangent)
    local_z = direction.dotProduct(normal)

    # Spherical coordinates
    theta = math.acos(max(-1, min(1, local_z)))          # [0, π]
    # Clamp theta to [0, π/2] because direction must be in hemisphere
    if theta > math.pi / 2:
        theta = math.pi / 2
    phi = math.atan2(local_y, local_x)                   # [-π, π]

    # Normalize to [-1, 1]
    action_theta = (theta / (math.pi / 2)) * 2 - 1       # map [0, π/2] to [-1, 1]
    action_phi = phi / math.pi                            # map [-π, π] to [-1, 1]
    return np.array([action_theta, action_phi], dtype=np.float32)


def create_observation(point, normal, incoming_dir, bounce_count, color, material, sphere_id, max_bounces):
    """Build 22‑dim observation (matches training)."""
    is_reflective = float(getattr(material, 'reflective', False))
    is_transparent = float(getattr(material, 'transparent', False))
    is_emitive = float(getattr(material, 'emitive', False))
    refractive_index = float(getattr(material, 'refractive_index', 1.0))

    obs = np.array([
        point.x, point.y, point.z,
        incoming_dir.x, incoming_dir.y, incoming_dir.z,
        normal.x, normal.y, normal.z,
        is_reflective, is_transparent, is_emitive, refractive_index,
        color.r / 255.0, color.g / 255.0, color.b / 255.0,
        float(bounce_count) / max_bounces,
        0.0,
        float(sphere_id) / 100.0,
        0.5, 0.5, 0.5
    ], dtype=np.float32)
    return obs


def nearest_intersection(ray, spheres, exclude_ids=None):
    """Return the nearest intersection (Intersection object) or None."""
    best = None
    best_dist = float('inf')
    for s in spheres:
        if exclude_ids and s.id in exclude_ids:
            continue
        inter = ray.sphereDiscriminant(s)
        if inter and inter.intersects:
            dist = inter.point.distanceFrom(ray.origin)
            if dist < best_dist:
                best_dist = dist
                best = inter
    return best


# ----------------------------------------------------------------------
# Scene generator with variations (same as before, but we only need base scene)
# ----------------------------------------------------------------------
class ComplexSceneGenerator:
    """Generates variations of the complex scene (same as previous)."""
    def __init__(self):
        self.scene_count = 0

    def generate_scene(self, variation=0):
        # Start with base complex scene
        spheres = create_complex_scene()
        rng = random.Random(variation)

        # Perturb light positions and colours
        for sphere in spheres:
            if sphere.material.emitive:
                dx = rng.uniform(-0.3, 0.3)
                dy = rng.uniform(-0.3, 0.3)
                dz = rng.uniform(-0.3, 0.3)
                sphere.centre = Vector(
                    sphere.centre.x + dx,
                    sphere.centre.y + dy,
                    sphere.centre.z + dz
                )
                r = max(180, min(255, sphere.colour.r + rng.randint(-20, 20)))
                g = max(180, min(255, sphere.colour.g + rng.randint(-20, 20)))
                b = max(180, min(255, sphere.colour.b + rng.randint(-20, 20)))
                sphere.colour = Colour(r, g, b)

        # Perturb non‑light objects (except large walls)
        for sphere in spheres:
            if not sphere.material.emitive and sphere.id not in [1,2,3,4,5,6]:
                dx = rng.uniform(-0.2, 0.2)
                dy = rng.uniform(-0.2, 0.2)
                dz = rng.uniform(-0.2, 0.2)
                sphere.centre = Vector(
                    sphere.centre.x + dx,
                    sphere.centre.y + dy,
                    sphere.centre.z + dz
                )
                r = max(100, min(255, sphere.colour.r + rng.randint(-15, 15)))
                g = max(100, min(255, sphere.colour.g + rng.randint(-15, 15)))
                b = max(100, min(255, sphere.colour.b + rng.randint(-15, 15)))
                sphere.colour = Colour(r, g, b)

        # Optionally add/remove a small light
        if variation % 5 == 0:
            small_light = Sphere(
                id=999 + variation,
                centre=Vector(
                    rng.uniform(-2, 2),
                    rng.uniform(-1, 3),
                    rng.uniform(0, 5)
                ),
                radius=0.15,
                material=Material(reflective=0, transparent=0, emitive=1),
                colour=Colour(255, 240, 200)
            )
            spheres.append(small_light)
        elif variation % 7 == 0:
            small_lights = [s for s in spheres if s.material.emitive and s.radius < 0.5]
            if small_lights:
                to_remove = rng.choice(small_lights)
                spheres.remove(to_remove)

        return spheres

    def generate_batch(self, num_scenes):
        scenes = []
        for i in range(num_scenes):
            spheres = self.generate_scene(i)
            name = f"complex_v{i}"
            scenes.append((spheres, name))
            self.scene_count += 1
        return scenes


# ----------------------------------------------------------------------
# Trainer that uses real ray‑traced experiences
# ----------------------------------------------------------------------
class RayTracedComplexTrainer(MultiSceneFBTrainer):
    def __init__(self, num_training_scenes=100):
        super().__init__(num_training_scenes)
        self.scene_generator = ComplexSceneGenerator()
        self.config.max_bounces = 8
        self.agent = FBResearchAgent(self.config, device=self.device)
        # For observation creation, we need max_bounces
        self.max_bounces = self.config.max_bounces

    def generate_trajectory(self, spheres, max_steps=None):
        """
        Run a single random ray path starting from a random point on a random non‑light sphere.
        Returns a list of transitions (obs, action, next_obs, reward, hit_light) and a flag
        indicating whether a light was hit.
        """
        if max_steps is None:
            max_steps = self.max_bounces

        # Choose a random non‑light sphere
        non_light = [s for s in spheres if not s.material.emitive]
        if not non_light:
            return [], False
        sphere = random.choice(non_light)

        # Random starting point and normal
        point, normal = random_point_on_sphere(sphere)

        # Random initial incoming direction (cosine‑weighted)
        incoming_dir = sample_cosine_weighted_direction(normal)
        # Initial accumulated color (start black)
        acc_color = Colour(0, 0, 0)

        # Create initial observation (bounce_count = 0)
        current_obs = create_observation(point, normal, incoming_dir, 0, acc_color,
                                        sphere.material, sphere.id, self.max_bounces)

        current_point = point
        current_normal = normal
        current_sphere = sphere
        current_acc_color = acc_color
        bounce = 0
        transitions = []
        hit_light = False

        print(f"    Starting trajectory at point ({point.x:.2f}, {point.y:.2f}, {point.z:.2f})")

        while bounce < max_steps and not hit_light:
            print(f"    Step {bounce}: current point ({current_point.x:.2f}, {current_point.y:.2f}, {current_point.z:.2f})")

            # Sample action (direction) for next step
            next_dir = sample_cosine_weighted_direction(current_normal)
            action = direction_to_action(next_dir, current_normal)

            # Create ray from current point with a small offset along normal
            offset = current_normal.scaleByLength(0.001)
            ray_origin = current_point.addVector(offset)
            ray = Ray(ray_origin, next_dir)

            # Find next intersection (exclude current sphere)
            inter = nearest_intersection(ray, spheres, exclude_ids=[current_sphere.id])
            if not inter:
                print(f"      No intersection – ray escaped")
                break  # ray escaped

            next_obj = inter.object
            next_point = inter.point
            next_normal = inter.normal
            dist = next_point.distanceFrom(ray_origin)
            print(f"      Hit object {next_obj.id} at distance {dist:.4f}")

            # Determine reward and next observation
            if next_obj.material.emitive:
                hit_light = True
                reward = 1.0
                next_obs = create_observation(next_point, next_normal, ray.D, bounce+1,
                                            next_obj.colour, next_obj.material,
                                            next_obj.id, self.max_bounces)
                print(f"      *** LIGHT HIT! ***")
            else:
                reward = 0.0
                next_obs = create_observation(next_point, next_normal, ray.D, bounce+1,
                                            current_acc_color, next_obj.material,
                                            next_obj.id, self.max_bounces)

            # Store transition
            transitions.append((current_obs.copy(), action.copy(), next_obs.copy(), reward, hit_light))

            # Update for next step
            if not hit_light:
                current_point = next_point
                current_normal = next_normal
                current_sphere = next_obj
                current_obs = next_obs
                bounce += 1
            else:
                break

            # Safety: if bounce hasn't increased and we're not hitting light, break
            if bounce == 0 and not hit_light:
                print("      Warning: bounce not incrementing, breaking to avoid infinite loop")
                break

        print(f"    Trajectory ended after {bounce} steps, hit_light={hit_light}")
        return transitions, hit_light

    def generate_training_experience(self, spheres, scene_name, num_episodes=50):
        print(f"  Generating {num_episodes} episodes for {scene_name}...")
        episodes_generated = 0
        light_hits = 0
        
        for _ in tqdm(range(num_episodes), desc=f"Episodes"):
            transitions, hit = self.generate_trajectory(spheres, max_steps=self.max_bounces)
            for obs, action, next_obs, reward, hit_light in transitions:
                self.agent.record_success(obs, action, next_obs, reward, hit_light)
            if hit:
                light_hits += 1
            episodes_generated += 1
        
        hit_rate = light_hits / max(1, episodes_generated)
        return episodes_generated, hit_rate

    # Optionally, we might want to keep the parent's test_on_complex method or override it.
    # We'll keep the parent's test_on_complex (which uses the simplified evaluation) for consistency,
    # but we can also add a real ray‑traced test later.

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train FB agent on complex scene using real ray‑traced experiences")
    parser.add_argument("--quick", action="store_true", help="Quick test with 10 scenes")
    parser.add_argument("--scenes", type=int, default=100, help="Number of scenes to train on")
    args = parser.parse_args()

    if args.quick:
        num_scenes = 10
        steps_per_scene = 50
        print(f"Quick training with {num_scenes} complex scenes, {steps_per_scene} steps each.")
    else:
        num_scenes = args.scenes
        steps_per_scene = 150
        print(f"Full training with {num_scenes} complex scenes, {steps_per_scene} steps each.")

    trainer = RayTracedComplexTrainer(num_training_scenes=num_scenes)

    response = input("Start training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return

    # Run training
    trainer.run_training(
        num_scenes=num_scenes,
        scenes_per_batch=20,
        training_steps_per_scene=steps_per_scene
    )

    # Optionally test on a held‑out complex scene using the simplified metric (as before)
    trainer.test_on_complex(num_tests=200)

    print("\nTraining complete. Model saved in:", trainer.output_dir)


if __name__ == "__main__":
    main()