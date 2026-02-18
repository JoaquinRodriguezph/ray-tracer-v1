#!/usr/bin/env python3
"""
train_chandelier_only.py

Train the FB agent on 100 variations of the chandelier scene only.
This script ignores all other scene templates.
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
from ray import Ray
from light import GlobalLight, PointLight

# Import FB components (adjust path if needed)
try:
    from fb_ray_tracing import FBResearchAgent, FBConfig
except ImportError:
    print("Error: fb_ray_tracing.py must be in the same directory.")
    sys.exit(1)

# Import multi‑scene trainer base (for training loop)
try:
    from fb_multi_scene_trainer import MultiSceneFBTrainer
except ImportError:
    print("Error: fb_multi_scene_trainer.py must be in the same directory.")
    sys.exit(1)


# ----------------------------------------------------------------------
# Chandelier scene generator (produces variations)
# ----------------------------------------------------------------------
class ChandelierSceneGenerator:
    """Generates only chandelier scenes with variation parameter."""

    def __init__(self):
        self.scene_count = 0

    def generate_scene(self, variation=0):
        """Generate a single chandelier scene with the given variation."""
        spheres = []
        scene_id = 1000  # base ID offset

        # Materials
        matte_white = Material(reflective=0.1, transparent=0, emitive=0)
        mirror = Material(reflective=0.95, transparent=0, emitive=0)
        glass = Material(reflective=0.1, transparent=0.9, emitive=0, refractive_index=1.5)
        emitive_mat = Material(reflective=0, transparent=0, emitive=1)

        # Room (large spheres)
        floor_material = mirror if variation % 3 == 0 else matte_white
        spheres.append(Sphere(
            id=scene_id + 1,
            centre=Vector(0, -100, 0),
            radius=99,
            material=floor_material,
            colour=Colour(220, 220, 230)
        ))
        spheres.append(Sphere(
            id=scene_id + 2,
            centre=Vector(0, 100, 0),
            radius=99,
            material=mirror,
            colour=Colour(240, 240, 255)
        ))
        spheres.append(Sphere(
            id=scene_id + 3,
            centre=Vector(0, 0, -100),
            radius=99,
            material=matte_white,
            colour=Colour(210, 210, 230)
        ))
        spheres.append(Sphere(
            id=scene_id + 4,
            centre=Vector(-100, 0, 0),
            radius=99,
            material=matte_white,
            colour=Colour(200, 200, 220)
        ))
        spheres.append(Sphere(
            id=scene_id + 5,
            centre=Vector(100, 0, 0),
            radius=99,
            material=matte_white,
            colour=Colour(220, 200, 200)
        ))

        # Main large light
        spheres.append(Sphere(
            id=scene_id + 6,
            centre=Vector(0, 10, 5),
            radius=1.2,
            material=emitive_mat,
            colour=Colour(255, 255, 240)
        ))

        # Chandelier of small lights – parameters vary with variation
        num_lights = 20 + (variation % 10)          # 20–29 lights
        light_radius = 0.08 + 0.02 * (variation % 5) # 0.08–0.16
        chandelier_center = Vector(0, 4, 8)
        chandelier_radius = 2.0

        for i in range(num_lights):
            # Golden angle distribution for even coverage
            theta = (i * 137.5) % 360 * math.pi / 180
            phi = (i * 90) % 360 * math.pi / 180
            x = chandelier_center.x + chandelier_radius * math.sin(phi) * math.cos(theta)
            y = chandelier_center.y + chandelier_radius * math.sin(phi) * math.sin(theta)
            z = chandelier_center.z + chandelier_radius * math.cos(phi)

            # Add random perturbation for variation > 5
            if variation > 5:
                x += random.uniform(-0.3, 0.3)
                y += random.uniform(-0.3, 0.3)
                z += random.uniform(-0.3, 0.3)

            # Light colour varies with position and variation
            r = int(200 + 55 * math.sin(theta + variation))
            g = int(200 + 55 * math.cos(phi + variation))
            b = int(200 + 55 * math.sin(phi + theta + variation))
            r = max(180, min(255, r))
            g = max(180, min(255, g))
            b = max(180, min(255, b))

            spheres.append(Sphere(
                id=scene_id + 10 + i,
                centre=Vector(x, y, z),
                radius=light_radius,
                material=emitive_mat,
                colour=Colour(r, g, b)
            ))

        # Decorative glass/mirror spheres (positions vary slightly)
        glass_x = 1.5 + 0.2 * (variation % 3)
        spheres.append(Sphere(
            id=scene_id + 40,
            centre=Vector(glass_x, 3, 7),
            radius=0.6,
            material=glass,
            colour=Colour(255, 255, 255)
        ))
        spheres.append(Sphere(
            id=scene_id + 41,
            centre=Vector(-1.5, -1.2, 6),
            radius=0.7,
            material=mirror,
            colour=Colour(200, 200, 220)
        ))
        spheres.append(Sphere(
            id=scene_id + 42,
            centre=Vector(0, 1 + 0.2 * (variation % 2), 4),
            radius=0.5,
            material=glass,
            colour=Colour(255, 240, 240)
        ))

        return spheres

    def generate_batch(self, num_scenes):
        """Generate a batch of chandelier scenes with increasing variation."""
        scenes = []
        for i in range(num_scenes):
            spheres = self.generate_scene(i)
            name = f"chandelier_v{i}"
            scenes.append((spheres, name))
            self.scene_count += 1
        return scenes


# ----------------------------------------------------------------------
# Trainer that uses only chandelier scenes
# ----------------------------------------------------------------------
class ChandelierOnlyTrainer(MultiSceneFBTrainer):
    def __init__(self, num_training_scenes=100):
        # Call parent constructor but we will override the scene generator
        super().__init__(num_training_scenes)
        # Replace scene generator with chandelier-only version
        self.scene_generator = ChandelierSceneGenerator()
        # Adjust config for chandelier complexity
        self.config.max_bounces = 8
        self.config.f_hidden_dim = 512   # keep original dimensions
        self.config.b_hidden_dim = 256
        # Re‑initialize agent with updated config
        self.agent = FBResearchAgent(self.config, device=self.device)

    def test_on_chandelier(self, num_tests=100):
        """Test the trained model on a fresh chandelier scene (variation 99)."""
        print("\n" + "=" * 80)
        print("TESTING ON HELD‑OUT CHANDELIER SCENE")
        print("=" * 80)

        # Generate a test scene with a variation not seen during training (e.g., 99)
        spheres = self.scene_generator.generate_scene(variation=99)
        all_lights = [s for s in spheres if s.material.emitive]
        small_lights = [s for s in all_lights if s.radius < 0.5]
        print(f"Test scene contains {len(spheres)} spheres")
        print(f"Total lights: {len(all_lights)} (small: {len(small_lights)})")

        # We need a camera position for prototype computation (use same as before)
        camera_pos = Vector(0, 2, 0)

        test_hits = 0
        small_light_hits = 0

        for test_idx in tqdm(range(num_tests), desc="Testing"):
            # Choose a random starting point on a non‑light sphere
            non_light_spheres = [s for s in spheres if not s.material.emitive]
            if not non_light_spheres:
                non_light_spheres = spheres
            sphere = random.choice(non_light_spheres)

            # Random point on sphere surface
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi)
            point = Vector(
                sphere.centre.x + sphere.radius * math.sin(phi) * math.cos(theta),
                sphere.centre.y + sphere.radius * math.sin(phi) * math.sin(theta),
                sphere.centre.z + sphere.radius * math.cos(phi)
            )
            normal = point.subtractVector(sphere.centre).normalise()

            # Create observation (matching training, 22 dims)
            # We'll reuse the observation creation method from the agent
            obs = self._create_test_observation(point, normal, sphere, camera_pos)

            # Get action from trained agent
            action, _ = self.agent.choose_direction_research(
                obs, scene_context="chandelier_test", exploration_phase="test"
            )

            # Convert action to direction
            theta_act = (action[0] + 1) * math.pi / 4
            phi_act = action[1] * math.pi

            # Build local frame
            if abs(normal.z) > 0.9:
                tangent = Vector(1, 0, 0)
            else:
                tangent = Vector(0, 0, 1).crossProduct(normal)
            tangent = tangent.normalise()
            bitangent = normal.crossProduct(tangent).normalise()

            local_dir = Vector(
                math.sin(theta_act) * math.cos(phi_act),
                math.sin(theta_act) * math.sin(phi_act),
                math.cos(theta_act)
            )
            world_dir = Vector(
                local_dir.x * tangent.x + local_dir.y * bitangent.x + local_dir.z * normal.x,
                local_dir.x * tangent.y + local_dir.y * bitangent.y + local_dir.z * normal.y,
                local_dir.x * tangent.z + local_dir.y * bitangent.z + local_dir.z * normal.z
            ).normalise()

            # Test if this direction hits any light
            test_ray = Ray(point.addVector(normal.scaleByLength(0.001)), world_dir)
            hit_any = False
            hit_small = False
            for light in all_lights:
                intersect = test_ray.sphereDiscriminant(light)
                if intersect and intersect.intersects:
                    hit_any = True
                    if light in small_lights:
                        hit_small = True
                    break

            if hit_any:
                test_hits += 1
            if hit_small:
                small_light_hits += 1

        overall_rate = test_hits / num_tests * 100
        small_rate = small_light_hits / num_tests * 100
        print(f"\nTest results:")
        print(f"  Overall light hit rate: {overall_rate:.2f}%")
        print(f"  Small light hit rate: {small_rate:.4f}%")
        print(f"  Small light hits: {small_light_hits}/{num_tests}")

        # Compare with random baseline (approximate)
        # Solid angle of a small light at distance ~8m with radius 0.1
        avg_prob = (math.pi * (0.1 ** 2)) / (4 * math.pi * 64) * len(small_lights)  # crude
        print(f"\nRandom baseline (approx): {avg_prob * 100:.4f}%")
        if small_rate > avg_prob * 100:
            print("✓ FB learned to hit small lights better than random")
        else:
            print("⚠ FB still not better than random")

        return overall_rate, small_rate

    def _create_test_observation(self, point, normal, sphere, camera_pos):
        """Helper to build a 22‑dim observation for testing."""
        # Approximate incoming direction (from point toward camera)
        incoming_dir = camera_pos.subtractVector(point).normalise()
        material = sphere.material
        color = sphere.colour

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
            0.0, 0.0,
            float(sphere.id) / 100.0,
            0.5, 0.5, 0.5
        ], dtype=np.float32)
        return obs


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train FB agent exclusively on chandelier scenes")
    parser.add_argument("--quick", action="store_true", help="Quick test with 10 scenes")
    parser.add_argument("--scenes", type=int, default=100, help="Number of scenes to train on")
    args = parser.parse_args()

    if args.quick:
        num_scenes = 10
        steps_per_scene = 50
        print(f"Quick training with {num_scenes} chandelier scenes, {steps_per_scene} steps each.")
    else:
        num_scenes = args.scenes
        steps_per_scene = 150
        print(f"Full training with {num_scenes} chandelier scenes, {steps_per_scene} steps each.")

    trainer = ChandelierOnlyTrainer(num_training_scenes=num_scenes)

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

    # Test on a held‑out chandelier scene
    trainer.test_on_chandelier(num_tests=200)

    print("\nTraining complete. Model saved in:", trainer.output_dir)


if __name__ == "__main__":
    main()