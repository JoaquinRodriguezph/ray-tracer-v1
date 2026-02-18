"""
FB vs Traditional - Complex Scene Comparison
Uses the trained multi‑scene FB model and the complex scene.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime
import json
import sys
import math
import random

# Import base modules
try:
    from vector import Vector
    from colour import Colour
    from object import Sphere
    from material import Material
    from ray import Ray
    from light import GlobalLight, PointLight
    from complex_scene import create_complex_scene, create_camera_for_scene, create_lights_for_scene
    BASE_IMPORTS_OK = True
except ImportError as e:
    print(f"Error: Could not import base modules: {e}")
    BASE_IMPORTS_OK = False
    sys.exit(1)

# ----------------------------------------------------------------------
# Model architectures (exactly as in training, with z_dim=64)
# ----------------------------------------------------------------------
class EnhancedEncoder(nn.Module):
    """Encoder with hidden_dim=512, output 128 (z_dim*2)"""
    def __init__(self, obs_dim=22, z_dim=64, hidden_dim=512):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        class ResidualBlock(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim),
                    nn.ReLU(),
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim)
                )
            def forward(self, x):
                return x + self.net(x)
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(3)])
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim * 2)
        )
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        x_attn, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x = x + x_attn.squeeze(1)
        return self.output(x)


class EnhancedForwardModel(nn.Module):
    """Forward model with hidden_dim=512, 2 heads, 3 layers"""
    def __init__(self, z_dim=64, action_dim=2, hidden_dim=512, num_heads=2, num_layers=3):
        super().__init__()
        self.num_heads = num_heads
        self.input_net = nn.Sequential(
            nn.Linear(z_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.gated_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GLU(dim=-1)
            ) for _ in range(num_layers)
        ])
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, z_dim * 2)
            ) for _ in range(num_heads)
        ])
    def forward(self, z, action):
        x = torch.cat([z, action], dim=-1)
        x = self.input_net(x)
        for block in self.gated_blocks:
            x = block(x)
        predictions = []
        for head in self.heads:
            params = head(x)
            mean, log_var = params.chunk(2, dim=-1)
            predictions.append((mean, log_var))
        return predictions


class EnhancedBackwardModel(nn.Module):
    """Backward model with hidden_dim=256, 2 layers"""
    def __init__(self, z_dim=64, action_dim=2, hidden_dim=256, num_layers=2):
        super().__init__()
        self.input_net = nn.Sequential(
            nn.Linear(z_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        class ResidualBlock(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim),
                    nn.ReLU(),
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim)
                )
            def forward(self, x):
                return x + self.net(x)
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_layers)])
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_var_head = nn.Linear(hidden_dim, action_dim)
    def forward(self, z_t, z_next):
        x = torch.cat([z_t, z_next], dim=-1)
        x = self.input_net(x)
        for block in self.res_blocks:
            x = block(x)
        mean = torch.tanh(self.mean_head(x)) * 0.95
        log_var = self.log_var_head(x)
        return mean, log_var

# ----------------------------------------------------------------------
# Agent that loads the trained checkpoint and uses backward model + prototype
# ----------------------------------------------------------------------
class TrainedFBAgent:
    def __init__(self, model_path, scene_small_lights, camera_position, device='cpu'):
        self.device = device
        self.camera_position = camera_position

        self.encoder = EnhancedEncoder(obs_dim=22, z_dim=64, hidden_dim=512).to(device)
        self.forward_model = EnhancedForwardModel(z_dim=64, action_dim=2, hidden_dim=512,
                                                  num_heads=2, num_layers=3).to(device)
        self.backward_model = EnhancedBackwardModel(z_dim=64, action_dim=2, hidden_dim=256,
                                                    num_layers=2).to(device)
        self.load(model_path)

        self.light_prototype = self._compute_light_prototype(scene_small_lights)

    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        print(f"Loading model from {model_path}")

        if 'fb_learner_state' in checkpoint:
            state_dicts = checkpoint['fb_learner_state']
            self.encoder.load_state_dict(state_dicts['encoder'])
            self.forward_model.load_state_dict(state_dicts['forward_model'])
            self.backward_model.load_state_dict(state_dicts['backward_model'])
            print("✓ Loaded from fb_learner_state")
        else:
            self.encoder.load_state_dict(checkpoint['encoder_state'])
            self.forward_model.load_state_dict(checkpoint['forward_state'])
            self.backward_model.load_state_dict(checkpoint['backward_state'])
            print("✓ Loaded from direct state dicts")

        self.encoder.eval()
        self.forward_model.eval()
        self.backward_model.eval()

    def _create_observation(self, point, normal, incoming_dir, material=None, color=None, sphere_id=0):
        if material is None:
            material = Material(reflective=False)
        if color is None:
            color = Colour(0, 0, 0)

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
            float(sphere_id) / 100.0,
            0.5, 0.5, 0.5
        ], dtype=np.float32)
        return obs

    def _compute_light_prototype(self, small_lights, num_samples_per_light=5):
        all_latents = []
        for light in small_lights:
            to_camera = self.camera_position.subtractVector(light.centre).normalise()
            for _ in range(num_samples_per_light):
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                offset = Vector(
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi)
                ).scaleByLength(light.radius)
                point = light.centre.addVector(offset)
                normal = offset.normalise()
                incoming = to_camera

                obs = self._create_observation(point, normal, incoming,
                                               material=light.material,
                                               color=light.colour,
                                               sphere_id=light.id)
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    params = self.encoder(obs_t)
                    z_mean, _ = params.chunk(2, dim=-1)
                    all_latents.append(z_mean.squeeze(0).cpu().numpy())

        if not all_latents:
            print("Warning: No small lights for prototype, using zeros.")
            return np.zeros(64, dtype=np.float32)

        prototype = np.mean(all_latents, axis=0)
        norm = np.linalg.norm(prototype)
        if norm > 1e-8:
            prototype = prototype / norm
        print(f"Light prototype computed from {len(all_latents)} samples")
        return prototype

    def encode(self, observation):
        with torch.no_grad():
            obs_t = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            params = self.encoder(obs_t)
            z_mean, _ = params.chunk(2, dim=-1)
            return z_mean.squeeze(0).cpu().numpy()

    def choose_direction(self, observation):
        current_z = self.encode(observation)
        current_z_t = torch.FloatTensor(current_z).unsqueeze(0).to(self.device)
        target_z_t = torch.FloatTensor(self.light_prototype).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mean, _ = self.backward_model(current_z_t, target_z_t)
            action = mean.squeeze(0).cpu().numpy()
        return np.clip(action, -1, 1)


# ----------------------------------------------------------------------
# Traditional and FB renderers (adapted from previous)
# ----------------------------------------------------------------------
class TraditionalRenderer:
    def __init__(self):
        self.scene = []
        self.camera_position = Vector(0, 2, 0)
        self.camera_angle = None
        self.global_lights = []
        self.point_lights = []
        self.light_sources = []
        self.small_lights = []
        self.stats = {
            'total_rays': 0, 'total_intersections': 0,
            'light_hits': 0, 'small_light_hits': 0,
            'render_time': 0, 'rays_per_second': 0
        }

    def set_render_settings(self, width=200, height=150, max_bounces=3, samples_per_pixel=16):
        self.image_width = width
        self.image_height = height
        self.max_bounces = max_bounces
        self.samples_per_pixel = samples_per_pixel
        self.aspect_ratio = width / height
        self.fov = 60

    def generate_camera_ray(self, pixel_x, pixel_y, sample_x=0.5, sample_y=0.5):
        ndc_x = (pixel_x + sample_x) / self.image_width
        ndc_y = (pixel_y + sample_y) / self.image_height
        screen_x = 2.0 * ndc_x - 1.0
        screen_y = 1.0 - 2.0 * ndc_y
        screen_x *= self.aspect_ratio
        fov_rad = np.radians(self.fov)
        half_height = np.tan(fov_rad / 2)
        half_width = half_height * self.aspect_ratio
        screen_x *= half_width
        screen_y *= half_height
        ray_dir = Vector(screen_x, screen_y, -1).normalise()
        return Ray(self.camera_position, ray_dir)

    def trace_ray_traditional(self, ray, bounce_count=0):
        self.stats['total_rays'] += 1
        if bounce_count >= self.max_bounces:
            return Colour(2, 2, 5)

        nearest = None
        nearest_dist = float('inf')
        for sphere in self.scene:
            intersect = ray.sphereDiscriminant(sphere)
            if intersect and intersect.intersects:
                dist = intersect.point.distanceFrom(ray.origin)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = intersect

        if not nearest:
            return Colour(2, 2, 5)

        self.stats['total_intersections'] += 1
        sphere = nearest.object
        point = nearest.point
        normal = nearest.normal
        material = sphere.material

        if material.emitive:
            self.stats['light_hits'] += 1
            if sphere in self.small_lights:
                self.stats['small_light_hits'] += 1
            return sphere.colour

        # Direct lighting (simplified)
        direct_light = Colour(0, 0, 0)
        for light in self.light_sources:
            if light == sphere:
                continue
            to_light = light.centre.subtractVector(point)
            to_light_norm = to_light.normalise()
            cos_angle = max(0, normal.dotProduct(to_light_norm))
            if cos_angle > 0:
                dist = to_light.magnitude()
                attenuation = 1.0 / (dist ** 2)
                light_contrib = Colour(
                    int(light.colour.r * cos_angle * attenuation * 0.3),
                    int(light.colour.g * cos_angle * attenuation * 0.3),
                    int(light.colour.b * cos_angle * attenuation * 0.3)
                )
                direct_light = direct_light.addColour(light_contrib)

        # Indirect bounce
        indirect_light = Colour(0, 0, 0)
        if material.reflective > 0.9:  # treat as mirror
            reflect_dir = ray.D.reflectInVector(normal)
            reflect_ray = Ray(point.addVector(normal.scaleByLength(0.001)), reflect_dir)
            indirect_light = self.trace_ray_traditional(reflect_ray, bounce_count + 1)
        else:
            # Diffuse bounce (cosine-weighted)
            r1, r2 = np.random.random(), np.random.random()
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
            bounce_dir = Vector(
                local_dir.x * tangent.x + local_dir.y * bitangent.x + local_dir.z * normal.x,
                local_dir.x * tangent.y + local_dir.y * bitangent.y + local_dir.z * normal.y,
                local_dir.x * tangent.z + local_dir.y * bitangent.z + local_dir.z * normal.z
            ).normalise()
            bounce_ray = Ray(point.addVector(normal.scaleByLength(0.001)), bounce_dir)
            indirect_light = self.trace_ray_traditional(bounce_ray, bounce_count + 1)

        total_light = Colour(
            min(255, direct_light.r + indirect_light.r),
            min(255, direct_light.g + indirect_light.g),
            min(255, direct_light.b + indirect_light.b)
        )
        final_color = Colour(
            int(sphere.colour.r * (total_light.r / 255.0)),
            int(sphere.colour.g * (total_light.g / 255.0)),
            int(sphere.colour.b * (total_light.b / 255.0))
        )
        return final_color

    def render(self, width=200, height=150, samples_per_pixel=4, max_bounces=3):
        self.set_render_settings(width, height, max_bounces, samples_per_pixel)
        print(f"\nTraditional - Rendering {width}x{height}, {samples_per_pixel} spp")
        self.stats = {k: 0 for k in self.stats}
        start_time = time.time()
        image = np.zeros((height, width, 3), dtype=np.float32)

        for y in tqdm(range(height), desc="Traditional"):
            for x in range(width):
                pixel_color = Colour(0, 0, 0)
                for sample in range(samples_per_pixel):
                    jitter_x = np.random.random() - 0.5
                    jitter_y = np.random.random() - 0.5
                    ray = self.generate_camera_ray(x, y, 0.5 + jitter_x, 0.5 + jitter_y)
                    sample_color = self.trace_ray_traditional(ray)
                    pixel_color = pixel_color.addColour(sample_color)
                pixel_color = Colour(
                    pixel_color.r // samples_per_pixel,
                    pixel_color.g // samples_per_pixel,
                    pixel_color.b // samples_per_pixel
                )
                image[y, x] = [
                    min(1.0, pixel_color.r / 255.0),
                    min(1.0, pixel_color.g / 255.0),
                    min(1.0, pixel_color.b / 255.0)
                ]

        render_time = time.time() - start_time
        self.stats['render_time'] = render_time
        if render_time > 0:
            self.stats['rays_per_second'] = self.stats['total_rays'] / render_time
        return image


class WorkingFBRenderer:
    def __init__(self, model_path=None, scene_small_lights=None, camera_position=None):
        self.scene = []
        self.camera_position = camera_position if camera_position is not None else Vector(0, 2, 0)
        self.camera_angle = None
        self.global_lights = []
        self.point_lights = []
        self.light_sources = []
        self.small_lights = scene_small_lights if scene_small_lights is not None else []

        self.fb_agent = TrainedFBAgent(model_path, self.small_lights, self.camera_position, device='cpu') if model_path else None
        self.fb_loaded = model_path is not None
        self.fb_usage_prob = 1.0 if self.fb_loaded else 0.0

        self.stats = {
            'total_rays': 0, 'total_intersections': 0,
            'light_hits': 0, 'small_light_hits': 0,
            'fb_used': 0, 'fb_success': 0,
            'render_time': 0, 'rays_per_second': 0
        }

    def set_render_settings(self, width=200, height=150, max_bounces=3, samples_per_pixel=4):
        self.image_width = width
        self.image_height = height
        self.max_bounces = max_bounces
        self.samples_per_pixel = samples_per_pixel
        self.aspect_ratio = width / height
        self.fov = 60

    def generate_camera_ray(self, pixel_x, pixel_y, sample_x=0.5, sample_y=0.5):
        ndc_x = (pixel_x + sample_x) / self.image_width
        ndc_y = (pixel_y + sample_y) / self.image_height
        screen_x = 2.0 * ndc_x - 1.0
        screen_y = 1.0 - 2.0 * ndc_y
        screen_x *= self.aspect_ratio
        fov_rad = np.radians(self.fov)
        half_height = np.tan(fov_rad / 2)
        half_width = half_height * self.aspect_ratio
        screen_x *= half_width
        screen_y *= half_height
        ray_dir = Vector(screen_x, screen_y, -1).normalise()
        return Ray(self.camera_position, ray_dir)

    def create_observation(self, point, normal, ray_dir, bounce_count, color, material, sphere_id):
        is_reflective = float(getattr(material, 'reflective', False))
        is_transparent = float(getattr(material, 'transparent', False))
        is_emitive = float(getattr(material, 'emitive', False))
        refractive_index = float(getattr(material, 'refractive_index', 1.0))
        obs = np.array([
            point.x, point.y, point.z,
            ray_dir.x, ray_dir.y, ray_dir.z,
            normal.x, normal.y, normal.z,
            is_reflective, is_transparent, is_emitive, refractive_index,
            color.r / 255.0, color.g / 255.0, color.b / 255.0,
            float(bounce_count) / self.max_bounces,
            0.0,
            float(sphere_id if sphere_id is not None else 0) / 100.0,
            0.5, 0.5, 0.5
        ], dtype=np.float32)
        return obs

    def trace_ray_fb(self, ray, bounce_count=0, accumulated_color=Colour(0, 0, 0)):
        self.stats['total_rays'] += 1
        if bounce_count >= self.max_bounces:
            return Colour(2, 2, 5)

        nearest = None
        nearest_dist = float('inf')
        for sphere in self.scene:
            intersect = ray.sphereDiscriminant(sphere)
            if intersect and intersect.intersects:
                dist = intersect.point.distanceFrom(ray.origin)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = intersect

        if not nearest:
            return Colour(2, 2, 5)

        self.stats['total_intersections'] += 1
        sphere = nearest.object
        point = nearest.point
        normal = nearest.normal
        material = sphere.material

        if material.emitive:
            self.stats['light_hits'] += 1
            if sphere in self.small_lights:
                self.stats['small_light_hits'] += 1
            return sphere.colour

        direct_light = Colour(0, 0, 0)
        for light in self.light_sources:
            if light == sphere:
                continue
            to_light = light.centre.subtractVector(point)
            to_light_norm = to_light.normalise()
            cos_angle = max(0, normal.dotProduct(to_light_norm))
            if cos_angle > 0:
                dist = to_light.magnitude()
                attenuation = 1.0 / (dist ** 2)
                light_contrib = Colour(
                    int(light.colour.r * cos_angle * attenuation * 0.3),
                    int(light.colour.g * cos_angle * attenuation * 0.3),
                    int(light.colour.b * cos_angle * attenuation * 0.3)
                )
                direct_light = direct_light.addColour(light_contrib)

        indirect_light = Colour(0, 0, 0)

        if material.reflective > 0.9:
            reflect_dir = ray.D.reflectInVector(normal)
            reflect_ray = Ray(point.addVector(normal.scaleByLength(0.001)), reflect_dir)
            indirect_light = self.trace_ray_fb(reflect_ray, bounce_count + 1, accumulated_color)
        else:
            use_fb = (self.fb_loaded and np.random.random() < self.fb_usage_prob)
            if use_fb:
                self.stats['fb_used'] += 1
                obs = self.create_observation(
                    point, normal, ray.D, bounce_count,
                    accumulated_color, material, sphere.id
                )
                action = self.fb_agent.choose_direction(obs)
                self.stats['fb_success'] += 1
                theta = (action[0] + 1) * np.pi/4
                phi = action[1] * np.pi
                local_dir = Vector(
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                )
                if abs(normal.z) > 0.9:
                    tangent = Vector(1, 0, 0)
                else:
                    tangent = Vector(0, 0, 1).crossProduct(normal)
                tangent = tangent.normalise()
                bitangent = normal.crossProduct(tangent).normalise()
                bounce_dir = Vector(
                    local_dir.x * tangent.x + local_dir.y * bitangent.x + local_dir.z * normal.x,
                    local_dir.x * tangent.y + local_dir.y * bitangent.y + local_dir.z * normal.y,
                    local_dir.x * tangent.z + local_dir.y * bitangent.z + local_dir.z * normal.z
                ).normalise()
            else:
                r1, r2 = np.random.random(), np.random.random()
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
                bounce_dir = Vector(
                    local_dir.x * tangent.x + local_dir.y * bitangent.x + local_dir.z * normal.x,
                    local_dir.x * tangent.y + local_dir.y * bitangent.y + local_dir.z * normal.y,
                    local_dir.x * tangent.z + local_dir.y * bitangent.z + local_dir.z * normal.z
                ).normalise()

            bounce_ray = Ray(point.addVector(normal.scaleByLength(0.001)), bounce_dir)
            indirect_light = self.trace_ray_fb(bounce_ray, bounce_count + 1, accumulated_color)

        total_light = Colour(
            min(255, direct_light.r + indirect_light.r),
            min(255, direct_light.g + indirect_light.g),
            min(255, direct_light.b + indirect_light.b)
        )
        final_color = Colour(
            int(sphere.colour.r * (total_light.r / 255.0)),
            int(sphere.colour.g * (total_light.g / 255.0)),
            int(sphere.colour.b * (total_light.b / 255.0))
        )
        return final_color

    def render(self, width=200, height=150, samples_per_pixel=4, max_bounces=3):
        self.set_render_settings(width, height, max_bounces, samples_per_pixel)
        print(f"\nFB-Accelerated - Rendering {width}x{height}, {samples_per_pixel} spp")
        print(f"FB loaded: {self.fb_loaded}, Usage: {self.fb_usage_prob*100:.0f}%")
        self.stats = {k: 0 for k in self.stats}
        start_time = time.time()
        image = np.zeros((height, width, 3), dtype=np.float32)

        for y in tqdm(range(height), desc="FB-Accelerated"):
            for x in range(width):
                pixel_color = Colour(0, 0, 0)
                for sample in range(samples_per_pixel):
                    jitter_x = np.random.random() - 0.5
                    jitter_y = np.random.random() - 0.5
                    ray = self.generate_camera_ray(x, y, 0.5 + jitter_x, 0.5 + jitter_y)
                    sample_color = self.trace_ray_fb(ray)
                    pixel_color = pixel_color.addColour(sample_color)
                pixel_color = Colour(
                    pixel_color.r // samples_per_pixel,
                    pixel_color.g // samples_per_pixel,
                    pixel_color.b // samples_per_pixel
                )
                image[y, x] = [
                    min(1.0, pixel_color.r / 255.0),
                    min(1.0, pixel_color.g / 255.0),
                    min(1.0, pixel_color.b / 255.0)
                ]

        render_time = time.time() - start_time
        self.stats['render_time'] = render_time
        if render_time > 0:
            self.stats['rays_per_second'] = self.stats['total_rays'] / render_time
        return image


# ----------------------------------------------------------------------
# Main comparison
# ----------------------------------------------------------------------
def find_latest_model():
    candidates = list(Path(".").glob("fb_multi_scene_training_*/fb_multi_scene_final.pth"))
    if candidates:
        candidates.sort(key=lambda p: p.parent.stat().st_mtime, reverse=True)
        return candidates[0]
    return None

def main():
    print("="*80)
    print("FB vs TRADITIONAL - COMPLEX SCENE COMPARISON")
    print("="*80)

    model_path = find_latest_model()
    if model_path:
        print(f"Found FB model: {model_path}")
    else:
        print("Warning: No trained FB model found. FB renderer will fall back to random sampling.")
        model_path = None

    # Generate complex scene
    print("Generating complex scene...")
    scene_spheres = create_complex_scene()
    light_sources = [s for s in scene_spheres if s.material.emitive]
    small_lights = [s for s in light_sources if s.radius < 0.5]
    print(f"Scene contains {len(scene_spheres)} spheres")
    print(f"Light sources: {len(light_sources)} (small: {len(small_lights)})")

    # Camera position (from original scene)
    camera_pos, _ = create_camera_for_scene()
    global_lights, point_lights = create_lights_for_scene()

    # Create renderers
    traditional = TraditionalRenderer()
    traditional.scene = scene_spheres
    traditional.light_sources = light_sources
    traditional.small_lights = small_lights
    traditional.camera_position = camera_pos
    traditional.global_lights = global_lights
    traditional.point_lights = point_lights

    fb = WorkingFBRenderer(model_path, small_lights, camera_pos)
    fb.scene = scene_spheres
    fb.light_sources = light_sources
    fb.small_lights = small_lights
    fb.camera_position = camera_pos
    fb.global_lights = global_lights
    fb.point_lights = point_lights

    # Rendering parameters (lower for quick test)
    width, height = 200, 100
    samples_per_pixel = 8
    max_bounces = 8

    print(f"\nTest Configuration:")
    print(f"  Image: {width}x{height}")
    print(f"  Samples per pixel: {samples_per_pixel}")
    print(f"  Max bounces: {max_bounces}")

    # Traditional render
    print(f"\n{'='*60}")
    print("1. TRADITIONAL RAY TRACING")
    print(f"{'='*60}")
    trad_image = traditional.render(width, height, samples_per_pixel, max_bounces)

    # FB render
    print(f"\n{'='*60}")
    print("2. FB-ACCELERATED RAY TRACING")
    print(f"{'='*60}")
    fb_image = fb.render(width, height, samples_per_pixel, max_bounces)

    # Comparison stats
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")

    trad_time = traditional.stats['render_time']
    fb_time = fb.stats['render_time']
    trad_rays = traditional.stats['total_rays']
    fb_rays = fb.stats['total_rays']
    trad_small_hits = traditional.stats['small_light_hits']
    fb_small_hits = fb.stats['small_light_hits']

    print(f"\nRender Time:")
    print(f"  Traditional: {trad_time:.1f}s")
    print(f"  FB: {fb_time:.1f}s")
    if trad_time > 0 and fb_time > 0:
        speedup = trad_time / fb_time
        print(f"  Speedup: {speedup:.2f}x {'(FB faster)' if speedup > 1 else '(Traditional faster)'}")

    print(f"\nTotal Rays:")
    print(f"  Traditional: {trad_rays:,}")
    print(f"  FB: {fb_rays:,}")
    if trad_rays > 0:
        efficiency = fb_rays / trad_rays
        print(f"  FB used {efficiency:.2f}x rays compared to traditional")

    print(f"\nSmall Light Hits:")
    trad_rate = trad_small_hits / max(1, trad_rays) * 100
    fb_rate = fb_small_hits / max(1, fb_rays) * 100
    print(f"  Traditional: {trad_small_hits:,} ({trad_rate:.4f}%)")
    print(f"  FB: {fb_small_hits:,} ({fb_rate:.4f}%)")
    if trad_small_hits > 0:
        improvement = fb_small_hits / trad_small_hits
        print(f"  Small light improvement: {improvement:.2f}x")

    if fb.stats['fb_used'] > 0:
        fb_success_rate = fb.stats['fb_success'] / fb.stats['fb_used'] * 100
        print(f"\nFB success rate: {fb_success_rate:.1f}%")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"./complex_comparison_{timestamp}")
    out_dir.mkdir(exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(np.clip(trad_image, 0, 1))
    ax1.set_title(f'Traditional\n{trad_time:.1f}s, {trad_rays:,} rays')
    ax1.axis('off')
    ax2.imshow(np.clip(fb_image, 0, 1))
    ax2.set_title(f'FB-Accelerated\n{fb_time:.1f}s, {fb_rays:,} rays')
    ax2.axis('off')
    diff = np.abs(fb_image - trad_image)
    ax3.imshow(np.clip(diff * 3, 0, 1), cmap='hot')
    ax3.set_title('Difference (Enhanced 3x)')
    ax3.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / "comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    stats = {
        'traditional': traditional.stats,
        'fb': fb.stats,
        'comparison': {
            'speedup': trad_time / fb_time if fb_time > 0 else 0,
            'ray_efficiency': fb_rays / trad_rays if trad_rays > 0 else 0,
            'small_light_improvement': fb_small_hits / trad_small_hits if trad_small_hits > 0 else 0
        }
    }
    with open(out_dir / "statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print("RESULTS SAVED")
    print(f"{'='*60}")
    print(f"Images saved to: {out_dir}/comparison.png")
    print(f"Statistics saved to: {out_dir}/statistics.json")

    speedup = stats['comparison']['speedup']
    if speedup > 1.1:
        print(f"\n✓ SUCCESS: FB is {speedup:.2f}x faster!")
    elif speedup > 0.9:
        print(f"\n⚠ MIXED: Similar performance")
    else:
        print(f"\n✗ Traditional is {1/speedup:.2f}x faster")

    return stats

if __name__ == "__main__":
    main()