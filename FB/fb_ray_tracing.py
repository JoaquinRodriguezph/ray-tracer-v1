"""
Enhanced Forward-Backward Representation Learning for Ray Tracing
Direct comparison to RL approach in ray_tracer_rl_complete.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import json
from pathlib import Path

from vector import Vector
from colour import Colour
from object import Sphere
from material import Material
from ray import Ray, Intersection


@dataclass
class FBConfig:
    """Configuration for Forward-Backward learning"""
    # Architecture
    z_dim: int = 64                # Latent dimension
    f_hidden_dim: int = 512       # Forward model hidden size
    b_hidden_dim: int = 256       # Backward model hidden size
    num_forward_heads: int = 3    # Multiple forward predictions
    num_layers: int = 2           # Number of hidden layers
    
    # Training
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_capacity: int = 100000
    update_freq: int = 100
    target_update_freq: int = 1000
    
    # Loss weights
    fb_weight: float = 1.0        # Forward-backward consistency
    contrastive_weight: float = 0.5  # Contrastive learning
    predictive_weight: float = 0.3   # Self-supervised prediction
    norm_weight: float = 0.1      # Norm regularization
    diversity_weight: float = 0.05  # Encourage diverse predictions
    
    # Exploration
    noise_scale: float = 0.1
    min_noise: float = 0.01
    noise_decay: float = 0.995
    
    # Ray tracing specific
    max_bounces: int = 8
    samples_per_pixel: int = 1
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)


class VectorQuantizer(nn.Module):
    """Vector quantization layer for discrete representations"""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1/num_embeddings, 1/num_embeddings)
    
    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, x):
        return x + self.net(x)


class EnhancedEncoder(nn.Module):
    """Enhanced encoder with residual connections and attention"""
    def __init__(self, obs_dim: int, z_dim: int, hidden_dim: int, use_vq: bool = False):
        super().__init__()
        self.use_vq = use_vq
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(3)
        ])
        
        # Self-attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Output layers
        if use_vq:
            self.pre_vq = nn.Linear(hidden_dim, z_dim)
            self.vq = VectorQuantizer(num_embeddings=512, embedding_dim=z_dim)
            self.post_vq = nn.Linear(z_dim, z_dim * 2)
        else:
            self.output = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, z_dim * 2)
            )
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        x = self.input_proj(obs)
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Self-attention (treat batch as sequence)
        x_attn, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x = x + x_attn.squeeze(1)
        
        if self.use_vq:
            # Vector quantization path
            pre_vq = self.pre_vq(x)
            quantized, vq_loss, indices = self.vq(pre_vq)
            params = self.post_vq(quantized)
            mean, log_var = params.chunk(2, dim=-1)
            return mean, log_var, {'vq_loss': vq_loss, 'indices': indices}
        else:
            params = self.output(x)
            mean, log_var = params.chunk(2, dim=-1)
            return mean, log_var, {}


class EnhancedForwardModel(nn.Module):
    """Forward model with uncertainty estimation"""
    def __init__(self, z_dim: int, action_dim: int, hidden_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.num_heads = num_heads
        
        # Input processing
        self.input_net = nn.Sequential(
            nn.Linear(z_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Gated blocks
        self.gated_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GLU(dim=-1)
            ) for _ in range(num_layers)
        ])
        
        # Multiple prediction heads with different initializations
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, z_dim * 2)  # Predict mean and variance
            ) for _ in range(num_heads)
        ])
        
    def forward(self, z: torch.Tensor, action: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
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
    """Backward model with probabilistic outputs"""
    def __init__(self, z_dim: int, action_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        
        self.input_net = nn.Sequential(
            nn.Linear(z_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        # Predict mean and log variance of action distribution
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_var_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z_t, z_next], dim=-1)
        x = self.input_net(x)
        
        for block in self.res_blocks:
            x = block(x)
        
        mean = torch.tanh(self.mean_head(x)) * 0.95  # Keep in [-0.95, 0.95]
        log_var = self.log_var_head(x)
        
        return mean, log_var


class FBLatentPlanner:
    """Latent space planner for goal-directed behavior"""
    def __init__(self, z_dim: int, num_subgoals: int = 3):
        self.z_dim = z_dim
        self.num_subgoals = num_subgoals
        
        # Store successful paths
        self.successful_paths = deque(maxlen=100)
        self.subgoal_memory = defaultdict(list)
        
    def plan_to_light(self, current_z: np.ndarray, light_z: np.ndarray, 
                  backward_model: EnhancedBackwardModel, device: str = 'cpu') -> List[np.ndarray]:
        """Plan a path to light in latent space"""
        
        # Try direct planning first
        z_t_tensor = torch.FloatTensor(current_z).unsqueeze(0).to(device)
        z_light_tensor = torch.FloatTensor(light_z).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mean, log_var = backward_model(z_t_tensor, z_light_tensor)
            action = mean + torch.exp(0.5 * log_var) * torch.randn_like(mean)
            action = action.squeeze(0).detach().cpu().numpy()
        
        # Check if this looks reasonable
        if np.linalg.norm(action) < 2.0:  # Not too extreme
            return [action]
        
        # If direct planning fails, try subgoal decomposition
        subgoals = self._generate_subgoals(current_z, light_z)
        actions = []  # Initialize actions list
        
        for subgoal in subgoals:
            z_subgoal_tensor = torch.FloatTensor(subgoal).unsqueeze(0).to(device)
            mean, log_var = backward_model(z_t_tensor, z_subgoal_tensor)
            action = mean + torch.exp(0.5 * log_var) * torch.randn_like(mean)
            actions.append(action.squeeze(0).detach().cpu().numpy())
            z_t_tensor = z_subgoal_tensor
        
        return actions
    
    def _generate_subgoals(self, start_z: np.ndarray, goal_z: np.ndarray) -> List[np.ndarray]:
        """Generate intermediate subgoals"""
        subgoals = []
        
        # Linear interpolation in latent space
        for i in range(1, self.num_subgoals + 1):
            alpha = i / (self.num_subgoals + 1)
            subgoal = (1 - alpha) * start_z + alpha * goal_z
            
            # Add some noise for diversity
            if i < self.num_subgoals:
                noise = np.random.normal(0, 0.1, self.z_dim)
                subgoal += noise
            
            subgoals.append(subgoal)
        
        return subgoals
    
    def record_successful_path(self, z_path: List[np.ndarray], actions: List[np.ndarray]):
        """Record successful path for future reuse"""
        self.successful_paths.append((z_path, actions))
        
        # Extract subgoals
        for i in range(len(z_path) - 1):
            key = tuple(z_path[i].round(3))  # Rounded as key
            self.subgoal_memory[key].append(z_path[i + 1])


class EnhancedFBLearner:
    """Enhanced FB learning with better latent space structure"""
    
    def __init__(self, config: FBConfig, obs_dim: int, action_dim: int, device: str = 'cpu'):
        self.config = config
        self.device = device
        self.action_dim = action_dim
        
        # Enhanced networks
        self.encoder = EnhancedEncoder(obs_dim, config.z_dim, config.f_hidden_dim).to(device)
        self.forward_model = EnhancedForwardModel(
            config.z_dim, action_dim, config.f_hidden_dim, config.num_layers, config.num_forward_heads
        ).to(device)
        self.backward_model = EnhancedBackwardModel(
            config.z_dim, action_dim, config.b_hidden_dim, config.num_layers
        ).to(device)
        
        # Target networks
        self.target_encoder = EnhancedEncoder(obs_dim, config.z_dim, config.f_hidden_dim).to(device)
        self.target_forward = EnhancedForwardModel(
            config.z_dim, action_dim, config.f_hidden_dim, config.num_layers, config.num_forward_heads
        ).to(device)
        
        # Latent planner
        self.planner = FBLatentPlanner(config.z_dim)
        
        # Initialize targets as copies
        self._update_target_networks(tau=1.0)
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) +
            list(self.forward_model.parameters()) +
            list(self.backward_model.parameters()),
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2, eta_min=1e-6
        )
        
        # Replay buffer with priorities
        self.replay_buffer = deque(maxlen=config.buffer_capacity)
        self.priorities = deque(maxlen=config.buffer_capacity)
        
        # Statistics
        self.train_steps = 0
        self.current_noise = config.noise_scale
        
    def _update_target_networks(self, tau: float = 0.005):
        """Soft update target networks"""
        with torch.no_grad():
            for target_param, param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
            for target_param, param in zip(self.target_forward.parameters(), self.forward_model.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def encode(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Encode observation to latent representation"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, log_var, _ = self.encoder(obs_tensor)
            
            if deterministic:
                z = mean
            else:
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                z = mean + eps * std
            
            return z.squeeze(0).detach().cpu().numpy()
    
    def compute_intrinsic_reward(self, z_t: np.ndarray, z_next: np.ndarray) -> float:
        """Compute intrinsic reward based on novelty and learning progress"""
        # Convert to tensors
        z_t_tensor = torch.FloatTensor(z_t).unsqueeze(0).to(self.device)
        z_next_tensor = torch.FloatTensor(z_next).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Predict forward from multiple heads
            random_action = torch.randn(1, self.action_dim).to(self.device) * 0.1
            predictions = self.forward_model(z_t_tensor, random_action)
            
            # Compute prediction uncertainty (epistemic)
            means = torch.stack([p[0] for p in predictions])
            vars = torch.stack([torch.exp(p[1]) for p in predictions])
            
            # Mean variance across heads
            avg_variance = vars.mean().item()
            
            # Distance to nearest neighbor in replay buffer (novelty)
            if self.replay_buffer:
                # Sample some states from buffer
                buffer_zs = []
                for i in range(min(100, len(self.replay_buffer))):
                    buf_obs, _, _, _ = self.replay_buffer[i]
                    buf_z = self.encode(buf_obs, deterministic=True)
                    buffer_zs.append(buf_z)
                
                if buffer_zs:
                    distances = [np.linalg.norm(z_next - buf_z) for buf_z in buffer_zs]
                    novelty = np.min(distances) if distances else 1.0
                else:
                    novelty = 1.0
            else:
                novelty = 1.0
            
            # Learning progress: how well can we predict this transition?
            # Use backward model to predict action
            pred_mean, pred_log_var = self.backward_model(z_t_tensor, z_next_tensor)
            pred_uncertainty = torch.exp(pred_log_var).mean().item()
            
            # Combined intrinsic reward
            reward = (novelty * 0.5 +  # Novelty bonus
                     (1.0 / (1.0 + pred_uncertainty)) * 0.3 +  # Certainty bonus
                     (1.0 / (1.0 + avg_variance)) * 0.2)  # Model confidence
            
            return float(reward)
    
    def store_transition_with_priority(self, obs: np.ndarray, action: np.ndarray, 
                                       next_obs: np.ndarray, reward: float = 0.0):
        """Store transition with computed priority"""
        # Compute priority based on prediction error
        z = self.encode(obs, deterministic=True)
        z_next = self.encode(next_obs, deterministic=True)
        
        z_tensor = torch.FloatTensor(z).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.forward_model(z_tensor, action_tensor)
            # Compute prediction error
            errors = []
            for mean_pred, log_var_pred in predictions:
                error = F.mse_loss(mean_pred, torch.FloatTensor(z_next).unsqueeze(0).to(self.device))
                errors.append(error.item())
            
            priority = np.mean(errors)  # Higher error = higher priority
        
        self.replay_buffer.append((obs, action, next_obs, reward))
        self.priorities.append(priority)
    
    def _sample_batch_with_priority(self, batch_size: int):
        """Sample batch using priority weights"""
        if len(self.priorities) == 0:
            indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        else:
            priorities = np.array(self.priorities) + 1e-6  # Add small constant
            probs = priorities / priorities.sum()
            indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False, p=probs)
        
        batch = zip(*[self.replay_buffer[i] for i in indices])
        return batch, indices
    
    def train_step(self) -> Dict[str, float]:
        """Enhanced training step with multiple losses"""
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        # Sample batch
        batch, batch_indices = self._sample_batch_with_priority(self.config.batch_size)
        obs, actions, next_obs, rewards = batch
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(obs)).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(actions)).to(self.device)
        next_obs_tensor = torch.FloatTensor(np.array(next_obs)).to(self.device)
        rewards_tensor = torch.FloatTensor(np.array(rewards)).to(self.device).unsqueeze(1)
        
        # Encode observations
        z_t_mean, z_t_log_var, enc_extra = self.encoder(obs_tensor)
        z_t = z_t_mean + torch.exp(0.5 * z_t_log_var) * torch.randn_like(z_t_mean)

        # Get next observation encoding using target encoder
        with torch.no_grad():
            z_next_mean, z_next_log_var, _ = self.target_encoder(next_obs_tensor)
            z_next_target = z_next_mean  # Deterministic target encoding
        
        # ===== 1. Forward-Backward Consistency Loss =====
        # Forward predictions
        forward_preds = self.forward_model(z_t, actions_tensor)
        
        # Forward loss with uncertainty
        forward_losses = []
        for mean_pred, log_var_pred in forward_preds:
            # Negative log likelihood
            precision = torch.exp(-log_var_pred)
            forward_loss = 0.5 * (precision * (mean_pred - z_next_target.detach())**2 + log_var_pred).mean()
            forward_losses.append(forward_loss)
        
        forward_loss = torch.stack(forward_losses).mean()
        
        # Backward prediction
        pred_action_mean, pred_action_log_var = self.backward_model(z_t, z_next_target.detach())
        
        # Negative log likelihood for actions
        action_precision = torch.exp(-pred_action_log_var)
        backward_loss = 0.5 * (action_precision * (pred_action_mean - actions_tensor)**2 + 
                              pred_action_log_var).mean()
        
        # ===== 2. Contrastive Loss (SimCLR style) =====
        # Augment observations
        obs_aug = obs_tensor + torch.randn_like(obs_tensor) * 0.1
        z_aug_mean, z_aug_log_var, _ = self.encoder(obs_aug)
        z_aug = z_aug_mean + torch.exp(0.5 * z_aug_log_var) * torch.randn_like(z_aug_mean)
        
        # InfoNCE loss
        temperature = 0.1
        batch_size = obs_tensor.shape[0]
        
        # Positive pairs: (z_t, z_aug)
        pos_sim = F.cosine_similarity(z_t, z_aug, dim=-1) / temperature
        
        # Negative pairs: shuffle
        neg_indices = torch.randperm(batch_size)
        z_neg = z_t[neg_indices]
        neg_sim = F.cosine_similarity(z_t.unsqueeze(1), z_neg.unsqueeze(0), dim=-1) / temperature
        
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        contrastive_loss = F.cross_entropy(logits, labels)
        
        # ===== 3. Diversity Loss =====
        # Encourage diverse forward predictions
        if len(forward_preds) > 1:
            pred_means = torch.stack([p[0] for p in forward_preds])
            mean_of_means = pred_means.mean(dim=0)
            diversity_loss = -F.mse_loss(pred_means, mean_of_means.unsqueeze(0).expand_as(pred_means))
        else:
            diversity_loss = torch.tensor(0.0, device=self.device)
        
        # ===== 4. Regularization Losses =====
        # KL divergence regularization
        kl_loss = -0.5 * torch.mean(1 + z_t_log_var - z_t_mean.pow(2) - z_t_log_var.exp())
        
        # Norm regularization
        norm_loss = torch.mean(torch.norm(z_t, dim=-1))
        
        # VQ loss if using vector quantization
        vq_loss = enc_extra.get('vq_loss', torch.tensor(0.0, device=self.device))
        
        # ===== 5. Total Loss =====
        total_loss = (
            self.config.fb_weight * (forward_loss + backward_loss) +
            self.config.contrastive_weight * contrastive_loss +
            self.config.diversity_weight * diversity_loss +
            self.config.norm_weight * norm_loss +
            0.1 * kl_loss +
            0.1 * vq_loss
        )
        
        # ===== 6. Optimize =====
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.forward_model.parameters()) +
            list(self.backward_model.parameters()),
            max_norm=1.0
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
        # ===== 7. Update Targets and Priorities =====
        self.train_steps += 1
        if self.train_steps % self.config.target_update_freq == 0:
            self._update_target_networks()
        
        # Update priorities based on new prediction errors
        with torch.no_grad():
            new_predictions = self.forward_model(z_t.detach(), actions_tensor)
            new_errors = []
            for mean_pred, log_var_pred in new_predictions:
                error = F.mse_loss(mean_pred, z_next_target).item()
                new_errors.append(error)
            
            new_priority = np.mean(new_errors)
            
            # Update priorities for this batch
            for idx in batch_indices:
                if idx < len(self.priorities):
                    self.priorities[idx] = new_priority
        
        # Decay exploration noise
        self.current_noise = max(
            self.config.min_noise,
            self.current_noise * self.config.noise_decay
        )
        
        return {
            'total_loss': total_loss.item(),
            'forward_loss': forward_loss.item(),
            'backward_loss': backward_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'kl_loss': kl_loss.item(),
            'norm_loss': norm_loss.item(),
            'vq_loss': vq_loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0],
            'noise_scale': self.current_noise,
            'buffer_size': len(self.replay_buffer)
        }


class FBResearchAgent:
    """
    Complete FB agent for research comparison with RL
    
    Designed to answer the research questions:
    1. Performance comparison with RL
    2. Adaptability to new scenes
    3. Training efficiency
    4. Generalization capability
    """
    
    def __init__(self, config: FBConfig, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.config = config
        self.device = device
        
        # Observation dimension matches RL environment
        self.obs_dim = 22  # From ray_tracer_env.py: position(3) + direction(3) + normal(3) + material(4) + color(3) + bounces(1) + through(1)
        action_dim = 2  # theta, phi like RL
        
        # Create FB learner
        self.fb_learner = EnhancedFBLearner(config, self.obs_dim, action_dim, device)
        
        # Light memory
        self.light_memory = {
            'encodings': [],  # Light encodings
            'contexts': [],   # Scene contexts where lights were found
            'paths': []       # Successful paths to lights
        }
        
        # Scene adaptation memory
        self.scene_memory = defaultdict(list)
        
        # Research statistics
        self.research_stats = {
            'total_rays': 0,
            'light_hits': 0,
            'variance_reduction': [],
            'adaptation_speed': [],
            'generalization_scores': [],
            'fb_vs_random_ratios': []
        }
        
        # Strategy weights (adaptive)
        self.strategy_weights = {
            'fb_guided': 0.7,
            'light_directed': 0.2,
            'random_explore': 0.1
        }
        
        print(f"FB Research Agent initialized with config:")
        print(f"  z_dim: {config.z_dim}")
        print(f"  num_forward_heads: {config.num_forward_heads}")
        print(f"  device: {device}")
        print(f"  obs_dim: {self.obs_dim}")
    
    def create_observation(self, intersection: Intersection, ray: Ray, 
                          bounce_count: int, accumulated_color: Colour) -> np.ndarray:
        """Create observation compatible with RL environment"""
        if intersection and intersection.intersects:
            pos = intersection.point
            normal = intersection.normal
            material = intersection.object.material
            
            obs = np.array([
                # Position (3)
                pos.x, pos.y, pos.z,
                # Ray direction (3)
                ray.D.x, ray.D.y, ray.D.z,
                # Surface normal (3)
                normal.x, normal.y, normal.z,
                # Material properties (4)
                material.reflective,
                material.transparent,
                material.emitive,
                material.refractive_index,
                # Current accumulated color (3) - normalized
                accumulated_color.r / 255.0,
                accumulated_color.g / 255.0,
                accumulated_color.b / 255.0,
                # Bounce and history (3)
                float(bounce_count) / self.config.max_bounces,
                0.0,  # through_count (not used in simplified version)
                float(intersection.object.id if hasattr(intersection.object, 'id') else 0) / 100.0,
                # Additional info for FB (3)
                ray.origin.x / 10.0,  # Normalized camera position
                ray.origin.y / 10.0,
                ray.origin.z / 10.0
            ], dtype=np.float32)
        else:
            # No intersection
            obs = np.array([
                ray.origin.x, ray.origin.y, ray.origin.z,
                ray.D.x, ray.D.y, ray.D.z,
                0, 0, 0,
                0, 0, 0, 1,
                0, 0, 0,
                float(bounce_count) / self.config.max_bounces,
                0, 0,
                ray.origin.x / 10.0,
                ray.origin.y / 10.0,
                ray.origin.z / 10.0
            ], dtype=np.float32)
        
        return obs
    
    def adapt_to_scene(self, initial_observations: List[np.ndarray]):
        """Quickly adapt to new scene with few-shot learning"""
        print("Adapting to new scene...")
        
        adaptation_losses = []
        
        # Use initial observations to update encoder
        for obs in initial_observations:
            # Create augmented versions
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            obs_aug = obs_tensor + torch.randn_like(obs_tensor) * 0.2
            
            # Contrastive loss between original and augmented
            z_mean, z_log_var, _ = self.fb_learner.encoder(obs_tensor)
            z_aug_mean, z_aug_log_var, _ = self.fb_learner.encoder(obs_aug)
            
            # Positive pair loss
            pos_loss = F.mse_loss(z_mean, z_aug_mean)
            
            # Store for scene memory
            self.scene_memory['initial_encodings'].append(z_mean.detach().cpu().numpy())
            
            adaptation_losses.append(pos_loss.item())
        
        print(f"  Adaptation complete. Avg loss: {np.mean(adaptation_losses):.4f}")
        self.research_stats['adaptation_speed'].append(len(initial_observations))
        
        # Adjust strategy weights for exploration
        self.strategy_weights['random_explore'] = 0.3  # More exploration in new scene
        self.strategy_weights['fb_guided'] = 0.5
    
    def choose_direction_research(self, observation: np.ndarray, scene_context: str = "default", 
                                 exploration_phase: str = "normal") -> Tuple[np.ndarray, Dict]:
        total = sum(self.strategy_weights.values())
        if total > 0:
            # Always normalize to ensure sum is exactly 1
            for key in self.strategy_weights:
                self.strategy_weights[key] /= total
        else:
            # Reset to default if total is 0 or negative
            self.strategy_weights = {
                'fb_guided': 0.7,
                'light_directed': 0.2, 
                'random_explore': 0.1
            }
        
        self.research_stats['total_rays'] += 1
        
        """
        Choose direction with research tracking
        
        exploration_phase: "warmup", "explore", "exploit", "test"
        """
        self.research_stats['total_rays'] += 1
        
        # Encode current observation
        current_z = self.fb_learner.encode(observation, deterministic=True)
        
        # Record scene context
        if scene_context not in self.scene_memory:
            self.scene_memory[scene_context] = []
        self.scene_memory[scene_context].append(current_z)
        
        # Choose strategy based on phase
        if exploration_phase == "warmup":
            # Pure exploration
            action = np.random.uniform(-1, 1, 2)
            strategy = "random_warmup"
            
        elif exploration_phase == "explore":
            # Exploration with FB guidance
            strategies = ['fb_guided', 'light_directed', 'random_explore']
            strategy = np.random.choice(strategies, p=list(self.strategy_weights.values()))
            
            if strategy == 'fb_guided':
                action = self._fb_guided_action(current_z, exploration=True)
            elif strategy == 'light_directed' and self.light_memory['encodings']:
                action = self._light_directed_action(current_z)
            else:
                action = np.random.uniform(-1, 1, 2)
                
        elif exploration_phase == "exploit":
            # Exploitation: use learned knowledge
            if self.light_memory['encodings']:
                # Try to hit known lights
                action = self._light_directed_action(current_z)
                strategy = "light_exploit"
            else:
                # Use FB model
                action = self._fb_guided_action(current_z, exploration=False)
                strategy = "fb_exploit"
                
        else:  # "test"
            # Deterministic for testing
            action = self._fb_guided_action(current_z, exploration=False)
            strategy = "test_deterministic"
        
        # Add noise based on phase
        noise_level = {
            "warmup": 0.3,
            "explore": 0.1,
            "exploit": 0.05,
            "test": 0.01
        }.get(exploration_phase, 0.1)
        
        if noise_level > 0:
            action = action + np.random.normal(0, noise_level, 2)
            action = np.clip(action, -1, 1)
        
        # Compute intrinsic reward for research
        intrinsic_reward = self.fb_learner.compute_intrinsic_reward(current_z, current_z)  # Placeholder
        
        info = {
            'strategy': strategy,
            'phase': exploration_phase,
            'intrinsic_reward': intrinsic_reward,
            'latent_norm': np.linalg.norm(current_z),
            'scene_context': scene_context
        }
        
        return action, info
    
    def _fb_guided_action(self, current_z: np.ndarray, exploration: bool = True) -> np.ndarray:
        """FB-guided action sampling"""
        if exploration:
            # Exploration: sample random target in latent space
            noise = np.random.normal(0, self.fb_learner.current_noise, current_z.shape)
            target_z = current_z + noise
            
            current_z_tensor = torch.FloatTensor(current_z).unsqueeze(0).to(self.device)
            target_z_tensor = torch.FloatTensor(target_z).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                mean, log_var = self.fb_learner.backward_model(current_z_tensor, target_z_tensor)
                action = mean + torch.exp(0.5 * log_var) * torch.randn_like(mean)
                return action.squeeze(0).detach().cpu().numpy()
        else:
            # Exploitation: use learned paths
            if self.light_memory['paths']:
                # Try to follow known successful path
                path_idx = np.random.randint(len(self.light_memory['paths']))
                z_path, actions = self.light_memory['paths'][path_idx]
                
                # Find closest point in path
                distances = [np.linalg.norm(current_z - z) for z in z_path]
                closest_idx = np.argmin(distances)
                
                if closest_idx < len(actions):
                    return actions[closest_idx]
            
            # Fallback: small perturbation
            return np.random.uniform(-0.2, 0.2, 2)
    
    def _light_directed_action(self, current_z: np.ndarray) -> np.ndarray:
        """Action directed toward known lights"""
        if not self.light_memory['encodings']:
            return np.random.uniform(-1, 1, 2)
        
        # Find closest light
        light_encodings = self.light_memory['encodings']
        distances = [np.linalg.norm(current_z - light_z) for light_z in light_encodings]
        closest_idx = np.argmin(distances)
        target_z = light_encodings[closest_idx]
        
        # Plan to light
        actions = self.fb_learner.planner.plan_to_light(
            current_z, target_z, self.fb_learner.backward_model, self.device
        )
        
        return actions[0] if actions else np.random.uniform(-0.5, 0.5, 2)
    
    def record_success(self, observation: np.ndarray, action: np.ndarray, 
                      next_observation: np.ndarray, reward: float, is_light_hit: bool):
        """Record successful transition for learning"""
        
        # Store transition with priority
        self.fb_learner.store_transition_with_priority(
            observation, action, next_observation, reward
        )
        
        if is_light_hit:
            self.research_stats['light_hits'] += 1
            
            # Record light encoding
            light_z = self.fb_learner.encode(observation, deterministic=True)
            self.light_memory['encodings'].append(light_z)
            
            # Keep only recent lights
            if len(self.light_memory['encodings']) > 20:
                self.light_memory['encodings'].pop(0)
        
        # Update strategy weights based on success
        if reward > 0:
            # Increase FB guidance weight
            self.strategy_weights['fb_guided'] = min(0.8, self.strategy_weights['fb_guided'] + 0.01)
            self.strategy_weights['random_explore'] = max(0.05, self.strategy_weights['random_explore'] - 0.005)
    
    def compute_variance_reduction(self, traditional_samples: List[float], fb_samples: List[float]) -> float:
        """Compute variance reduction compared to traditional sampling"""
        if len(traditional_samples) < 2 or len(fb_samples) < 2:
            return 0.0
        
        var_traditional = np.var(traditional_samples)
        var_fb = np.var(fb_samples)
        
        if var_traditional > 0:
            reduction = (var_traditional - var_fb) / var_traditional
            self.research_stats['variance_reduction'].append(reduction)
            return reduction
        
        return 0.0
    
    def compute_generalization_score(self, scene_contexts: List[str]) -> float:
        """Compute how well representations generalize across scenes"""
        if len(scene_contexts) < 2:
            return 0.0
        
        # Compute average distance between scene representations
        scene_centroids = []
        for context in scene_contexts:
            if context in self.scene_memory and len(self.scene_memory[context]) > 0:
                centroids = np.mean(self.scene_memory[context], axis=0)
                scene_centroids.append(centroids)
        
        if len(scene_centroids) < 2:
            return 0.0
        
        # Compute pairwise distances
        distances = []
        for i in range(len(scene_centroids)):
            for j in range(i + 1, len(scene_centroids)):
                dist = np.linalg.norm(scene_centroids[i] - scene_centroids[j])
                distances.append(dist)
        
        # Normalize by within-scene variance
        within_var = np.mean([np.var(self.scene_memory[ctx], axis=0).mean() 
                            for ctx in scene_contexts if ctx in self.scene_memory])
        
        if within_var > 0:
            generalization = np.mean(distances) / within_var
            self.research_stats['generalization_scores'].append(generalization)
            return generalization
        
        return 0.0
    
    def train(self, steps: int = 1000) -> Dict[str, List[float]]:
        """Train FB learner and return training statistics"""
        training_stats = defaultdict(list)
        
        for step in range(steps):
            if len(self.fb_learner.replay_buffer) >= self.fb_learner.config.batch_size:
                stats = self.fb_learner.train_step()
                
                for key, value in stats.items():
                    training_stats[key].append(value)
                
                # Update research stats
                fb_guided_ratio = self.strategy_weights['fb_guided']
                self.research_stats['fb_vs_random_ratios'].append(fb_guided_ratio)
        
        return training_stats
    
    def get_research_metrics(self) -> Dict[str, Any]:
        """Get comprehensive research metrics"""
        metrics = {
            'performance': {
                'light_hit_rate': self.research_stats['light_hits'] / max(1, self.research_stats['total_rays']),
                'avg_variance_reduction': np.mean(self.research_stats['variance_reduction']) if self.research_stats['variance_reduction'] else 0,
                'total_rays': self.research_stats['total_rays'],
                'light_hits': self.research_stats['light_hits']
            },
            'adaptability': {
                'avg_adaptation_speed': np.mean(self.research_stats['adaptation_speed']) if self.research_stats['adaptation_speed'] else 0,
                'num_scenes_encountered': len(self.scene_memory),
                'scene_specific_memory': {k: len(v) for k, v in self.scene_memory.items() if k != 'initial_encodings'}
            },
            'efficiency': {
                'buffer_utilization': len(self.fb_learner.replay_buffer) / self.fb_learner.config.buffer_capacity,
                'avg_fb_guided_ratio': np.mean(self.research_stats['fb_vs_random_ratios']) if self.research_stats['fb_vs_random_ratios'] else 0,
                'current_noise_scale': self.fb_learner.current_noise
            },
            'generalization': {
                'avg_generalization_score': np.mean(self.research_stats['generalization_scores']) if self.research_stats['generalization_scores'] else 0,
                'light_memory_size': len(self.light_memory['encodings']),
                'successful_paths': len(self.light_memory['paths'])
            }
        }
        
        return metrics
    
    def save(self, path: str):
        """Save agent state"""
        save_data = {
            'fb_learner_state': {
                'encoder': self.fb_learner.encoder.state_dict(),
                'forward_model': self.fb_learner.forward_model.state_dict(),
                'backward_model': self.fb_learner.backward_model.state_dict(),
                'optimizer': self.fb_learner.optimizer.state_dict(),
                'scheduler': self.fb_learner.scheduler.state_dict(),
                'train_steps': self.fb_learner.train_steps,
                'current_noise': self.fb_learner.current_noise
            },
            'light_memory': self.light_memory,
            'scene_memory': dict(self.scene_memory),
            'research_stats': self.research_stats,
            'strategy_weights': self.strategy_weights,
            'config': self.config.to_dict()
        }
        
        torch.save(save_data, path)
        print(f"Agent saved to {path}")
    
    def load(self, path: str):
        """Load agent state"""
        save_data = torch.load(path, map_location=self.device)
        
        # Load FB learner
        self.fb_learner.encoder.load_state_dict(save_data['fb_learner_state']['encoder'])
        self.fb_learner.forward_model.load_state_dict(save_data['fb_learner_state']['forward_model'])
        self.fb_learner.backward_model.load_state_dict(save_data['fb_learner_state']['backward_model'])
        self.fb_learner.optimizer.load_state_dict(save_data['fb_learner_state']['optimizer'])
        self.fb_learner.scheduler.load_state_dict(save_data['fb_learner_state']['scheduler'])
        self.fb_learner.train_steps = save_data['fb_learner_state']['train_steps']
        self.fb_learner.current_noise = save_data['fb_learner_state']['current_noise']
        
        # Load memories and stats
        self.light_memory = save_data['light_memory']
        self.scene_memory = defaultdict(list, save_data['scene_memory'])
        self.research_stats = save_data['research_stats']
        self.strategy_weights = save_data['strategy_weights']
        
        # Update target networks
        self.fb_learner._update_target_networks(tau=1.0)
        
        print(f"Agent loaded from {path}")
        print(f"  Training steps: {self.fb_learner.train_steps}")
        print(f"  Light memory: {len(self.light_memory['encodings'])} lights")
        print(f"  Scene memory: {len(self.scene_memory)} scenes")


def create_research_comparison_scenes():
    """Create different scenes for research comparison"""
    scenes = {}
    
    # Scene 1: Training scene (similar to RL training)
    scenes['training'] = [
        Sphere(Vector(0, -100, -3), 99, Material(0, 0, 0.1, 1), Colour(150, 150, 150), id=1),
        Sphere(Vector(0, 0, -3), 0.7, Material(1, 0, 0, 1), Colour(255, 255, 255), id=2),
        Sphere(Vector(-1.5, 0.3, -3), 0.5, Material(1, 0, 0, 1), Colour(200, 200, 255), id=3),
        Sphere(Vector(1.5, -0.2, -3), 0.5, Material(1, 0, 0, 1), Colour(255, 200, 200), id=4),
        Sphere(Vector(0, 2.5, -3), 0.6, Material(0, 0, 1, 1), Colour(255, 255, 200), id=99),
        Sphere(Vector(-2, 1.8, -3), 0.4, Material(0, 0, 1, 1), Colour(200, 255, 200), id=100),
    ]
    
    # Scene 2: Different lighting
    scenes['different_lighting'] = [
        Sphere(Vector(0, -100, -3), 99, Material(0, 0, 0.1, 1), Colour(150, 150, 150), id=1),
        Sphere(Vector(0, 0, -3), 0.7, Material(1, 0, 0, 1), Colour(255, 255, 255), id=2),
        Sphere(Vector(-1.5, 0.3, -3), 0.5, Material(1, 0, 0, 1), Colour(200, 200, 255), id=3),
        Sphere(Vector(1.5, -0.2, -3), 0.5, Material(1, 0, 0, 1), Colour(255, 200, 200), id=4),
        # Lights in different positions
        Sphere(Vector(2, 2, -2), 0.5, Material(0, 0, 1, 1), Colour(255, 200, 200), id=99),
        Sphere(Vector(-2, 3, -4), 0.4, Material(0, 0, 1, 1), Colour(200, 255, 200), id=100),
    ]
    
    # Scene 3: Different materials
    scenes['different_materials'] = [
        Sphere(Vector(0, -100, -3), 99, Material(0, 0, 0.1, 1), Colour(150, 150, 150), id=1),
        Sphere(Vector(0, 0, -3), 0.7, Material(0.5, 0.5, 0, 1.5), Colour(255, 255, 255), id=2),  # Glass
        Sphere(Vector(-1.5, 0.3, -3), 0.5, Material(0.8, 0, 0, 1), Colour(200, 200, 255), id=3),  # Reflective
        Sphere(Vector(1.5, -0.2, -3), 0.5, Material(0, 0.8, 0, 1), Colour(255, 200, 200), id=4),  # Transparent
        Sphere(Vector(0, 2.5, -3), 0.6, Material(0, 0, 1, 1), Colour(255, 255, 200), id=99),
        Sphere(Vector(-2, 1.8, -3), 0.4, Material(0, 0, 1, 1), Colour(200, 255, 200), id=100),
    ]
    
    # Scene 4: Complex scene
    scenes['complex'] = [
        Sphere(Vector(0, -100, -3), 99, Material(0, 0, 0.1, 1), Colour(150, 150, 150), id=1),
        Sphere(Vector(0, 0, -3), 0.7, Material(1, 0, 0, 1), Colour(255, 255, 255), id=2),
        Sphere(Vector(-2, 0, -4), 0.6, Material(0.8, 0, 0, 1), Colour(200, 200, 255), id=3),
        Sphere(Vector(2, -0.5, -3.5), 0.6, Material(0.8, 0, 0, 1), Colour(255, 200, 200), id=4),
        Sphere(Vector(-1, 1, -5), 0.4, Material(0.6, 0, 0, 1), Colour(255, 200, 150), id=5),
        Sphere(Vector(1, 1.5, -4), 0.4, Material(0.6, 0, 0, 1), Colour(200, 255, 200), id=6),
        # Multiple lights
        Sphere(Vector(0, 3, -2), 0.5, Material(0, 0, 1, 1), Colour(255, 255, 200), id=99),
        Sphere(Vector(-3, 2, -3), 0.4, Material(0, 0, 1, 1), Colour(255, 200, 200), id=100),
        Sphere(Vector(3, 2.5, -4), 0.4, Material(0, 0, 1, 1), Colour(200, 255, 200), id=101),
    ]
    
    return scenes


if __name__ == "__main__":
    # Test the enhanced FB agent
    config = FBConfig(
        z_dim=64,
        f_hidden_dim=512,
        b_hidden_dim=256,
        num_forward_heads=3,
        num_layers=2,
        learning_rate=3e-4,
        buffer_capacity=100000
    )
    
    agent = FBResearchAgent(config)
    print("Enhanced FB Research Agent created successfully!")
    
    # Test with a sample observation
    test_obs = np.random.randn(22).astype(np.float32)
    action, info = agent.choose_direction_research(test_obs, scene_context="test", exploration_phase="explore")
    print(f"Test action: {action}")
    print(f"Strategy: {info['strategy']}")
    
    # Show research questions being addressed
    print("\nResearch Questions Addressed:")
    print("1. Performance: Variance reduction tracking")
    print("2. Adaptability: Scene adaptation memory")
    print("3. Efficiency: Buffer utilization tracking")
    print("4. Generalization: Cross-scene metrics")