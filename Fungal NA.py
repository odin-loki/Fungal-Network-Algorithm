import torch
import ray
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkState(Enum):
    EXPLORING = 1
    CONNECTING = 2
    OPTIMIZING = 3
    STABLE = 4

@dataclass
class NetworkConfig:
    num_nodes: int
    space_dims: Tuple[int, int, int]
    growth_rate: float = 0.1
    decay_rate: float = 0.05
    connection_threshold: float = 1.0
    pattern_threshold: float = 0.7
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

@ray.remote
class NetworkRegion:
    def __init__(self, config: NetworkConfig, region_id: int):
        self.config = config
        self.region_id = region_id
        self.device = torch.device(config.device)
        
        # Network state
        self.positions = torch.rand((config.num_nodes, 3), device=self.device)
        self.resources = torch.ones(config.num_nodes, device=self.device)
        self.connections = torch.zeros((config.num_nodes, config.num_nodes), device=self.device)
        self.state = NetworkState.EXPLORING
        
        # Pattern templates
        self.patterns = {
            'circle': self._create_circle(),
            'cross': self._create_cross(),
            'triangle': self._create_triangle(),
            'square': self._create_square(),
            'hexagon': self._create_hexagon(),
            'star': self._create_star(),
            'spiral': self._create_spiral(),
            'grid': self._create_grid(),
            'web': self._create_web(),
            'tree': self._create_tree()
        }
        
        # Feature tracking
        self.pattern_history = []
        self.efficiency_history = []
        self.current_pattern = None
        
    def _create_circle(self) -> torch.Tensor:
        angles = torch.linspace(0, 2*torch.pi, self.config.num_nodes)
        x = torch.cos(angles)
        y = torch.sin(angles)
        return self._normalize_pattern(torch.stack([x, y, torch.zeros_like(x)], dim=1))
        
    def _create_cross(self) -> torch.Tensor:
        n = self.config.num_nodes // 4
        arm = torch.linspace(-1, 1, n)
        x = torch.cat([arm, torch.zeros_like(arm)*2])
        y = torch.cat([torch.zeros_like(arm), arm*2])
        return self._normalize_pattern(torch.stack([x, y, torch.zeros_like(x)], dim=1))
        
    def _create_triangle(self) -> torch.Tensor:
        n = self.config.num_nodes // 3
        angles = torch.linspace(0, 2*torch.pi, 4)[:-1]  # 3 points
        x = torch.cos(angles).repeat_interleave(n)
        y = torch.sin(angles).repeat_interleave(n)
        return self._normalize_pattern(torch.stack([x, y, torch.zeros_like(x)], dim=1))
        
    def _create_square(self) -> torch.Tensor:
        n = self.config.num_nodes // 4
        side = torch.linspace(-1, 1, n)
        x = torch.cat([side, torch.ones_like(side), -side, -torch.ones_like(side)])
        y = torch.cat([torch.ones_like(side), -side, -torch.ones_like(side), side])
        return self._normalize_pattern(torch.stack([x, y, torch.zeros_like(x)], dim=1))
        
    def _create_hexagon(self) -> torch.Tensor:
        angles = torch.linspace(0, 2*torch.pi, 7)[:-1]  # 6 points
        n = self.config.num_nodes // 6
        x = torch.cos(angles).repeat_interleave(n)
        y = torch.sin(angles).repeat_interleave(n)
        return self._normalize_pattern(torch.stack([x, y, torch.zeros_like(x)], dim=1))
        
    def _create_star(self) -> torch.Tensor:
        points = 5
        angles = torch.linspace(0, 2*torch.pi, points*2+1)[:-1]
        radii = torch.tensor([1.0, 0.5]).repeat(points)
        x = (radii * torch.cos(angles))
        y = (radii * torch.sin(angles))
        return self._normalize_pattern(torch.stack([x, y, torch.zeros_like(x)], dim=1))
        
    def _create_spiral(self) -> torch.Tensor:
        t = torch.linspace(0, 6*torch.pi, self.config.num_nodes)
        r = t/6
        x = r * torch.cos(t)
        y = r * torch.sin(t)
        return self._normalize_pattern(torch.stack([x, y, torch.zeros_like(x)], dim=1))
        
    def _create_grid(self) -> torch.Tensor:
        side = int(np.sqrt(self.config.num_nodes))
        x = torch.linspace(-1, 1, side).repeat(side)
        y = torch.repeat_interleave(torch.linspace(-1, 1, side), side)
        return self._normalize_pattern(torch.stack([x, y, torch.zeros_like(x)], dim=1))
        
    def _create_web(self) -> torch.Tensor:
        rings = 5
        points_per_ring = self.config.num_nodes // rings
        r = torch.linspace(0.2, 1.0, rings).repeat_interleave(points_per_ring)
        t = torch.linspace(0, 2*torch.pi, points_per_ring).repeat(rings)
        x = r * torch.cos(t)
        y = r * torch.sin(t)
        return self._normalize_pattern(torch.stack([x, y, torch.zeros_like(x)], dim=1))
        
    def _create_tree(self) -> torch.Tensor:
        levels = 4
        branches = 2
        points_per_branch = self.config.num_nodes // (branches**levels)
        x, y = [], []
        for level in range(levels):
            spread = 2.0 ** (level-levels+1)
            height = 1 - level/levels
            branch_points = branches**level
            for b in range(branch_points):
                x_pos = (b - branch_points/2 + 0.5) * spread
                x.extend([x_pos] * points_per_branch)
                y.extend([height] * points_per_branch)
        return self._normalize_pattern(torch.stack([torch.tensor(x), torch.tensor(y), torch.zeros(len(x))], dim=1))
        
    def _normalize_pattern(self, pattern: torch.Tensor) -> torch.Tensor:
        return pattern / torch.max(torch.abs(pattern))
        
    def update(self, dt: float) -> None:
        """Update network region"""
        self._update_resources(dt)
        self._update_connections()
        self._optimize()
        self._track_features()
        
    def _update_resources(self, dt: float) -> None:
        resource_grad = torch.gradient(self.resources)
        self.resources += dt * self.config.growth_rate * resource_grad[0]
        self.resources *= (1 - self.config.decay_rate * dt)
        self.resources.clamp_(min=0)
        
    def _update_connections(self) -> None:
        distances = torch.cdist(self.positions, self.positions)
        resource_benefit = torch.outer(self.resources, self.resources)
        self.connections = torch.where(
            distances < self.config.connection_threshold,
            resource_benefit,
            torch.zeros_like(self.connections)
        )
        
    def _optimize(self) -> None:
        self.connections *= (self.connections > self.config.connection_threshold)
        connected = torch.nonzero(self.connections)
        if len(connected) > 0:
            for i, j in connected:
                direction = self.positions[j] - self.positions[i]
                force = self.connections[i,j] * direction
                self.positions[i] += 0.1 * force
                self.positions[j] -= 0.1 * force
                
    def _track_features(self) -> None:
        # Track current pattern
        self.current_pattern = self._match_pattern()
        self.pattern_history.append(self.current_pattern)
        
        # Track efficiency
        efficiency = torch.sum(self.connections) / (torch.sum(self.resources) + 1e-6)
        self.efficiency_history.append(efficiency.item())
        
    def _match_pattern(self) -> Optional[str]:
        best_match = None
        best_score = 0.0
        
        state = self.positions.reshape(-1)
        for name, pattern in self.patterns.items():
            score = torch.nn.functional.cosine_similarity(
                state.unsqueeze(0),
                pattern.reshape(-1).unsqueeze(0)
            ).item()
            if score > best_score:
                best_score = score
                best_match = name
                
        return best_match if best_score > self.config.pattern_threshold else None

class FungalNetwork:
    def __init__(self, config: NetworkConfig, num_regions: int = 4):
        ray.init()
        self.config = config
        self.regions = [NetworkRegion.remote(config, i) for i in range(num_regions)]
        
    def step(self, dt: float) -> None:
        ray.get([region.update.remote(dt) for region in self.regions])
        
    def get_state(self) -> Dict:
        states = ray.get([region.get_state.remote() for region in self.regions])
        return {
            'positions': torch.cat([s['positions'] for s in states]),
            'resources': torch.cat([s['resources'] for s in states]),
            'connections': torch.block_diag(*[s['connections'] for s in states]),
            'patterns': [s['current_pattern'] for s in states],
            'efficiency': [s['efficiency'] for s in states]
        }

if __name__ == "__main__":
    config = NetworkConfig(
        num_nodes=1000,
        space_dims=(100, 100, 100),
        growth_rate=0.1,
        decay_rate=0.05,
        connection_threshold=1.0
    )
    
    network = FungalNetwork(config)
    for step in range(1000):
        network.step(0.1)
        if step % 100 == 0:
            state = network.get_state()
            logger.info(f"Step {step}, Patterns: {state['patterns']}")
