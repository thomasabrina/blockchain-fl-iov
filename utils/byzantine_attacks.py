#!/usr/bin/env python3
"""
Byzantine Attack Implementations
Simulates various malicious behaviors to test system robustness
"""

import numpy as np
import tensorflow as tf
import logging
from typing import List, Dict, Any, Tuple
import random

logger = logging.getLogger(__name__)

class ByzantineAttacker:
    """Implements various Byzantine attacks for FL robustness testing"""
    
    def __init__(self, attack_type: str = 'none', intensity: float = 0.1):
        self.attack_type = attack_type
        self.intensity = intensity
        self.attack_history = []
        
        logger.info(f"Byzantine attacker initialized: {attack_type} (intensity: {intensity})")
    
    def apply_attack(self, client_id: int, weights: List[np.ndarray], 
                    x_data: np.ndarray = None, y_data: np.ndarray = None) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """Apply Byzantine attack based on attack type"""
        
        if self.attack_type == 'none':
            return weights, x_data, y_data
        
        logger.warning(f"Client {client_id}: Applying {self.attack_type} attack (intensity: {self.intensity})")
        
        if self.attack_type == 'label_flip':
            return weights, x_data, self._label_flip_attack(y_data)
        elif self.attack_type == 'gaussian_noise':
            return self._gaussian_noise_attack(weights), x_data, y_data
        elif self.attack_type == 'model_poison':
            return self._model_poison_attack(weights), x_data, y_data
        elif self.attack_type == 'sign_flip':
            return self._sign_flip_attack(weights), x_data, y_data
        elif self.attack_type == 'zero_gradient':
            return self._zero_gradient_attack(weights), x_data, y_data
        else:
            logger.error(f"Unknown attack type: {self.attack_type}")
            return weights, x_data, y_data
    
    def _label_flip_attack(self, y_data: np.ndarray) -> np.ndarray:
        """Flip labels randomly based on intensity"""
        if y_data is None:
            return y_data
        
        y_attacked = y_data.copy()
        num_samples = len(y_data)
        num_flip = int(num_samples * self.intensity)
        
        # Randomly select samples to flip
        flip_indices = np.random.choice(num_samples, num_flip, replace=False)
        
        # Get unique classes
        if len(y_data.shape) > 1:  # One-hot encoded
            num_classes = y_data.shape[1]
            for idx in flip_indices:
                # Flip to random different class
                current_class = np.argmax(y_data[idx])
                new_class = np.random.choice([c for c in range(num_classes) if c != current_class])
                y_attacked[idx] = np.zeros(num_classes)
                y_attacked[idx][new_class] = 1
        else:  # Integer labels
            num_classes = len(np.unique(y_data))
            for idx in flip_indices:
                current_class = y_data[idx]
                new_class = np.random.choice([c for c in range(num_classes) if c != current_class])
                y_attacked[idx] = new_class
        
        self.attack_history.append({
            'type': 'label_flip',
            'samples_flipped': num_flip,
            'total_samples': num_samples,
            'flip_rate': num_flip / num_samples
        })
        
        logger.info(f"Label flip attack: {num_flip}/{num_samples} labels flipped ({num_flip/num_samples:.2%})")
        return y_attacked
    
    def _gaussian_noise_attack(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Add Gaussian noise to model weights"""
        attacked_weights = []
        total_params = 0
        
        for weight_matrix in weights:
            # Calculate noise scale based on weight magnitude and intensity
            weight_std = np.std(weight_matrix)
            noise_scale = weight_std * self.intensity
            
            # Add Gaussian noise
            noise = np.random.normal(0, noise_scale, weight_matrix.shape)
            attacked_weight = weight_matrix + noise
            attacked_weights.append(attacked_weight)
            
            total_params += weight_matrix.size
        
        self.attack_history.append({
            'type': 'gaussian_noise',
            'noise_scale': noise_scale,
            'total_params': total_params,
            'intensity': self.intensity
        })
        
        logger.info(f"Gaussian noise attack: Added noise to {total_params} parameters (scale: {noise_scale:.6f})")
        return attacked_weights
    
    def _model_poison_attack(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Poison model by scaling weights maliciously"""
        attacked_weights = []
        
        # Scale factor based on intensity (higher intensity = more aggressive scaling)
        scale_factor = 1.0 + (self.intensity * 10)  # Can scale up to 2x with intensity=0.1
        
        for weight_matrix in weights:
            # Apply malicious scaling
            attacked_weight = weight_matrix * scale_factor
            attacked_weights.append(attacked_weight)
        
        self.attack_history.append({
            'type': 'model_poison',
            'scale_factor': scale_factor,
            'intensity': self.intensity
        })
        
        logger.info(f"Model poison attack: Scaled weights by factor {scale_factor:.2f}")
        return attacked_weights
    
    def _sign_flip_attack(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Flip signs of random weights"""
        attacked_weights = []
        total_flipped = 0
        
        for weight_matrix in weights:
            attacked_weight = weight_matrix.copy()
            
            # Randomly select weights to flip based on intensity
            num_weights = weight_matrix.size
            num_flip = int(num_weights * self.intensity)
            
            if num_flip > 0:
                flat_weights = attacked_weight.flatten()
                flip_indices = np.random.choice(num_weights, num_flip, replace=False)
                flat_weights[flip_indices] *= -1
                attacked_weight = flat_weights.reshape(weight_matrix.shape)
                total_flipped += num_flip
            
            attacked_weights.append(attacked_weight)
        
        self.attack_history.append({
            'type': 'sign_flip',
            'weights_flipped': total_flipped,
            'flip_rate': self.intensity
        })
        
        logger.info(f"Sign flip attack: Flipped {total_flipped} weight signs")
        return attacked_weights
    
    def _zero_gradient_attack(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Set random weights to zero (gradient nullification)"""
        attacked_weights = []
        total_zeroed = 0
        
        for weight_matrix in weights:
            attacked_weight = weight_matrix.copy()
            
            # Randomly zero out weights based on intensity
            num_weights = weight_matrix.size
            num_zero = int(num_weights * self.intensity)
            
            if num_zero > 0:
                flat_weights = attacked_weight.flatten()
                zero_indices = np.random.choice(num_weights, num_zero, replace=False)
                flat_weights[zero_indices] = 0
                attacked_weight = flat_weights.reshape(weight_matrix.shape)
                total_zeroed += num_zero
            
            attacked_weights.append(attacked_weight)
        
        self.attack_history.append({
            'type': 'zero_gradient',
            'weights_zeroed': total_zeroed,
            'zero_rate': self.intensity
        })
        
        logger.info(f"Zero gradient attack: Zeroed {total_zeroed} weights")
        return attacked_weights
    
    def get_attack_statistics(self) -> Dict[str, Any]:
        """Get statistics about performed attacks"""
        if not self.attack_history:
            return {'total_attacks': 0, 'attack_types': []}
        
        attack_types = [attack['type'] for attack in self.attack_history]
        unique_types = list(set(attack_types))
        
        stats = {
            'total_attacks': len(self.attack_history),
            'attack_types': unique_types,
            'attack_type_counts': {atype: attack_types.count(atype) for atype in unique_types},
            'average_intensity': self.intensity,
            'attack_history': self.attack_history[-10:]  # Last 10 attacks
        }
        
        return stats

class ByzantineClientManager:
    """Manages Byzantine clients in the FL system"""
    
    def __init__(self, total_clients: int, byzantine_clients: int, attack_type: str = 'none', intensity: float = 0.1):
        self.total_clients = total_clients
        self.byzantine_clients = byzantine_clients
        self.attack_type = attack_type
        self.intensity = intensity
        
        # Randomly select which clients are Byzantine
        self.byzantine_client_ids = random.sample(range(total_clients), byzantine_clients)
        
        # Create attackers for Byzantine clients
        self.attackers = {}
        for client_id in self.byzantine_client_ids:
            self.attackers[client_id] = ByzantineAttacker(attack_type, intensity)
        
        logger.info(f"Byzantine client manager initialized:")
        logger.info(f"  Total clients: {total_clients}")
        logger.info(f"  Byzantine clients: {byzantine_clients}")
        logger.info(f"  Byzantine client IDs: {self.byzantine_client_ids}")
        logger.info(f"  Attack type: {attack_type}")
        logger.info(f"  Attack intensity: {intensity}")
    
    def is_byzantine(self, client_id: int) -> bool:
        """Check if a client is Byzantine"""
        return client_id in self.byzantine_client_ids
    
    def apply_attack(self, client_id: int, weights: List[np.ndarray], 
                    x_data: np.ndarray = None, y_data: np.ndarray = None) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """Apply attack if client is Byzantine"""
        if self.is_byzantine(client_id):
            return self.attackers[client_id].apply_attack(client_id, weights, x_data, y_data)
        else:
            return weights, x_data, y_data
    
    def get_byzantine_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about Byzantine behavior"""
        stats = {
            'total_clients': self.total_clients,
            'byzantine_clients': self.byzantine_clients,
            'byzantine_ratio': self.byzantine_clients / self.total_clients,
            'byzantine_client_ids': self.byzantine_client_ids,
            'attack_type': self.attack_type,
            'attack_intensity': self.intensity,
            'individual_stats': {}
        }
        
        for client_id, attacker in self.attackers.items():
            stats['individual_stats'][client_id] = attacker.get_attack_statistics()
        
        return stats

def create_byzantine_scenario(scenario_name: str, total_clients: int) -> Dict[str, Any]:
    """Create predefined Byzantine attack scenarios"""
    scenarios = {
        'light_attack': {
            'byzantine_clients': max(1, total_clients // 10),  # 10% Byzantine
            'attack_type': 'gaussian_noise',
            'intensity': 0.05
        },
        'moderate_attack': {
            'byzantine_clients': max(2, total_clients // 5),   # 20% Byzantine
            'attack_type': 'model_poison',
            'intensity': 0.1
        },
        'heavy_attack': {
            'byzantine_clients': max(3, total_clients // 3),   # 33% Byzantine
            'attack_type': 'sign_flip',
            'intensity': 0.2
        },
        'label_attack': {
            'byzantine_clients': max(2, total_clients // 4),   # 25% Byzantine
            'attack_type': 'label_flip',
            'intensity': 0.3
        },
        'mixed_attack': {
            'byzantine_clients': max(3, total_clients // 4),   # 25% Byzantine
            'attack_type': random.choice(['gaussian_noise', 'model_poison', 'sign_flip']),
            'intensity': random.uniform(0.1, 0.3)
        }
    }
    
    if scenario_name not in scenarios:
        logger.error(f"Unknown scenario: {scenario_name}")
        return scenarios['light_attack']
    
    scenario = scenarios[scenario_name]
    logger.info(f"Created Byzantine scenario '{scenario_name}':")
    logger.info(f"  Byzantine clients: {scenario['byzantine_clients']}/{total_clients}")
    logger.info(f"  Attack type: {scenario['attack_type']}")
    logger.info(f"  Intensity: {scenario['intensity']}")
    
    return scenario
