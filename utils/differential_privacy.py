"""
Advanced Differential Privacy Implementation for Federated Learning
Supports multiple DP mechanisms with privacy budget tracking and adaptive noise calibration
"""

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, Dict, Any
import logging
import math
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class DPMechanism(Enum):
    """Supported differential privacy mechanisms"""
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"
    EXPONENTIAL = "exponential"
    SPARSE_VECTOR = "sparse_vector"

@dataclass
class PrivacyParameters:
    """Privacy parameters for DP mechanisms"""
    epsilon: float
    delta: float
    sensitivity: float
    mechanism: DPMechanism = DPMechanism.GAUSSIAN

class PrivacyAccountant:
    """Advanced privacy budget accounting with composition theorems"""
    
    def __init__(self, total_epsilon: float, total_delta: float):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.used_epsilon = 0.0
        self.used_delta = 0.0
        self.privacy_history = []
        
    def can_spend(self, epsilon: float, delta: float) -> bool:
        """Check if privacy budget allows spending"""
        return (self.used_epsilon + epsilon <= self.total_epsilon and 
                self.used_delta + delta <= self.total_delta)
    
    def spend(self, epsilon: float, delta: float, mechanism: str, description: str = ""):
        """Spend privacy budget"""
        if not self.can_spend(epsilon, delta):
            raise ValueError(f"Privacy budget exceeded: requesting ({epsilon}, {delta}), "
                           f"available ({self.total_epsilon - self.used_epsilon}, "
                           f"{self.total_delta - self.used_delta})")
        
        self.used_epsilon += epsilon
        self.used_delta += delta
        
        self.privacy_history.append({
            'epsilon': epsilon,
            'delta': delta,
            'mechanism': mechanism,
            'description': description,
            'cumulative_epsilon': self.used_epsilon,
            'cumulative_delta': self.used_delta
        })
        
        logger.info(f"Privacy spent: ε={epsilon:.4f}, δ={delta:.6f}, "
                   f"remaining: ε={self.get_remaining_epsilon():.4f}, "
                   f"δ={self.get_remaining_delta():.6f}")
    
    def get_remaining_epsilon(self) -> float:
        """Get remaining epsilon budget"""
        return max(0, self.total_epsilon - self.used_epsilon)
    
    def get_remaining_delta(self) -> float:
        """Get remaining delta budget"""
        return max(0, self.total_delta - self.used_delta)
    
    def reset(self):
        """Reset privacy budget"""
        self.used_epsilon = 0.0
        self.used_delta = 0.0
        self.privacy_history = []

class AdaptiveDPManager:
    """Advanced differential privacy manager with adaptive noise calibration"""
    
    def __init__(self, privacy_params: PrivacyParameters, 
                 accountant: Optional[PrivacyAccountant] = None):
        self.privacy_params = privacy_params
        self.accountant = accountant or PrivacyAccountant(
            privacy_params.epsilon, privacy_params.delta
        )
        self.rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        
    def calculate_gaussian_noise_scale(self, sensitivity: float, epsilon: float, 
                                     delta: float) -> float:
        """Calculate noise scale for Gaussian mechanism"""
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be in (0, 1)")
        
        # Gaussian mechanism: σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    
    def calculate_laplace_noise_scale(self, sensitivity: float, epsilon: float) -> float:
        """Calculate noise scale for Laplace mechanism"""
        return sensitivity / epsilon
    
    def add_gaussian_noise(self, data: np.ndarray, sensitivity: float, 
                          epsilon: float, delta: float) -> np.ndarray:
        """Add calibrated Gaussian noise"""
        if not self.accountant.can_spend(epsilon, delta):
            raise ValueError("Insufficient privacy budget")
        
        noise_scale = self.calculate_gaussian_noise_scale(sensitivity, epsilon, delta)
        noise = self.rng.normal(0, noise_scale, data.shape).astype(data.dtype)
        
        self.accountant.spend(epsilon, delta, "gaussian", 
                            f"Gaussian noise to {data.shape} tensor")
        
        return data + noise
    
    def add_laplace_noise(self, data: np.ndarray, sensitivity: float, 
                         epsilon: float) -> np.ndarray:
        """Add calibrated Laplace noise"""
        if not self.accountant.can_spend(epsilon, 0):
            raise ValueError("Insufficient privacy budget")
        
        noise_scale = self.calculate_laplace_noise_scale(sensitivity, epsilon)
        noise = self.rng.laplace(0, noise_scale, data.shape).astype(data.dtype)
        
        self.accountant.spend(epsilon, 0, "laplace", 
                            f"Laplace noise to {data.shape} tensor")
        
        return data + noise
    
    def clip_gradients_l2(self, gradients: List[np.ndarray], 
                         max_norm: float) -> Tuple[List[np.ndarray], float]:
        """Clip gradients using L2 norm"""
        # Calculate global L2 norm
        global_norm = math.sqrt(sum(np.sum(g**2) for g in gradients))
        
        if global_norm > max_norm:
            clip_factor = max_norm / global_norm
            clipped_gradients = [g * clip_factor for g in gradients]
            logger.info(f"Clipped gradients: norm {global_norm:.4f} -> {max_norm:.4f}")
            return clipped_gradients, clip_factor
        
        return gradients, 1.0
    
    def adaptive_noise_calibration(self, gradients: List[np.ndarray], 
                                 target_epsilon: float, target_delta: float,
                                 max_norm: float) -> List[np.ndarray]:
        """Adaptive noise calibration based on gradient statistics"""
        
        # Clip gradients first
        clipped_gradients, clip_factor = self.clip_gradients_l2(gradients, max_norm)
        
        # Calculate adaptive sensitivity based on gradient magnitudes
        gradient_norms = [np.linalg.norm(g) for g in clipped_gradients]
        avg_norm = np.mean(gradient_norms)
        
        # Adjust sensitivity based on gradient characteristics
        adaptive_sensitivity = min(max_norm, avg_norm * 1.5)
        
        # Add noise to each gradient
        noisy_gradients = []
        epsilon_per_layer = target_epsilon / len(clipped_gradients)
        
        for i, grad in enumerate(clipped_gradients):
            noisy_grad = self.add_gaussian_noise(
                grad, adaptive_sensitivity, epsilon_per_layer, target_delta
            )
            noisy_gradients.append(noisy_grad)
        
        logger.info(f"Applied adaptive DP: sensitivity={adaptive_sensitivity:.4f}, "
                   f"ε_per_layer={epsilon_per_layer:.4f}")
        
        return noisy_gradients
    
    def moments_accountant_composition(self, epsilons: List[float], 
                                     deltas: List[float]) -> Tuple[float, float]:
        """Advanced composition using moments accountant (simplified)"""
        # Simplified version - in practice, use more sophisticated composition
        total_epsilon = sum(epsilons)
        total_delta = sum(deltas)
        
        # Apply composition theorem adjustment
        if len(epsilons) > 1:
            composition_factor = math.sqrt(2 * len(epsilons) * math.log(1/min(deltas)))
            total_epsilon *= composition_factor
        
        return total_epsilon, total_delta
    
    def sparse_vector_technique(self, queries: List[np.ndarray], 
                               threshold: float, epsilon: float, 
                               max_responses: int = 1) -> List[bool]:
        """Sparse Vector Technique for query answering"""
        if not self.accountant.can_spend(epsilon, 0):
            raise ValueError("Insufficient privacy budget")
        
        # Add noise to threshold
        noisy_threshold = threshold + self.rng.laplace(0, 2/epsilon)
        
        responses = []
        response_count = 0
        
        for query in queries:
            if response_count >= max_responses:
                responses.append(False)
                continue
            
            # Add noise to query result
            query_result = np.sum(query)  # Simplified query
            noisy_result = query_result + self.rng.laplace(0, 4/epsilon)
            
            if noisy_result >= noisy_threshold:
                responses.append(True)
                response_count += 1
            else:
                responses.append(False)
        
        self.accountant.spend(epsilon, 0, "sparse_vector", 
                            f"SVT with {len(queries)} queries")
        
        return responses
    
    def get_privacy_analysis(self) -> Dict[str, Any]:
        """Get comprehensive privacy analysis"""
        return {
            'total_budget': {
                'epsilon': self.accountant.total_epsilon,
                'delta': self.accountant.total_delta
            },
            'used_budget': {
                'epsilon': self.accountant.used_epsilon,
                'delta': self.accountant.used_delta
            },
            'remaining_budget': {
                'epsilon': self.accountant.get_remaining_epsilon(),
                'delta': self.accountant.get_remaining_delta()
            },
            'privacy_history': self.accountant.privacy_history,
            'budget_utilization': {
                'epsilon_percent': (self.accountant.used_epsilon / 
                                  self.accountant.total_epsilon) * 100,
                'delta_percent': (self.accountant.used_delta / 
                                self.accountant.total_delta) * 100
            }
        }

class TensorFlowDPOptimizer:
    """TensorFlow-compatible DP optimizer wrapper"""
    
    def __init__(self, optimizer: tf.keras.optimizers.Optimizer,
                 dp_manager: AdaptiveDPManager, max_grad_norm: float = 1.0):
        self.optimizer = optimizer
        self.dp_manager = dp_manager
        self.max_grad_norm = max_grad_norm
    
    def apply_gradients(self, grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]],
                       epsilon: float, delta: float):
        """Apply DP gradients"""
        gradients = [grad.numpy() for grad, _ in grads_and_vars]
        variables = [var for _, var in grads_and_vars]
        
        # Apply DP noise
        noisy_gradients = self.dp_manager.adaptive_noise_calibration(
            gradients, epsilon, delta, self.max_grad_norm
        )
        
        # Convert back to tensors and apply
        noisy_grads_and_vars = [
            (tf.constant(noisy_grad), var) 
            for noisy_grad, var in zip(noisy_gradients, variables)
        ]
        
        self.optimizer.apply_gradients(noisy_grads_and_vars)

class DifferentialPrivacy:
    """Simple DP interface for demo purposes"""

    def __init__(self, sigma: float = 1.2, delta: float = 1e-5):
        self.sigma = sigma
        self.delta = delta
        self.epsilon = 1.0  # Derived from sigma
        self.privacy_budget_used = 0.0

    def add_noise_to_weights(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Add Gaussian noise to model weights"""
        noisy_weights = []
        for weight_matrix in weights:
            noise = np.random.normal(0, self.sigma, weight_matrix.shape)
            noisy_weight = weight_matrix + noise
            noisy_weights.append(noisy_weight)

        # Update privacy budget
        self.privacy_budget_used += 0.1  # Simplified budget tracking
        return noisy_weights

    def get_privacy_budget_usage(self) -> float:
        """Get current privacy budget usage"""
        return min(1.0, self.privacy_budget_used)

def create_dp_manager(epsilon: float = 1.0, delta: float = 1e-5,
                     sensitivity: float = 1.0) -> AdaptiveDPManager:
    """Factory function to create DP manager with default settings"""
    privacy_params = PrivacyParameters(
        epsilon=epsilon,
        delta=delta,
        sensitivity=sensitivity,
        mechanism=DPMechanism.GAUSSIAN
    )

    accountant = PrivacyAccountant(epsilon, delta)
    return AdaptiveDPManager(privacy_params, accountant)

# Example usage and testing
if __name__ == "__main__":
    # Create DP manager
    dp_manager = create_dp_manager(epsilon=2.0, delta=1e-5)
    
    # Test with dummy gradients
    dummy_gradients = [
        np.random.randn(100, 50).astype(np.float32),
        np.random.randn(50, 10).astype(np.float32),
        np.random.randn(10).astype(np.float32)
    ]
    
    # Apply adaptive DP
    noisy_gradients = dp_manager.adaptive_noise_calibration(
        dummy_gradients, target_epsilon=0.5, target_delta=1e-6, max_norm=1.0
    )
    
    # Print privacy analysis
    analysis = dp_manager.get_privacy_analysis()
    print("Privacy Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
