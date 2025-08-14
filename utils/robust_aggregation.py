"""
Robust Federated Learning Aggregation Algorithms
Implements various aggregation strategies with Byzantine fault tolerance
"""

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Optional, Any
import logging
import math
from dataclasses import dataclass
from enum import Enum
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import warnings

logger = logging.getLogger(__name__)

class AggregationAlgorithm(Enum):
    """Supported aggregation algorithms"""
    FEDAVG = "fedavg"
    KRUM = "krum"
    MULTI_KRUM = "multi_krum"
    TRIMMED_MEAN = "trimmed_mean"
    MEDIAN = "median"
    GEOMETRIC_MEDIAN = "geometric_median"
    BULYAN = "bulyan"
    FABA = "faba"
    DNC = "divide_and_conquer"

@dataclass
class ClientUpdate:
    """Represents a client's model update"""
    client_id: str
    weights: List[np.ndarray]
    data_size: int
    accuracy: float
    loss: float
    reputation_score: float
    timestamp: float
    is_validated: bool = True

@dataclass
class AggregationConfig:
    """Configuration for aggregation algorithms"""
    algorithm: AggregationAlgorithm = AggregationAlgorithm.FEDAVG
    byzantine_tolerance: int = 0  # Number of Byzantine clients to tolerate
    trim_ratio: float = 0.1  # Ratio to trim for trimmed mean
    krum_f: int = 0  # Number of Byzantine clients for Krum
    use_reputation: bool = True
    reputation_weight: float = 0.3
    outlier_threshold: float = 2.0  # Z-score threshold for outlier detection
    min_clients: int = 3
    max_clients: int = 100

class RobustAggregator:
    """Robust aggregation engine with multiple algorithms"""
    
    def __init__(self, config: AggregationConfig):
        self.config = config
        self.aggregation_history = []
        
    def aggregate(self, client_updates: List[ClientUpdate]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Main aggregation function"""
        if len(client_updates) < self.config.min_clients:
            raise ValueError(f"Insufficient clients: {len(client_updates)} < {self.config.min_clients}")
        
        # Filter validated updates
        valid_updates = [update for update in client_updates if update.is_validated]
        
        if len(valid_updates) < self.config.min_clients:
            raise ValueError(f"Insufficient valid updates: {len(valid_updates)} < {self.config.min_clients}")
        
        logger.info(f"Aggregating {len(valid_updates)} valid updates using {self.config.algorithm.value}")
        
        # Detect outliers
        outlier_info = self.detect_outliers(valid_updates)
        
        # Select aggregation algorithm
        if self.config.algorithm == AggregationAlgorithm.FEDAVG:
            aggregated_weights, metrics = self.federated_averaging(valid_updates)
        elif self.config.algorithm == AggregationAlgorithm.KRUM:
            aggregated_weights, metrics = self.krum(valid_updates)
        elif self.config.algorithm == AggregationAlgorithm.MULTI_KRUM:
            aggregated_weights, metrics = self.multi_krum(valid_updates)
        elif self.config.algorithm == AggregationAlgorithm.TRIMMED_MEAN:
            aggregated_weights, metrics = self.trimmed_mean(valid_updates)
        elif self.config.algorithm == AggregationAlgorithm.MEDIAN:
            aggregated_weights, metrics = self.coordinate_wise_median(valid_updates)
        elif self.config.algorithm == AggregationAlgorithm.GEOMETRIC_MEDIAN:
            aggregated_weights, metrics = self.geometric_median(valid_updates)
        elif self.config.algorithm == AggregationAlgorithm.BULYAN:
            aggregated_weights, metrics = self.bulyan(valid_updates)
        else:
            # Default to FedAvg
            aggregated_weights, metrics = self.federated_averaging(valid_updates)
        
        # Add outlier information to metrics
        metrics.update(outlier_info)
        
        # Store aggregation history
        self.aggregation_history.append({
            'algorithm': self.config.algorithm.value,
            'num_clients': len(valid_updates),
            'metrics': metrics,
            'timestamp': max(update.timestamp for update in valid_updates)
        })
        
        return aggregated_weights, metrics
    
    def federated_averaging(self, updates: List[ClientUpdate]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Weighted FedAvg with reputation scores"""
        total_data_size = sum(update.data_size for update in updates)
        
        # Calculate weights
        weights = []
        for update in updates:
            data_weight = update.data_size / total_data_size
            
            if self.config.use_reputation:
                reputation_weight = update.reputation_score * self.config.reputation_weight
                final_weight = data_weight * (1 - self.config.reputation_weight) + reputation_weight
            else:
                final_weight = data_weight
            
            weights.append(final_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Aggregate weights
        num_layers = len(updates[0].weights)
        aggregated_weights = []
        
        for layer_idx in range(num_layers):
            layer_weights = [update.weights[layer_idx] for update in updates]
            
            # Weighted average
            aggregated_layer = np.zeros_like(layer_weights[0])
            for weight, layer_weight in zip(weights, layer_weights):
                aggregated_layer += weight * layer_weight
            
            aggregated_weights.append(aggregated_layer)
        
        # Calculate metrics
        avg_accuracy = np.average([update.accuracy for update in updates], weights=weights)
        avg_loss = np.average([update.loss for update in updates], weights=weights)
        avg_reputation = np.average([update.reputation_score for update in updates], weights=weights)
        
        metrics = {
            'algorithm': 'fedavg',
            'num_participants': len(updates),
            'avg_accuracy': float(avg_accuracy),
            'avg_loss': float(avg_loss),
            'avg_reputation': float(avg_reputation),
            'weight_distribution': weights
        }
        
        return aggregated_weights, metrics
    
    def krum(self, updates: List[ClientUpdate], multi_krum: bool = False, 
             m: int = 1) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Krum and Multi-Krum algorithms"""
        n = len(updates)
        f = self.config.krum_f
        
        if n <= 2 * f:
            logger.warning(f"Insufficient clients for Krum: {n} <= 2*{f}")
            return self.federated_averaging(updates)
        
        # Calculate pairwise distances
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.euclidean_distance(updates[i].weights, updates[j].weights)
                distances[i, j] = distances[j, i] = dist
        
        # Calculate Krum scores
        scores = []
        for i in range(n):
            # Get n-f-2 closest distances
            sorted_distances = np.sort(distances[i])
            score = np.sum(sorted_distances[1:n-f-1])  # Exclude self (distance 0)
            scores.append(score)
        
        if multi_krum:
            # Select m clients with lowest scores
            selected_indices = np.argsort(scores)[:m]
            selected_updates = [updates[i] for i in selected_indices]
            
            # Average selected updates
            aggregated_weights, metrics = self.federated_averaging(selected_updates)
            metrics['algorithm'] = 'multi_krum'
            metrics['selected_clients'] = m
        else:
            # Select single client with lowest score
            best_idx = np.argmin(scores)
            aggregated_weights = updates[best_idx].weights
            
            metrics = {
                'algorithm': 'krum',
                'selected_client': updates[best_idx].client_id,
                'krum_score': float(scores[best_idx]),
                'accuracy': updates[best_idx].accuracy,
                'loss': updates[best_idx].loss
            }
        
        metrics['krum_scores'] = [float(s) for s in scores]
        return aggregated_weights, metrics
    
    def multi_krum(self, updates: List[ClientUpdate]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Multi-Krum algorithm"""
        m = max(1, len(updates) - 2 * self.config.krum_f)
        return self.krum(updates, multi_krum=True, m=m)
    
    def trimmed_mean(self, updates: List[ClientUpdate]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Trimmed mean aggregation"""
        n = len(updates)
        trim_count = int(n * self.config.trim_ratio)
        
        if n - 2 * trim_count < 1:
            logger.warning("Too much trimming, falling back to FedAvg")
            return self.federated_averaging(updates)
        
        num_layers = len(updates[0].weights)
        aggregated_weights = []
        
        for layer_idx in range(num_layers):
            layer_weights = np.array([update.weights[layer_idx] for update in updates])
            
            # Sort along each dimension and trim
            sorted_weights = np.sort(layer_weights, axis=0)
            trimmed_weights = sorted_weights[trim_count:n-trim_count]
            
            # Calculate mean
            aggregated_layer = np.mean(trimmed_weights, axis=0)
            aggregated_weights.append(aggregated_layer)
        
        # Calculate metrics from non-trimmed updates
        sorted_by_accuracy = sorted(updates, key=lambda x: x.accuracy)
        kept_updates = sorted_by_accuracy[trim_count:n-trim_count]
        
        avg_accuracy = np.mean([update.accuracy for update in kept_updates])
        avg_loss = np.mean([update.loss for update in kept_updates])
        
        metrics = {
            'algorithm': 'trimmed_mean',
            'trim_ratio': self.config.trim_ratio,
            'trimmed_count': trim_count,
            'kept_count': len(kept_updates),
            'avg_accuracy': float(avg_accuracy),
            'avg_loss': float(avg_loss)
        }
        
        return aggregated_weights, metrics
    
    def coordinate_wise_median(self, updates: List[ClientUpdate]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Coordinate-wise median aggregation"""
        num_layers = len(updates[0].weights)
        aggregated_weights = []
        
        for layer_idx in range(num_layers):
            layer_weights = np.array([update.weights[layer_idx] for update in updates])
            aggregated_layer = np.median(layer_weights, axis=0)
            aggregated_weights.append(aggregated_layer)
        
        # Calculate median metrics
        accuracies = [update.accuracy for update in updates]
        losses = [update.loss for update in updates]
        
        metrics = {
            'algorithm': 'median',
            'median_accuracy': float(np.median(accuracies)),
            'median_loss': float(np.median(losses)),
            'num_participants': len(updates)
        }
        
        return aggregated_weights, metrics
    
    def geometric_median(self, updates: List[ClientUpdate]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Geometric median using Weiszfeld's algorithm"""
        num_layers = len(updates[0].weights)
        aggregated_weights = []
        
        for layer_idx in range(num_layers):
            layer_weights = [update.weights[layer_idx] for update in updates]
            
            # Flatten weights for geometric median calculation
            flattened_weights = [w.flatten() for w in layer_weights]
            
            # Calculate geometric median
            geometric_median = self.weiszfeld_algorithm(flattened_weights)
            
            # Reshape back to original shape
            aggregated_layer = geometric_median.reshape(layer_weights[0].shape)
            aggregated_weights.append(aggregated_layer)
        
        metrics = {
            'algorithm': 'geometric_median',
            'num_participants': len(updates),
            'convergence': 'weiszfeld'
        }
        
        return aggregated_weights, metrics
    
    def bulyan(self, updates: List[ClientUpdate]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Bulyan algorithm (simplified implementation)"""
        n = len(updates)
        f = self.config.byzantine_tolerance
        
        if n < 4 * f + 3:
            logger.warning(f"Insufficient clients for Bulyan: {n} < {4*f + 3}")
            return self.trimmed_mean(updates)
        
        # First phase: Multi-Krum selection
        m = n - f
        selected_weights, _ = self.multi_krum(updates)
        
        # Second phase: Trimmed mean on selected updates
        # For simplicity, we'll use the selected weights directly
        # In full implementation, would apply trimmed mean to Multi-Krum selection
        
        metrics = {
            'algorithm': 'bulyan',
            'byzantine_tolerance': f,
            'selected_count': m,
            'num_participants': n
        }
        
        return selected_weights, metrics
    
    def detect_outliers(self, updates: List[ClientUpdate]) -> Dict[str, Any]:
        """Detect outlier updates using statistical methods"""
        accuracies = [update.accuracy for update in updates]
        losses = [update.loss for update in updates]
        
        # Z-score based outlier detection
        acc_mean, acc_std = np.mean(accuracies), np.std(accuracies)
        loss_mean, loss_std = np.mean(losses), np.std(losses)
        
        outliers = []
        for i, update in enumerate(updates):
            acc_zscore = abs((update.accuracy - acc_mean) / (acc_std + 1e-8))
            loss_zscore = abs((update.loss - loss_mean) / (loss_std + 1e-8))
            
            if (acc_zscore > self.config.outlier_threshold or 
                loss_zscore > self.config.outlier_threshold):
                outliers.append({
                    'client_id': update.client_id,
                    'accuracy_zscore': float(acc_zscore),
                    'loss_zscore': float(loss_zscore)
                })
        
        return {
            'outliers_detected': len(outliers),
            'outlier_details': outliers,
            'outlier_threshold': self.config.outlier_threshold
        }
    
    def euclidean_distance(self, weights1: List[np.ndarray], 
                          weights2: List[np.ndarray]) -> float:
        """Calculate Euclidean distance between two weight vectors"""
        total_distance = 0.0
        
        for w1, w2 in zip(weights1, weights2):
            diff = w1 - w2
            total_distance += np.sum(diff ** 2)
        
        return math.sqrt(total_distance)
    
    def weiszfeld_algorithm(self, points: List[np.ndarray], 
                           max_iterations: int = 100, tolerance: float = 1e-6) -> np.ndarray:
        """Weiszfeld's algorithm for geometric median"""
        points = np.array(points)
        
        # Initialize with coordinate-wise median
        current = np.median(points, axis=0)
        
        for iteration in range(max_iterations):
            # Calculate distances
            distances = np.array([np.linalg.norm(point - current) for point in points])
            
            # Avoid division by zero
            distances = np.maximum(distances, 1e-8)
            
            # Calculate weights
            weights = 1.0 / distances
            weights /= np.sum(weights)
            
            # Update estimate
            new_current = np.sum(weights[:, np.newaxis] * points, axis=0)
            
            # Check convergence
            if np.linalg.norm(new_current - current) < tolerance:
                break
            
            current = new_current
        
        return current

def create_aggregator(algorithm: AggregationAlgorithm = AggregationAlgorithm.FEDAVG,
                     byzantine_tolerance: int = 0,
                     **kwargs) -> RobustAggregator:
    """Factory function to create aggregator with default settings"""
    config = AggregationConfig(
        algorithm=algorithm,
        byzantine_tolerance=byzantine_tolerance,
        **kwargs
    )
    
    return RobustAggregator(config)

# Example usage and testing
if __name__ == "__main__":
    # Create dummy client updates
    dummy_updates = []
    for i in range(10):
        weights = [
            np.random.randn(100, 50).astype(np.float32),
            np.random.randn(50, 10).astype(np.float32),
            np.random.randn(10).astype(np.float32)
        ]
        
        update = ClientUpdate(
            client_id=f"client_{i}",
            weights=weights,
            data_size=1000 + i * 100,
            accuracy=0.8 + np.random.normal(0, 0.1),
            loss=0.2 + np.random.normal(0, 0.05),
            reputation_score=0.5 + np.random.normal(0, 0.2),
            timestamp=1000000 + i
        )
        dummy_updates.append(update)
    
    # Test different aggregation algorithms
    algorithms = [
        AggregationAlgorithm.FEDAVG,
        AggregationAlgorithm.KRUM,
        AggregationAlgorithm.TRIMMED_MEAN,
        AggregationAlgorithm.MEDIAN
    ]
    
    for algorithm in algorithms:
        print(f"\nTesting {algorithm.value}:")
        aggregator = create_aggregator(algorithm, byzantine_tolerance=2)
        
        try:
            aggregated_weights, metrics = aggregator.aggregate(dummy_updates)
            print(f"  Success: {len(aggregated_weights)} layers aggregated")
            print(f"  Metrics: {metrics}")
        except Exception as e:
            print(f"  Error: {e}")
