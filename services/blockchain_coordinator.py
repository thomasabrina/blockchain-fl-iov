#!/usr/bin/env python3
"""
Blockchain Coordinator Service
Integrates FL training with blockchain smart contracts for decentralized coordination
"""

import json
import logging
import time
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np

from blockchain.fabric_client import get_fabric_client, FabricClient
from utils.robust_aggregation import RobustAggregator, AggregationAlgorithm, ClientUpdate, AggregationConfig

logger = logging.getLogger(__name__)

@dataclass
class BlockchainFLRound:
    """Represents a blockchain-coordinated FL round"""
    round_id: str
    algorithm: str
    participant_count: int
    byzantine_tolerance: int
    start_time: float
    status: str = "scheduled"  # scheduled, in_progress, completed, failed
    completion_time: Optional[float] = None
    global_model_hash: Optional[str] = None
    aggregation_metrics: Optional[Dict] = None

class BlockchainCoordinator:
    """Coordinates FL training through blockchain smart contracts"""
    
    def __init__(self):
        self.fabric_client: FabricClient = get_fabric_client()
        # Initialize aggregator with default config
        default_config = AggregationConfig()
        self.aggregator = RobustAggregator(default_config)
        self.current_round: Optional[BlockchainFLRound] = None
        self.round_history: List[BlockchainFLRound] = []
        
    def initialize_blockchain(self) -> bool:
        """Initialize blockchain network and deploy contracts"""
        logger.info("🔗 Initializing blockchain network...")
        
        # Start Fabric network
        if not self.fabric_client.start_network():
            logger.error("Failed to start Fabric network")
            return False
        
        # Deploy chaincode
        if not self.fabric_client.deploy_chaincode():
            logger.error("Failed to deploy chaincode")
            return False
            
        logger.info("✅ Blockchain network initialized successfully")
        return True
    
    def create_fl_round(self, round_id: str, participants: List[str], 
                       byzantine_tolerance: int = 0) -> bool:
        """Create new FL round through smart contract"""
        try:
            logger.info(f"🚀 Creating FL round {round_id} with {len(participants)} participants")
            
            # Select optimal aggregation algorithm based on threat level
            algorithm = self._select_algorithm(len(participants), byzantine_tolerance)
            
            # Create round object
            fl_round = BlockchainFLRound(
                round_id=round_id,
                algorithm=algorithm.value,
                participant_count=len(participants),
                byzantine_tolerance=byzantine_tolerance,
                start_time=time.time()
            )
            
            # Invoke smart contract to create round
            result = self.fabric_client.invoke_chaincode(
                "CreateAggregationRound",
                [
                    round_id,
                    algorithm.value,
                    str(len(participants)),
                    str(byzantine_tolerance),
                    json.dumps(participants)
                ]
            )
            
            if result and result.get("success"):
                fl_round.status = "in_progress"
                self.current_round = fl_round
                logger.info(f"✅ FL round {round_id} created successfully")
                return True
            else:
                logger.error(f"Failed to create FL round: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating FL round: {e}")
            return False
    
    def submit_model_update(self, client_id: str, model_weights: List[np.ndarray],
                          data_size: int, accuracy: float, loss: float) -> bool:
        """Submit model update through smart contract"""
        try:
            # Calculate model hash
            model_hash = self._calculate_model_hash(model_weights)
            
            logger.info(f"📤 Submitting model update from client {client_id}")
            
            # Submit to smart contract
            result = self.fabric_client.invoke_chaincode(
                "SubmitModelUpdate",
                [
                    self.current_round.round_id,
                    client_id,
                    model_hash,
                    str(data_size),
                    str(accuracy),
                    str(loss)
                ]
            )
            
            if result and result.get("success"):
                logger.info(f"✅ Model update from {client_id} submitted successfully")
                return True
            else:
                logger.error(f"Failed to submit model update: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error submitting model update: {e}")
            return False
    
    def perform_aggregation(self, client_updates: List[ClientUpdate]) -> Optional[List[np.ndarray]]:
        """Perform robust aggregation using blockchain coordination"""
        try:
            if not self.current_round:
                logger.error("No active FL round")
                return None
                
            logger.info(f"🔄 Performing {self.current_round.algorithm} aggregation...")
            
            # Configure aggregation
            config = AggregationConfig(
                algorithm=AggregationAlgorithm(self.current_round.algorithm),
                byzantine_tolerance=self.current_round.byzantine_tolerance,
                use_reputation=True
            )
            
            # Perform aggregation
            aggregated_weights, metrics = self.aggregator.aggregate(client_updates, config)
            
            if aggregated_weights is None:
                logger.error("Aggregation failed")
                return None
            
            # Calculate aggregated model hash
            aggregated_hash = self._calculate_model_hash(aggregated_weights)
            
            # Record aggregation result in blockchain
            result = self.fabric_client.invoke_chaincode(
                "RecordAggregationResult",
                [
                    self.current_round.round_id,
                    aggregated_hash,
                    json.dumps(metrics)
                ]
            )
            
            if result and result.get("success"):
                self.current_round.global_model_hash = aggregated_hash
                self.current_round.aggregation_metrics = metrics
                logger.info(f"✅ Aggregation completed successfully")
                return aggregated_weights
            else:
                logger.error(f"Failed to record aggregation result: {result}")
                return None
                
        except Exception as e:
            logger.error(f"Error performing aggregation: {e}")
            return None
    
    def complete_fl_round(self) -> bool:
        """Complete current FL round"""
        try:
            if not self.current_round:
                logger.error("No active FL round")
                return False
                
            logger.info(f"🏁 Completing FL round {self.current_round.round_id}")
            
            # Mark round as completed
            result = self.fabric_client.invoke_chaincode(
                "CompleteAggregationRound",
                [self.current_round.round_id]
            )
            
            if result and result.get("success"):
                self.current_round.status = "completed"
                self.current_round.completion_time = time.time()
                self.round_history.append(self.current_round)
                self.current_round = None
                logger.info("✅ FL round completed successfully")
                return True
            else:
                logger.error(f"Failed to complete FL round: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error completing FL round: {e}")
            return False
    
    def get_round_status(self, round_id: str) -> Optional[Dict]:
        """Query round status from blockchain"""
        try:
            result = self.fabric_client.query_chaincode(
                "GetAggregationRound",
                [round_id]
            )
            
            if result and result.get("success"):
                return result
            else:
                logger.error(f"Failed to query round status: {result}")
                return None
                
        except Exception as e:
            logger.error(f"Error querying round status: {e}")
            return None
    
    def shutdown(self) -> bool:
        """Shutdown blockchain network"""
        logger.info("🔌 Shutting down blockchain network...")
        return self.fabric_client.stop_network()
    
    def _select_algorithm(self, num_participants: int, byzantine_tolerance: int) -> AggregationAlgorithm:
        """Select optimal aggregation algorithm based on threat level"""
        if byzantine_tolerance == 0:
            return AggregationAlgorithm.FEDAVG
        elif byzantine_tolerance <= num_participants // 4:
            return AggregationAlgorithm.KRUM
        elif byzantine_tolerance <= num_participants // 3:
            return AggregationAlgorithm.TRIMMED_MEAN
        else:
            return AggregationAlgorithm.GEOMETRIC_MEDIAN
    
    def _calculate_model_hash(self, model_weights: List[np.ndarray]) -> str:
        """Calculate hash of model weights"""
        # Concatenate all weights
        all_weights = np.concatenate([w.flatten() for w in model_weights])
        # Calculate SHA-256 hash
        hash_obj = hashlib.sha256(all_weights.tobytes())
        return hash_obj.hexdigest()

# Singleton instance
_blockchain_coordinator = None

def get_blockchain_coordinator() -> BlockchainCoordinator:
    """Get singleton blockchain coordinator instance"""
    global _blockchain_coordinator
    if _blockchain_coordinator is None:
        _blockchain_coordinator = BlockchainCoordinator()
    return _blockchain_coordinator
