#!/usr/bin/env python3
"""
Real-time Monitoring System for Blockchain-Enhanced Federated Learning
Monitors three layers: Blockchain, Vehicles/Clients, Model Training, and Attacks
"""

import time
import json
import threading
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import subprocess
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BlockchainStatus:
    """Blockchain layer status"""
    network_active: bool = False
    orderer_nodes: int = 0
    peer_nodes: int = 0
    channels: List[str] = None
    latest_block: int = 0
    transactions_per_second: float = 0.0
    consensus_type: str = "raft"
    network_latency: float = 0.0
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = []

@dataclass
class VehicleStatus:
    """Vehicle/Client layer status"""
    client_id: int
    status: str = "idle"  # idle, training, uploading, byzantine
    location: tuple = (0.0, 0.0)  # (lat, lon)
    data_samples: int = 0
    model_accuracy: float = 0.0
    training_loss: float = 0.0
    is_byzantine: bool = False
    attack_type: str = "none"
    last_update: datetime = None
    
    def __post_init__(self):
        if self.last_update is None:
            self.last_update = datetime.now()

@dataclass
class TrainingStatus:
    """Model training layer status"""
    current_round: int = 0
    total_rounds: int = 0
    global_accuracy: float = 0.0
    global_loss: float = 0.0
    participating_clients: int = 0
    aggregation_method: str = "fedavg"
    convergence_rate: float = 0.0
    training_time: float = 0.0
    model_parameters: int = 0

@dataclass
class AttackStatus:
    """Attack monitoring status"""
    total_attacks: int = 0
    active_attackers: List[int] = None
    attack_types: Dict[str, int] = None
    detection_rate: float = 0.0
    mitigation_active: bool = False
    security_score: float = 100.0
    
    def __post_init__(self):
        if self.active_attackers is None:
            self.active_attackers = []
        if self.attack_types is None:
            self.attack_types = {}

class RealTimeMonitor:
    """Real-time monitoring system for the FL system"""
    
    def __init__(self, update_interval: float = 2.0):
        self.update_interval = update_interval
        self.running = False
        
        # Status objects
        self.blockchain_status = BlockchainStatus()
        self.vehicle_statuses: Dict[int, VehicleStatus] = {}
        self.training_status = TrainingStatus()
        self.attack_status = AttackStatus()
        
        # Historical data (last 100 points)
        self.history = {
            'accuracy': deque(maxlen=100),
            'loss': deque(maxlen=100),
            'tps': deque(maxlen=100),
            'attacks': deque(maxlen=100),
            'timestamps': deque(maxlen=100)
        }

        # Training progress log
        self.training_log = deque(maxlen=20)  # Keep last 20 log entries
        
        # Monitoring thread
        self.monitor_thread = None
        
        logger.info("🔍 Real-time monitor initialized")
    
    def start_monitoring(self):
        """Start the monitoring system"""
        if self.running:
            logger.warning("Monitor already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🚀 Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("🛑 Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Update all status layers
                self._update_blockchain_status()
                self._update_training_status()
                self._update_attack_status()
                
                # Update history
                self._update_history()
                
                # Display status
                self._display_status()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def _update_blockchain_status(self):
        """Update blockchain layer status"""
        try:
            # Check Docker containers
            result = subprocess.run(['docker', 'ps', '--format', 'table {{.Names}}\t{{.Status}}'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                orderer_count = sum(1 for line in lines if 'orderer' in line.lower())
                peer_count = sum(1 for line in lines if 'peer' in line.lower())
                
                self.blockchain_status.network_active = len(lines) > 0
                self.blockchain_status.orderer_nodes = orderer_count
                self.blockchain_status.peer_nodes = peer_count
                
                # Simulate other metrics
                if self.blockchain_status.network_active:
                    self.blockchain_status.latest_block += np.random.randint(0, 3)
                    self.blockchain_status.transactions_per_second = np.random.uniform(10, 50)
                    self.blockchain_status.network_latency = np.random.uniform(50, 200)
                
        except Exception as e:
            logger.debug(f"Blockchain status update failed: {e}")
            self.blockchain_status.network_active = False
    
    def _update_training_status(self):
        """Update training layer status"""
        # This will be updated by the FL system
        pass
    
    def _update_attack_status(self):
        """Update attack monitoring status"""
        # Count active Byzantine clients
        byzantine_clients = [cid for cid, status in self.vehicle_statuses.items() if status.is_byzantine]
        self.attack_status.active_attackers = byzantine_clients
        
        # Update attack counts
        attack_counts = defaultdict(int)
        for status in self.vehicle_statuses.values():
            if status.is_byzantine and status.attack_type != "none":
                attack_counts[status.attack_type] += 1
        
        self.attack_status.attack_types = dict(attack_counts)
        self.attack_status.total_attacks = sum(attack_counts.values())
        
        # Calculate security score
        total_clients = len(self.vehicle_statuses)
        if total_clients > 0:
            byzantine_ratio = len(byzantine_clients) / total_clients
            self.attack_status.security_score = max(0, 100 - (byzantine_ratio * 100))
    
    def _update_history(self):
        """Update historical data"""
        now = datetime.now()
        self.history['timestamps'].append(now)
        self.history['accuracy'].append(self.training_status.global_accuracy)
        self.history['loss'].append(self.training_status.global_loss)
        self.history['tps'].append(self.blockchain_status.transactions_per_second)
        self.history['attacks'].append(self.attack_status.total_attacks)
    
    def _display_status(self):
        """Display current status in terminal"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 80)
        print("🔍 REAL-TIME BLOCKCHAIN FL MONITORING SYSTEM")
        print("=" * 80)
        print(f"⏰ Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Blockchain Layer
        print("🔗 BLOCKCHAIN LAYER")
        print("-" * 40)
        status_icon = "🟢" if self.blockchain_status.network_active else "🔴"
        print(f"{status_icon} Network Status: {'ACTIVE' if self.blockchain_status.network_active else 'INACTIVE'}")
        print(f"📊 Orderer Nodes: {self.blockchain_status.orderer_nodes}")
        print(f"🔗 Peer Nodes: {self.blockchain_status.peer_nodes}")
        print(f"📦 Latest Block: #{self.blockchain_status.latest_block}")
        print(f"⚡ TPS: {self.blockchain_status.transactions_per_second:.1f}")
        print(f"🌐 Latency: {self.blockchain_status.network_latency:.1f}ms")
        print()
        
        # Vehicle/Client Layer
        print("🚗 VEHICLE/CLIENT LAYER")
        print("-" * 40)
        total_clients = len(self.vehicle_statuses)
        active_clients = sum(1 for v in self.vehicle_statuses.values() if v.status != 'idle')
        byzantine_clients = sum(1 for v in self.vehicle_statuses.values() if v.is_byzantine)
        
        print(f"📱 Total Clients: {total_clients}")
        print(f"🔄 Active Clients: {active_clients}")
        print(f"⚠️  Byzantine Clients: {byzantine_clients}")
        
        if self.vehicle_statuses:
            avg_accuracy = np.mean([v.model_accuracy for v in self.vehicle_statuses.values()])
            print(f"📈 Avg Client Accuracy: {avg_accuracy:.2%}")
        print()
        
        # Training Layer
        print("🧠 MODEL TRAINING LAYER")
        print("-" * 40)
        print(f"🔄 Round: {self.training_status.current_round}/{self.training_status.total_rounds}")
        print(f"🎯 Global Accuracy: {self.training_status.global_accuracy:.2%}")
        print(f"📉 Global Loss: {self.training_status.global_loss:.4f}")
        print(f"👥 Participating: {self.training_status.participating_clients}")
        print(f"🔧 Aggregation: {self.training_status.aggregation_method.upper()}")
        print(f"⏱️  Training Time: {self.training_status.training_time:.1f}s")
        print()
        
        # Attack Monitoring
        print("🛡️  ATTACK MONITORING")
        print("-" * 40)
        security_icon = "🟢" if self.attack_status.security_score > 80 else "🟡" if self.attack_status.security_score > 50 else "🔴"
        print(f"{security_icon} Security Score: {self.attack_status.security_score:.1f}/100")
        print(f"⚠️  Total Attacks: {self.attack_status.total_attacks}")
        # Handle both list and int types for active_attackers
        if isinstance(self.attack_status.active_attackers, list):
            active_count = len(self.attack_status.active_attackers)
        else:
            active_count = self.attack_status.active_attackers if self.attack_status.active_attackers else 0
        print(f"🎯 Active Attackers: {active_count}")
        
        if self.attack_status.attack_types:
            print("📊 Attack Types:")
            for attack_type, count in self.attack_status.attack_types.items():
                print(f"   • {attack_type}: {count}")
        print()
        
        # Recent Activity
        if len(self.history['timestamps']) > 1:
            print("📈 RECENT ACTIVITY (Last 10 updates)")
            print("-" * 40)
            recent_acc = list(self.history['accuracy'])[-10:]
            recent_loss = list(self.history['loss'])[-10:]
            recent_tps = list(self.history['tps'])[-10:]
            
            if recent_acc:
                print(f"Accuracy Trend: {' → '.join([f'{acc:.1%}' for acc in recent_acc[-3:]])}")
            if recent_loss:
                print(f"Loss Trend: {' → '.join([f'{loss:.3f}' for loss in recent_loss[-3:]])}")
            if recent_tps:
                print(f"TPS Trend: {' → '.join([f'{tps:.0f}' for tps in recent_tps[-3:]])}")
        

        # Show training progress log
        if hasattr(self, 'training_log') and self.training_log:
            print("\n🔄 TRAINING PROGRESS LOG")
            print("-" * 40)
            # Convert deque to list for slicing
            log_list = list(self.training_log)
            for log_entry in log_list[-5:]:  # Show last 5 entries
                print(f"⏰ {log_entry}")

        print("=" * 80)
    
    # Update methods for external systems
    def update_vehicle_status(self, client_id: int, **kwargs):
        """Update vehicle status"""
        if client_id not in self.vehicle_statuses:
            self.vehicle_statuses[client_id] = VehicleStatus(client_id=client_id)

        for key, value in kwargs.items():
            if hasattr(self.vehicle_statuses[client_id], key):
                setattr(self.vehicle_statuses[client_id], key, value)

        self.vehicle_statuses[client_id].last_update = datetime.now()
    
    def update_training_status(self, **kwargs):
        """Update training status"""
        for key, value in kwargs.items():
            if hasattr(self.training_status, key):
                setattr(self.training_status, key, value)

        # Force update last_update time
        if hasattr(self.training_status, 'last_update'):
            self.training_status.last_update = datetime.now()

    def log_training_progress(self, message: str):
        """Log training progress message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"{timestamp} - {message}"
        self.training_log.append(log_entry)
        print(f"📝 {log_entry}")  # Also print to console immediately

    def log_round_progress(self, round_num: int, total_rounds: int, step: str, details: str = ""):
        """Log detailed round progress"""
        progress_msg = f"Round {round_num}/{total_rounds} - {step}"
        if details:
            progress_msg += f": {details}"
        self.log_training_progress(progress_msg)
    
    def update_attack_status(self, **kwargs):
        """Update attack status"""
        for key, value in kwargs.items():
            if hasattr(self.attack_status, key):
                setattr(self.attack_status, key, value)
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get complete status summary"""
        return {
            'blockchain': asdict(self.blockchain_status),
            'vehicles': {cid: asdict(status) for cid, status in self.vehicle_statuses.items()},
            'training': asdict(self.training_status),
            'attacks': asdict(self.attack_status),
            'timestamp': datetime.now().isoformat()
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        metrics = {
            'status': self.get_status_summary(),
            'history': {
                'accuracy': list(self.history['accuracy']),
                'loss': list(self.history['loss']),
                'tps': list(self.history['tps']),
                'attacks': list(self.history['attacks']),
                'timestamps': [ts.isoformat() for ts in self.history['timestamps']]
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"📊 Metrics exported to {filepath}")

# Global monitor instance
_monitor_instance = None

def get_monitor() -> RealTimeMonitor:
    """Get the global monitor instance (singleton)"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = RealTimeMonitor()
    return _monitor_instance

def start_monitoring():
    """Start the global monitor"""
    monitor = get_monitor()
    monitor.start_monitoring()

def stop_monitoring():
    """Stop the global monitor"""
    monitor = get_monitor()
    monitor.stop_monitoring()

if __name__ == "__main__":
    # Demo mode
    print("🔍 Starting Real-time Monitor Demo...")
    
    # Initialize with some demo data
    monitor.update_training_status(total_rounds=10, model_parameters=55000000)
    
    for i in range(10):
        monitor.update_vehicle_status(
            client_id=i,
            status="idle",
            data_samples=np.random.randint(50, 200),
            model_accuracy=np.random.uniform(0.6, 0.9),
            is_byzantine=(i in [2, 7]),  # Clients 2 and 7 are Byzantine
            attack_type="gaussian_noise" if i in [2, 7] else "none"
        )
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Simulate training progress
        for round_num in range(1, 11):
            time.sleep(5)
            
            # Update training progress
            accuracy = 0.5 + (round_num * 0.04) + np.random.uniform(-0.02, 0.02)
            loss = 2.0 - (round_num * 0.15) + np.random.uniform(-0.1, 0.1)
            
            monitor.update_training_status(
                current_round=round_num,
                global_accuracy=accuracy,
                global_loss=loss,
                participating_clients=np.random.randint(8, 11),
                training_time=np.random.uniform(30, 60)
            )
            
            # Update some client statuses
            for i in range(10):
                if np.random.random() < 0.3:  # 30% chance to update
                    monitor.update_vehicle_status(
                        client_id=i,
                        status=np.random.choice(["training", "uploading", "idle"]),
                        model_accuracy=np.random.uniform(0.6, 0.9),
                        training_loss=np.random.uniform(0.1, 1.0)
                    )
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping monitor...")
        monitor.stop_monitoring()
        print("✅ Monitor stopped")
