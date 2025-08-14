#!/usr/bin/env python3
"""
Hyperledger Fabric Client for IoV Blockchain FL System
Provides Python interface to interact with Fabric network and smart contracts
"""

import subprocess
import json
import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class FabricConfig:
    """Configuration for Fabric network connection"""
    network_path: str = "./blockchain/fabric-samples/test-network"
    channel_name: str = "mychannel"
    chaincode_name: str = "fl_aggregator"
    org_name: str = "Org1MSP"
    peer_address: str = "peer0.org1.example.com:7051"
    orderer_address: str = "orderer.example.com:7050"
    
class FabricClient:
    """Client for interacting with Hyperledger Fabric network"""
    
    def __init__(self, config: FabricConfig):
        self.config = config
        self.network_running = False
        self.chaincode_deployed = False
        self._lock = threading.Lock()
        
    def start_network(self) -> bool:
        """Start the Fabric test network"""
        with self._lock:
            if self.network_running:
                logger.info("Network already running")
                return True
                
            try:
                logger.info("Starting Hyperledger Fabric network...")
                
                # Check if network path exists
                network_path = Path(self.config.network_path)
                if not network_path.exists():
                    logger.error(f"Network path does not exist: {network_path}")
                    return False
                
                # Change to network directory
                original_dir = os.getcwd()
                os.chdir(self.config.network_path)
                
                try:
                    # Start the network
                    result = subprocess.run(
                        ["./network.sh", "up", "createChannel", "-c", self.config.channel_name],
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    
                    if result.returncode == 0:
                        logger.info("Fabric network started successfully")
                        self.network_running = True
                        return True
                    else:
                        logger.error(f"Failed to start network: {result.stderr}")
                        return False
                        
                finally:
                    os.chdir(original_dir)
                    
            except Exception as e:
                logger.error(f"Error starting network: {e}")
                return False
    
    def deploy_chaincode(self) -> bool:
        """Deploy the FL aggregator chaincode"""
        with self._lock:
            if self.chaincode_deployed:
                logger.info("Chaincode already deployed")
                return True
                
            try:
                logger.info("Deploying FL aggregator chaincode...")
                
                original_dir = os.getcwd()
                os.chdir(self.config.network_path)
                
                try:
                    # Package chaincode
                    package_cmd = [
                        "peer", "lifecycle", "chaincode", "package",
                        f"{self.config.chaincode_name}.tar.gz",
                        "--path", "../../smart_contracts/paper_aggregator",
                        "--lang", "golang",
                        "--label", f"{self.config.chaincode_name}_1.0"
                    ]
                    
                    result = subprocess.run(package_cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        logger.error(f"Failed to package chaincode: {result.stderr}")
                        return False
                    
                    # Install chaincode
                    install_cmd = [
                        "peer", "lifecycle", "chaincode", "install",
                        f"{self.config.chaincode_name}.tar.gz"
                    ]
                    
                    result = subprocess.run(install_cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        logger.error(f"Failed to install chaincode: {result.stderr}")
                        return False
                    
                    logger.info("Chaincode deployed successfully")
                    self.chaincode_deployed = True
                    return True
                    
                finally:
                    os.chdir(original_dir)
                    
            except Exception as e:
                logger.error(f"Error deploying chaincode: {e}")
                return False
    
    def invoke_chaincode(self, function: str, args: List[str]) -> Optional[Dict]:
        """Invoke chaincode function"""
        if not self.network_running:
            logger.error("Network not running")
            return {"success": False, "error": "Network not running"}
            
        try:
            original_dir = os.getcwd()
            os.chdir(self.config.network_path)
            
            try:
                invoke_cmd = [
                    "peer", "chaincode", "invoke",
                    "-o", self.config.orderer_address,
                    "--tls",
                    "--cafile", "organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem",
                    "-C", self.config.channel_name,
                    "-n", self.config.chaincode_name,
                    "--peerAddresses", self.config.peer_address,
                    "--tlsRootCertFiles", "organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt",
                    "-c", json.dumps({"function": function, "Args": args})
                ]
                
                result = subprocess.run(invoke_cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    return {"success": True, "output": result.stdout}
                else:
                    logger.error(f"Chaincode invocation failed: {result.stderr}")
                    return {"success": False, "error": result.stderr}
                    
            finally:
                os.chdir(original_dir)
                
        except Exception as e:
            logger.error(f"Error invoking chaincode: {e}")
            return {"success": False, "error": str(e)}
    
    def query_chaincode(self, function: str, args: List[str]) -> Optional[Dict]:
        """Query chaincode function"""
        if not self.network_running:
            logger.error("Network not running")
            return {"success": False, "error": "Network not running"}
            
        try:
            original_dir = os.getcwd()
            os.chdir(self.config.network_path)
            
            try:
                query_cmd = [
                    "peer", "chaincode", "query",
                    "-C", self.config.channel_name,
                    "-n", self.config.chaincode_name,
                    "-c", json.dumps({"function": function, "Args": args})
                ]
                
                result = subprocess.run(query_cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    try:
                        return json.loads(result.stdout)
                    except json.JSONDecodeError:
                        return {"success": True, "output": result.stdout}
                else:
                    logger.error(f"Chaincode query failed: {result.stderr}")
                    return {"success": False, "error": result.stderr}
                    
            finally:
                os.chdir(original_dir)
                
        except Exception as e:
            logger.error(f"Error querying chaincode: {e}")
            return {"success": False, "error": str(e)}
    
    def stop_network(self) -> bool:
        """Stop the Fabric network"""
        with self._lock:
            if not self.network_running:
                logger.info("Network already stopped")
                return True
                
            try:
                logger.info("Stopping Hyperledger Fabric network...")
                
                original_dir = os.getcwd()
                os.chdir(self.config.network_path)
                
                try:
                    result = subprocess.run(
                        ["./network.sh", "down"],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    if result.returncode == 0:
                        logger.info("Fabric network stopped successfully")
                        self.network_running = False
                        self.chaincode_deployed = False
                        return True
                    else:
                        logger.error(f"Failed to stop network: {result.stderr}")
                        return False
                        
                finally:
                    os.chdir(original_dir)
                    
            except Exception as e:
                logger.error(f"Error stopping network: {e}")
                return False

# Singleton instance for global access
_fabric_client = None

def get_fabric_client() -> FabricClient:
    """Get singleton Fabric client instance"""
    global _fabric_client
    if _fabric_client is None:
        config = FabricConfig()
        _fabric_client = FabricClient(config)
    return _fabric_client
