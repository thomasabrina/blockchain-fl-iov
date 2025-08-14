"""
SUMO Traffic Simulation Integration for Federated Learning
Implements realistic vehicle mobility patterns and RSU simulation
"""

import os
import sys
import time
import json
import random
import logging
import threading
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import xml.etree.ElementTree as ET

# SUMO imports (requires SUMO installation)
try:
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.path.append('/usr/share/sumo/tools')
    
    import traci
    import sumolib
    SUMO_AVAILABLE = True
except ImportError:
    SUMO_AVAILABLE = False
    logging.warning("SUMO not available. Using simulation mode.")

logger = logging.getLogger(__name__)

class VehicleType(Enum):
    """Types of vehicles in simulation"""
    PASSENGER = "passenger"
    TRUCK = "truck"
    BUS = "bus"
    EMERGENCY = "emergency"
    AUTONOMOUS = "autonomous"

class RSUType(Enum):
    """Types of Road Side Units"""
    BASIC = "basic"
    ADVANCED = "advanced"
    EDGE_COMPUTING = "edge_computing"
    AGGREGATOR = "aggregator"

@dataclass
class VehicleInfo:
    """Vehicle information in simulation"""
    vehicle_id: str
    vehicle_type: VehicleType
    position: Tuple[float, float]
    speed: float
    route_id: str
    lane_id: str
    is_fl_participant: bool
    data_generation_rate: float
    communication_range: float
    last_update_time: float
    connected_rsus: List[str]
    # Paper-specific V2X parameters
    v2x_transmission_rate: float = 25.0  # Mbps (5-50 range as per paper)
    disconnection_probability: float = 0.03  # 3% (1-5% range as per paper)
    connection_quality: float = 1.0  # 0-1 scale (1.0 = perfect)
    last_disconnection_time: float = 0.0
    intermittent_connectivity: bool = False

@dataclass
class RSUInfo:
    """Road Side Unit information"""
    rsu_id: str
    rsu_type: RSUType
    position: Tuple[float, float]
    coverage_radius: float
    processing_capacity: float
    storage_capacity: float
    connected_vehicles: List[str]
    is_aggregator: bool
    blockchain_node: bool

@dataclass
class NetworkTopology:
    """Dynamic network topology"""
    timestamp: float
    vehicles: Dict[str, VehicleInfo]
    rsus: Dict[str, RSUInfo]
    vehicle_rsu_connections: Dict[str, List[str]]
    rsu_rsu_connections: Dict[str, List[str]]
    network_partitions: List[List[str]]

class SUMOTrafficSimulator:
    """SUMO-based traffic simulation for FL"""
    
    def __init__(self, config_file: str = "simulation/traffic.sumocfg"):
        self.config_file = config_file
        self.simulation_running = False
        self.current_step = 0
        self.vehicles = {}
        self.rsus = {}
        self.fl_participants = set()
        self.network_topology_history = []
        
        # Simulation parameters (matching paper specifications)
        self.simulation_duration = 3600  # 1 hour
        self.step_length = 1.0  # 1 second steps
        self.fl_participation_rate = 0.3  # 30% of vehicles participate in FL

        # Paper-specific parameters
        self.rsu_coverage_radius_min = 100.0  # 100 meters (paper: 100-300m)
        self.rsu_coverage_radius_max = 300.0  # 300 meters
        self.v2x_transmission_rate_min = 5.0   # 5 Mbps (paper: 5-50 Mbps)
        self.v2x_transmission_rate_max = 50.0  # 50 Mbps
        self.disconnection_prob_min = 0.01     # 1% (paper: 1-5%)
        self.disconnection_prob_max = 0.05     # 5%
        self.vehicle_comm_range = 100.0        # 100 meters
        
        # Initialize simulation
        self._initialize_simulation()

        # Create OpenStreetMap-style network topology
        self._create_osm_style_network()
        
    def _initialize_simulation(self):
        """Initialize SUMO simulation"""
        if not SUMO_AVAILABLE:
            logger.warning("SUMO not available, using mock simulation")
            self._create_mock_simulation()
            return
        
        try:
            # Test SUMO availability first
            import subprocess
            result = subprocess.run(['sumo', '--version'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                logger.warning("SUMO command failed, using mock simulation")
                self._create_mock_simulation()
                return

            # Create simulation files if they don't exist
            self._create_simulation_files()

            # Start SUMO
            sumo_cmd = [
                "sumo-gui" if os.environ.get('SUMO_GUI', 'false').lower() == 'true' else "sumo",
                "-c", self.config_file,
                "--step-length", str(self.step_length),
                "--no-warnings", "true"
            ]

            traci.start(sumo_cmd)
            self.simulation_running = True

            # Initialize RSUs
            self._initialize_rsus()

            logger.info("SUMO simulation initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize SUMO: {e}, using mock simulation")
            self._create_mock_simulation()
    
    def _create_simulation_files(self):
        """Create SUMO simulation configuration files"""
        os.makedirs("simulation", exist_ok=True)
        
        # Create network file (simple grid)
        self._create_network_file()
        
        # Create route file
        self._create_route_file()
        
        # Create configuration file
        self._create_config_file()
    
    def _create_network_file(self):
        """Create SUMO network file"""
        network_xml = """<?xml version="1.0" encoding="UTF-8"?>
<net version="1.16" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">

    <location netOffset="0.00,0.00" convBoundary="0.00,0.00,2000.00,2000.00" origBoundary="-10000000000.00,-10000000000.00,10000000000.00,10000000000.00" projParameter="!"/>

    <edge id="E0" from="J0" to="J1" priority="1">
        <lane id="E0_0" index="0" speed="13.89" length="2000.00" shape="0.00,-1.60 2000.00,-1.60"/>
        <lane id="E0_1" index="1" speed="13.89" length="2000.00" shape="0.00,1.60 2000.00,1.60"/>
    </edge>
    
    <edge id="E1" from="J1" to="J2" priority="1">
        <lane id="E1_0" index="0" speed="13.89" length="2000.00" shape="2001.60,0.00 2001.60,2000.00"/>
        <lane id="E1_1" index="1" speed="13.89" length="2000.00" shape="1998.40,0.00 1998.40,2000.00"/>
    </edge>
    
    <edge id="E2" from="J2" to="J3" priority="1">
        <lane id="E2_0" index="0" speed="13.89" length="2000.00" shape="2000.00,2001.60 0.00,2001.60"/>
        <lane id="E2_1" index="1" speed="13.89" length="2000.00" shape="2000.00,1998.40 0.00,1998.40"/>
    </edge>
    
    <edge id="E3" from="J3" to="J0" priority="1">
        <lane id="E3_0" index="0" speed="13.89" length="2000.00" shape="-1.60,2000.00 -1.60,0.00"/>
        <lane id="E3_1" index="1" speed="13.89" length="2000.00" shape="1.60,2000.00 1.60,0.00"/>
    </edge>

    <junction id="J0" type="priority" x="0.00" y="0.00" incLanes="E3_0 E3_1" intLanes=":J0_0_0 :J0_1_0" shape="-3.20,0.00 3.20,0.00 0.00,-3.20 0.00,3.20">
        <request index="0" response="00" foes="00" cont="0"/>
        <request index="1" response="00" foes="00" cont="0"/>
    </junction>
    
    <junction id="J1" type="priority" x="2000.00" y="0.00" incLanes="E0_0 E0_1" intLanes=":J1_0_0 :J1_1_0" shape="2000.00,3.20 2000.00,-3.20 2003.20,0.00 1996.80,0.00">
        <request index="0" response="00" foes="00" cont="0"/>
        <request index="1" response="00" foes="00" cont="0"/>
    </junction>
    
    <junction id="J2" type="priority" x="2000.00" y="2000.00" incLanes="E1_0 E1_1" intLanes=":J2_0_0 :J2_1_0" shape="2003.20,2000.00 1996.80,2000.00 2000.00,1996.80 2000.00,2003.20">
        <request index="0" response="00" foes="00" cont="0"/>
        <request index="1" response="00" foes="00" cont="0"/>
    </junction>
    
    <junction id="J3" type="priority" x="0.00" y="2000.00" incLanes="E2_0 E2_1" intLanes=":J3_0_0 :J3_1_0" shape="0.00,2003.20 0.00,1996.80 -3.20,2000.00 3.20,2000.00">
        <request index="0" response="00" foes="00" cont="0"/>
        <request index="1" response="00" foes="00" cont="0"/>
    </junction>

    <connection from="E0" to="E1" fromLane="0" toLane="0" via=":J1_0_0" dir="r" state="M"/>
    <connection from="E0" to="E1" fromLane="1" toLane="1" via=":J1_1_0" dir="r" state="M"/>
    <connection from="E1" to="E2" fromLane="0" toLane="0" via=":J2_0_0" dir="r" state="M"/>
    <connection from="E1" to="E2" fromLane="1" toLane="1" via=":J2_1_0" dir="r" state="M"/>
    <connection from="E2" to="E3" fromLane="0" toLane="0" via=":J3_0_0" dir="r" state="M"/>
    <connection from="E2" to="E3" fromLane="1" toLane="1" via=":J3_1_0" dir="r" state="M"/>
    <connection from="E3" to="E0" fromLane="0" toLane="0" via=":J0_0_0" dir="r" state="M"/>
    <connection from="E3" to="E0" fromLane="1" toLane="1" via=":J0_1_0" dir="r" state="M"/>

</net>"""
        
        with open("simulation/traffic.net.xml", "w") as f:
            f.write(network_xml)
    
    def _create_route_file(self):
        """Create SUMO route file with vehicle definitions"""
        route_xml = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    
    <!-- Vehicle types -->
    <vType id="passenger" accel="2.6" decel="4.5" sigma="0.5" length="4.5" maxSpeed="55.56"/>
    <vType id="truck" accel="1.3" decel="4.0" sigma="0.5" length="12.0" maxSpeed="27.78"/>
    <vType id="bus" accel="1.2" decel="4.0" sigma="0.5" length="15.0" maxSpeed="25.0"/>
    <vType id="autonomous" accel="3.0" decel="5.0" sigma="0.1" length="4.5" maxSpeed="55.56"/>
    
    <!-- Routes -->
    <route id="route0" edges="E0 E1 E2 E3"/>
    <route id="route1" edges="E1 E2 E3 E0"/>
    <route id="route2" edges="E2 E3 E0 E1"/>
    <route id="route3" edges="E3 E0 E1 E2"/>
    
    <!-- Vehicle flows -->
    <flow id="passenger_flow" type="passenger" route="route0" begin="0" end="3600" vehsPerHour="300"/>
    <flow id="truck_flow" type="truck" route="route1" begin="0" end="3600" vehsPerHour="50"/>
    <flow id="bus_flow" type="bus" route="route2" begin="0" end="3600" vehsPerHour="20"/>
    <flow id="autonomous_flow" type="autonomous" route="route3" begin="0" end="3600" vehsPerHour="100"/>
    
</routes>"""
        
        with open("simulation/traffic.rou.xml", "w") as f:
            f.write(route_xml)
    
    def _create_config_file(self):
        """Create SUMO configuration file"""
        config_xml = """<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    
    <input>
        <net-file value="traffic.net.xml"/>
        <route-files value="traffic.rou.xml"/>
    </input>
    
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1"/>
    </time>
    
    <processing>
        <collision.action value="warn"/>
        <collision.check-junctions value="true"/>
    </processing>
    
    <report>
        <verbose value="false"/>
        <no-warnings value="true"/>
    </report>
    
</configuration>"""
        
        with open("simulation/traffic.sumocfg", "w") as f:
            f.write(config_xml)
    
    def _initialize_rsus(self):
        """Initialize Road Side Units at strategic locations"""
        rsu_positions = [
            (500, 500, RSUType.AGGREGATOR),
            (1500, 500, RSUType.ADVANCED),
            (1500, 1500, RSUType.AGGREGATOR),
            (500, 1500, RSUType.ADVANCED),
            (1000, 0, RSUType.BASIC),
            (2000, 1000, RSUType.BASIC),
            (1000, 2000, RSUType.BASIC),
            (0, 1000, RSUType.BASIC),
            (1000, 1000, RSUType.EDGE_COMPUTING)
        ]
        
        for i, (x, y, rsu_type) in enumerate(rsu_positions):
            rsu_id = f"RSU_{i:02d}"
            
            rsu_info = RSUInfo(
                rsu_id=rsu_id,
                rsu_type=rsu_type,
                position=(x, y),
                coverage_radius=random.uniform(self.rsu_coverage_radius_min, self.rsu_coverage_radius_max),
                processing_capacity=1000.0 if rsu_type == RSUType.EDGE_COMPUTING else 500.0,
                storage_capacity=10000.0,
                connected_vehicles=[],
                is_aggregator=rsu_type in [RSUType.AGGREGATOR, RSUType.EDGE_COMPUTING],
                blockchain_node=rsu_type == RSUType.AGGREGATOR
            )
            
            self.rsus[rsu_id] = rsu_info
        
        logger.info(f"Initialized {len(self.rsus)} RSUs")
    
    def _create_mock_simulation(self):
        """Create mock simulation when SUMO is not available"""
        logger.info("Creating mock traffic simulation")
        
        # Create mock vehicles
        for i in range(50):
            vehicle_id = f"vehicle_{i:03d}"
            vehicle_type = random.choice(list(VehicleType))
            
            vehicle_info = VehicleInfo(
                vehicle_id=vehicle_id,
                vehicle_type=vehicle_type,
                position=(random.uniform(0, 2000), random.uniform(0, 2000)),
                speed=random.uniform(5, 15),
                route_id=f"route{i%4}",
                lane_id=f"E{i%4}_0",
                is_fl_participant=random.random() < self.fl_participation_rate,
                data_generation_rate=random.uniform(0.1, 1.0),
                communication_range=self.vehicle_comm_range,
                last_update_time=0,
                connected_rsus=[],
                # Paper-specific V2X parameters
                v2x_transmission_rate=random.uniform(self.v2x_transmission_rate_min, self.v2x_transmission_rate_max),
                disconnection_probability=random.uniform(self.disconnection_prob_min, self.disconnection_prob_max),
                connection_quality=random.uniform(0.8, 1.0),
                last_disconnection_time=0.0,
                intermittent_connectivity=False
            )
            
            self.vehicles[vehicle_id] = vehicle_info
            
            if vehicle_info.is_fl_participant:
                self.fl_participants.add(vehicle_id)
        
        # Initialize RSUs for mock simulation
        self._initialize_rsus()
        self.simulation_running = True

    def _create_osm_style_network(self):
        """Create OpenStreetMap-style network topology (as mentioned in paper)"""
        # This simulates the paper's approach of using OSM-derived topologies
        # In a real implementation, this would load actual OSM data

        # Create realistic road network structure
        self.road_network = {
            'highways': [
                {'id': 'highway_1', 'type': 'primary', 'length': 5000, 'lanes': 4},
                {'id': 'highway_2', 'type': 'secondary', 'length': 3000, 'lanes': 2},
            ],
            'intersections': [
                {'id': 'intersection_1', 'position': (1000, 1000), 'traffic_light': True},
                {'id': 'intersection_2', 'position': (2000, 1500), 'traffic_light': False},
            ],
            'urban_areas': [
                {'id': 'downtown', 'center': (1500, 1500), 'radius': 800},
                {'id': 'residential', 'center': (500, 500), 'radius': 600},
            ]
        }

        logger.info("OpenStreetMap-style network topology created")
        logger.info(f"Network includes {len(self.road_network['highways'])} highways, "
                   f"{len(self.road_network['intersections'])} intersections, "
                   f"{len(self.road_network['urban_areas'])} urban areas")
    
    def step_simulation(self) -> NetworkTopology:
        """Advance simulation by one step and return network topology"""
        if not self.simulation_running:
            return None
        
        if SUMO_AVAILABLE and traci.isLoaded():
            # SUMO simulation step
            traci.simulationStep()
            self.current_step = traci.simulation.getTime()
            
            # Update vehicle information
            self._update_vehicles_sumo()
        else:
            # Mock simulation step
            self.current_step += self.step_length
            self._update_vehicles_mock()
        
        # Update network topology
        topology = self._calculate_network_topology()
        self.network_topology_history.append(topology)
        
        # Keep only recent history
        if len(self.network_topology_history) > 100:
            self.network_topology_history.pop(0)
        
        return topology
    
    def _update_vehicles_sumo(self):
        """Update vehicle information from SUMO"""
        current_vehicles = set(traci.vehicle.getIDList())
        
        # Remove vehicles that left the simulation
        for vehicle_id in list(self.vehicles.keys()):
            if vehicle_id not in current_vehicles:
                del self.vehicles[vehicle_id]
                self.fl_participants.discard(vehicle_id)
        
        # Update existing and add new vehicles
        for vehicle_id in current_vehicles:
            if vehicle_id not in self.vehicles:
                # New vehicle
                vehicle_type_str = traci.vehicle.getTypeID(vehicle_id)
                vehicle_type = VehicleType(vehicle_type_str) if vehicle_type_str in [vt.value for vt in VehicleType] else VehicleType.PASSENGER
                
                is_participant = random.random() < self.fl_participation_rate
                
                vehicle_info = VehicleInfo(
                    vehicle_id=vehicle_id,
                    vehicle_type=vehicle_type,
                    position=traci.vehicle.getPosition(vehicle_id),
                    speed=traci.vehicle.getSpeed(vehicle_id),
                    route_id=traci.vehicle.getRouteID(vehicle_id),
                    lane_id=traci.vehicle.getLaneID(vehicle_id),
                    is_fl_participant=is_participant,
                    data_generation_rate=random.uniform(0.1, 1.0),
                    communication_range=self.vehicle_comm_range,
                    last_update_time=self.current_step,
                    connected_rsus=[],
                    # Paper-specific V2X parameters
                    v2x_transmission_rate=random.uniform(self.v2x_transmission_rate_min, self.v2x_transmission_rate_max),
                    disconnection_probability=random.uniform(self.disconnection_prob_min, self.disconnection_prob_max),
                    connection_quality=random.uniform(0.8, 1.0),
                    last_disconnection_time=0.0,
                    intermittent_connectivity=False
                )
                
                self.vehicles[vehicle_id] = vehicle_info
                
                if is_participant:
                    self.fl_participants.add(vehicle_id)
            else:
                # Update existing vehicle
                vehicle_info = self.vehicles[vehicle_id]
                vehicle_info.position = traci.vehicle.getPosition(vehicle_id)
                vehicle_info.speed = traci.vehicle.getSpeed(vehicle_id)
                vehicle_info.lane_id = traci.vehicle.getLaneID(vehicle_id)
                vehicle_info.last_update_time = self.current_step
    
    def _update_vehicles_mock(self):
        """Update vehicle information in mock simulation"""
        for vehicle_info in self.vehicles.values():
            # Simple movement simulation
            angle = random.uniform(0, 2 * np.pi)
            distance = vehicle_info.speed * self.step_length
            
            new_x = vehicle_info.position[0] + distance * np.cos(angle)
            new_y = vehicle_info.position[1] + distance * np.sin(angle)
            
            # Keep within bounds
            new_x = max(0, min(2000, new_x))
            new_y = max(0, min(2000, new_y))
            
            vehicle_info.position = (new_x, new_y)
            vehicle_info.speed = max(0, vehicle_info.speed + random.uniform(-1, 1))
            vehicle_info.last_update_time = self.current_step
    
    def _simulate_intermittent_connectivity(self):
        """Simulate intermittent connectivity and changing network quality (as per paper)"""
        for vehicle_id, vehicle_info in self.vehicles.items():
            # Simulate disconnection probability (1-5% as per paper)
            if random.random() < vehicle_info.disconnection_probability:
                # Temporary disconnection
                vehicle_info.intermittent_connectivity = True
                vehicle_info.last_disconnection_time = self.current_step
                vehicle_info.connection_quality = 0.0
            else:
                # Check if recovering from disconnection
                if vehicle_info.intermittent_connectivity:
                    # Recovery after 5-15 seconds
                    if self.current_step - vehicle_info.last_disconnection_time > random.uniform(5, 15):
                        vehicle_info.intermittent_connectivity = False
                        vehicle_info.connection_quality = random.uniform(0.7, 1.0)

                # Simulate varying network quality during training
                if not vehicle_info.intermittent_connectivity:
                    # Network quality varies based on speed and distance
                    quality_factor = max(0.3, 1.0 - (vehicle_info.speed / 100.0))  # Higher speed = lower quality
                    vehicle_info.connection_quality = min(1.0, quality_factor + random.uniform(-0.1, 0.1))

                    # Adjust V2X transmission rate based on quality
                    base_rate = random.uniform(self.v2x_transmission_rate_min, self.v2x_transmission_rate_max)
                    vehicle_info.v2x_transmission_rate = base_rate * vehicle_info.connection_quality

    def _calculate_network_topology(self) -> NetworkTopology:
        """Calculate current network topology with IoV dynamics"""
        # Simulate intermittent connectivity first
        self._simulate_intermittent_connectivity()

        # Update vehicle-RSU connections
        vehicle_rsu_connections = {}
        
        for vehicle_id, vehicle_info in self.vehicles.items():
            connected_rsus = []
            
            for rsu_id, rsu_info in self.rsus.items():
                distance = np.sqrt(
                    (vehicle_info.position[0] - rsu_info.position[0])**2 +
                    (vehicle_info.position[1] - rsu_info.position[1])**2
                )
                
                if distance <= rsu_info.coverage_radius:
                    connected_rsus.append(rsu_id)
            
            vehicle_info.connected_rsus = connected_rsus
            vehicle_rsu_connections[vehicle_id] = connected_rsus
        
        # Update RSU-RSU connections (simplified)
        rsu_rsu_connections = {}
        for rsu_id in self.rsus.keys():
            rsu_rsu_connections[rsu_id] = []  # Assume all RSUs are connected via backbone
        
        # Update RSU connected vehicles
        for rsu_info in self.rsus.values():
            rsu_info.connected_vehicles = [
                vehicle_id for vehicle_id, connected_rsus in vehicle_rsu_connections.items()
                if rsu_info.rsu_id in connected_rsus
            ]
        
        # Detect network partitions (simplified)
        network_partitions = [list(self.vehicles.keys())]  # Single partition for simplicity
        
        return NetworkTopology(
            timestamp=self.current_step,
            vehicles=self.vehicles.copy(),
            rsus=self.rsus.copy(),
            vehicle_rsu_connections=vehicle_rsu_connections,
            rsu_rsu_connections=rsu_rsu_connections,
            network_partitions=network_partitions
        )
    
    def get_fl_participants(self) -> List[str]:
        """Get list of current FL participants"""
        return [vid for vid in self.fl_participants if vid in self.vehicles]
    
    def get_aggregator_rsus(self) -> List[str]:
        """Get list of aggregator RSUs"""
        return [rsu_id for rsu_id, rsu_info in self.rsus.items() if rsu_info.is_aggregator]
    
    def get_simulation_stats(self) -> Dict[str, Any]:
        """Get simulation statistics (including paper-specific metrics)"""
        fl_participants = self.get_fl_participants()

        # Calculate paper-specific metrics
        connected_vehicles = [v for v in self.vehicles.values() if v.connected_rsus and not v.intermittent_connectivity]
        avg_transmission_rate = np.mean([v.v2x_transmission_rate for v in self.vehicles.values()]) if self.vehicles else 0
        avg_disconnection_prob = np.mean([v.disconnection_probability for v in self.vehicles.values()]) if self.vehicles else 0
        avg_connection_quality = np.mean([v.connection_quality for v in self.vehicles.values()]) if self.vehicles else 0
        intermittent_vehicles = len([v for v in self.vehicles.values() if v.intermittent_connectivity])

        # RSU coverage analysis
        rsu_coverage_radii = [r.coverage_radius for r in self.rsus.values()]

        return {
            'current_step': self.current_step,
            'total_vehicles': len(self.vehicles),
            'fl_participants': len(fl_participants),
            'total_rsus': len(self.rsus),
            'aggregator_rsus': len(self.get_aggregator_rsus()),
            'vehicle_types': {vt.value: sum(1 for v in self.vehicles.values() if v.vehicle_type == vt) for vt in VehicleType},
            'rsu_types': {rt.value: sum(1 for r in self.rsus.values() if r.rsu_type == rt) for rt in RSUType},
            'average_speed': np.mean([v.speed for v in self.vehicles.values()]) if self.vehicles else 0,
            'network_connectivity': len(connected_vehicles) / len(self.vehicles) if self.vehicles else 0,
            # Paper-specific metrics
            'avg_v2x_transmission_rate_mbps': avg_transmission_rate,
            'avg_disconnection_probability': avg_disconnection_prob * 100,  # Convert to percentage
            'avg_connection_quality': avg_connection_quality,
            'intermittent_connectivity_vehicles': intermittent_vehicles,
            'rsu_coverage_radius_range': f"{min(rsu_coverage_radii):.0f}-{max(rsu_coverage_radii):.0f}m" if rsu_coverage_radii else "N/A",
            'osm_network_elements': len(getattr(self, 'road_network', {}).get('highways', [])) + len(getattr(self, 'road_network', {}).get('intersections', []))
        }
    
    def stop_simulation(self):
        """Stop the simulation"""
        if SUMO_AVAILABLE and traci.isLoaded():
            traci.close()
        
        self.simulation_running = False
        logger.info("Simulation stopped")
    
    def export_topology_data(self, filename: str):
        """Export network topology data for analysis"""
        export_data = {
            'simulation_duration': self.current_step,
            'vehicles': {vid: {
                'vehicle_type': vinfo.vehicle_type.value,
                'is_fl_participant': vinfo.is_fl_participant,
                'final_position': vinfo.position,
                'data_generation_rate': vinfo.data_generation_rate
            } for vid, vinfo in self.vehicles.items()},
            'rsus': {rid: {
                'rsu_type': rinfo.rsu_type.value,
                'position': rinfo.position,
                'is_aggregator': rinfo.is_aggregator,
                'blockchain_node': rinfo.blockchain_node
            } for rid, rinfo in self.rsus.items()},
            'topology_history': [
                {
                    'timestamp': topo.timestamp,
                    'vehicle_count': len(topo.vehicles),
                    'connections': len([v for conns in topo.vehicle_rsu_connections.values() for v in conns])
                }
                for topo in self.network_topology_history[-10:]  # Last 10 snapshots
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Topology data exported to {filename}")

# Factory function
def create_sumo_simulator(config_file: str = "simulation/traffic.sumocfg") -> SUMOTrafficSimulator:
    """Create SUMO traffic simulator"""
    return SUMOTrafficSimulator(config_file)

# Example usage and testing
if __name__ == "__main__":
    print("Testing SUMO Traffic Simulation...")
    
    # Create simulator
    simulator = create_sumo_simulator()
    
    try:
        # Run simulation for a few steps
        for step in range(10):
            topology = simulator.step_simulation()
            
            if topology:
                stats = simulator.get_simulation_stats()
                print(f"Step {step}: {stats['total_vehicles']} vehicles, "
                      f"{stats['fl_participants']} FL participants, "
                      f"connectivity: {stats['network_connectivity']:.2f}")
            
            time.sleep(0.1)  # Small delay for visualization
        
        # Get final statistics
        final_stats = simulator.get_simulation_stats()
        print(f"\nFinal Statistics:")
        for key, value in final_stats.items():
            print(f"  {key}: {value}")
        
        # Export data
        simulator.export_topology_data("simulation_results.json")
        
    finally:
        simulator.stop_simulation()
    
    print("SUMO simulation test completed!")
