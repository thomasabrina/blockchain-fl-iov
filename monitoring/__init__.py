"""
Monitoring module for IoV Federated Learning system.
Provides real-time monitoring capabilities for system resources and training metrics.
"""

from .real_time_monitor import start_monitoring, stop_monitoring, get_monitor

__all__ = [
    'start_monitoring',
    'stop_monitoring',
    'get_monitor'
]
