"""
ElevatorGuard-AI Multi-Agent Package
Multi-modal predictive maintenance pipeline with autonomous edge agents:
- MessageBus & AgentMessage
- BaseAgent & AgentStatus
- CameraVisionAgent (Stream 1: Hoistway NDT Camera)
- VibrationSensorAgent (Stream 2: Telemetry & Dynamic Vibration)
- FusionSupervisorAgent (Stream 3: Multi-Modal Fusion & RUL Engine)
- SafetyInterlockAgent (Safety Enforcement Watchdog)
- MultiAgentPipelineCoordinator (Unified Multi-Stream Ingestion Gateway)
"""

from src.agents.message_bus import MessageBus, AgentMessage
from src.agents.base_agent import BaseAgent, AgentStatus
from src.agents.camera_agent import CameraVisionAgent
from src.agents.vibration_agent import VibrationSensorAgent
from src.agents.fusion_agent import FusionSupervisorAgent
from src.agents.safety_agent import SafetyInterlockAgent
from src.agents.coordinator import MultiAgentPipelineCoordinator

__all__ = [
    "MessageBus",
    "AgentMessage",
    "BaseAgent",
    "AgentStatus",
    "CameraVisionAgent",
    "VibrationSensorAgent",
    "FusionSupervisorAgent",
    "SafetyInterlockAgent",
    "MultiAgentPipelineCoordinator"
]
