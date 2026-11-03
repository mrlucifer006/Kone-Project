"""
ElevatorGuard-AI: Safety Interlock Agent
Dedicated autonomous safety watchdog enforcing EN 81-20/50, ISO 4309, and ISO 18738 standards:
Monitors real-time multi-modal telemetry streams for critical safety violations,
triggers automatic emergency braking, manages audible/visual interlock alarms, and audits incidents.
"""

import time
import threading
from typing import Optional, Dict, Any, List, Callable
from collections import deque

from src.agents.base_agent import BaseAgent, AgentStatus
from src.agents.message_bus import MessageBus, AgentMessage


class SafetyInterlockAgent(BaseAgent):
    """
    Autonomous safety enforcement agent with emergency brake authority.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        name: str = "SafetyInterlockAgent",
        emergency_stop_callback: Optional[Callable[[], None]] = None,
        reset_callback: Optional[Callable[[], None]] = None,
        incident_log_size: int = 100
    ):
        super().__init__(name=name, message_bus=message_bus, tick_interval=0.02)
        self.emergency_stop_callback = emergency_stop_callback
        self.reset_callback = reset_callback

        # Safety alarms state
        self.emergency_stop_active: bool = False
        self.active_alarms: List[str] = []
        self.incident_history = deque(maxlen=incident_log_size)
        self.total_critical_events = 0

        self._state_lock = threading.RLock()

    def on_start(self):
        """Subscribe to all telemetry channels."""
        self.subscribe("telemetry.vision")
        self.subscribe("telemetry.vibration")
        self.subscribe("telemetry.fusion")
        self.subscribe("system.command")

    def process_message(self, msg: AgentMessage):
        """Evaluate incoming events against safety thresholds."""
        topic = msg.topic
        payload = msg.payload

        if topic == "system.command":
            cmd = payload.get("command")
            if cmd == "EMERGENCY_STOP":
                self.trigger_emergency_stop(reason="MANUAL_COMMAND", details="Triggered via system command.")
            elif cmd == "RESET_ALARMS":
                self.reset_alarms()
            return

        alarms_to_add = []

        if topic == "telemetry.vision":
            if payload.get("is_discard_required", False):
                wb = payload.get("wire_breaks", 0)
                wear = payload.get("wear_pct", 0.0)
                corr = payload.get("corrosion_pct", 0.0)
                alarms_to_add.append(
                    f"ISO 4309 DISCARD BREACH (Breaks: {wb}/6, Wear: {wear*100:.1f}%, Corrosion: {corr*100:.1f}%)"
                )

        elif topic == "telemetry.vibration":
            alarms = payload.get("alarms", {})
            if alarms.get("tension_imbalance", False):
                ratio = payload.get("tension_imbalance_ratio", 0.0)
                alarms_to_add.append(f"EN 81-20 TENSION IMBALANCE ({ratio*100:.1f}% > 15.0% limit)")
            if alarms.get("vibration_exceeded", False):
                rms = payload.get("acc_rms_z", 0.0)
                alarms_to_add.append(f"ISO 18738 EXCESSIVE VIBRATION ({rms:.3f} m/s² > 0.45 limit)")
            if alarms.get("emergency_stop", False):
                alarms_to_add.append("HARDWARE_ESTOP_FLAG_ASSERTED")

        elif topic == "telemetry.fusion":
            if payload.get("safety_override_active", False) or payload.get("hi_composite", 1.0) <= 0.20:
                alarms_to_add.append(f"CRITICAL DISCARD HI_COMP ({payload.get('hi_composite', 0.0):.3f} <= 0.20)")

        if alarms_to_add:
            for alarm_text in alarms_to_add:
                self.trigger_emergency_stop(reason="SAFETY_VIOLATION", details=alarm_text)

    def trigger_emergency_stop(self, reason: str, details: str):
        """Assert emergency brake and log incident."""
        with self._state_lock:
            already_active = self.emergency_stop_active
            self.emergency_stop_active = True
            if details not in self.active_alarms:
                self.active_alarms.append(details)

            incident = {
                "timestamp": round(time.time(), 4),
                "reason": reason,
                "details": details,
                "action": "EMERGENCY_STOP_ENGAGED"
            }
            self.incident_history.append(incident)
            self.total_critical_events += 1

        # Execute physical/simulation callback if set
        if not already_active and self.emergency_stop_callback:
            try:
                self.emergency_stop_callback()
            except Exception as e:
                print(f"[SafetyInterlockAgent] Error executing e-stop callback: {e}")

        # Publish safety alert
        self.publish("safety.alert", {
            "level": "CRITICAL_SHUTDOWN",
            "reason": reason,
            "details": details,
            "emergency_stop_active": True,
            "active_alarms": list(self.active_alarms)
        })

    def reset_alarms(self):
        """Clear active safety alarms and release emergency brake."""
        with self._state_lock:
            self.emergency_stop_active = False
            self.active_alarms.clear()
            self.incident_history.append({
                "timestamp": round(time.time(), 4),
                "reason": "OPERATOR_RESET",
                "details": "All interlock alarms cleared by operator.",
                "action": "SAFETY_RESET"
            })

        if self.reset_callback:
            try:
                self.reset_callback()
            except Exception as e:
                print(f"[SafetyInterlockAgent] Error executing reset callback: {e}")

        self.publish("safety.alert", {
            "level": "NOMINAL",
            "reason": "OPERATOR_RESET",
            "details": "Safety interlock alarms cleared.",
            "emergency_stop_active": False,
            "active_alarms": []
        })

    def get_safety_state(self) -> Dict[str, Any]:
        """Return current safety and alarm states."""
        with self._state_lock:
            return {
                "emergency_stop_active": self.emergency_stop_active,
                "active_alarms": list(self.active_alarms),
                "total_critical_events": self.total_critical_events,
                "incident_history": list(self.incident_history)
            }

    def get_status(self) -> Dict[str, Any]:
        """Extended status including safety interlock metrics."""
        base_status = super().get_status()
        with self._state_lock:
            base_status.update({
                "emergency_stop_active": self.emergency_stop_active,
                "active_alarms_count": len(self.active_alarms),
                "active_alarms": list(self.active_alarms),
                "total_critical_events": self.total_critical_events
            })
        return base_status
