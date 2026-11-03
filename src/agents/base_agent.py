"""
ElevatorGuard-AI: Base Autonomous Agent
Provides standard lifecycle controls, background worker thread execution,
message queue handling, and heartbeat health metrics for multi-agent pipeline components.
"""

import time
import queue
import threading
from enum import Enum
from typing import Dict, Any, Optional
from src.agents.message_bus import MessageBus, AgentMessage


class AgentStatus(str, Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


class BaseAgent:
    """
    Abstract base class for all ElevatorGuard-AI autonomous pipeline agents.
    Runs a non-blocking dedicated worker thread processing incoming messages and periodic ticks.
    """

    def __init__(
        self,
        name: str,
        message_bus: MessageBus,
        tick_interval: float = 0.05,
        inbox_maxsize: int = 1000
    ):
        self.name = name
        self.message_bus = message_bus
        self.tick_interval = tick_interval
        self.status = AgentStatus.STOPPED
        self.inbox: queue.Queue = queue.Queue(maxsize=inbox_maxsize)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._subscriptions = []

        # Operational metrics
        self.start_time: Optional[float] = None
        self.last_active_time: Optional[float] = None
        self.messages_processed = 0
        self.messages_published = 0
        self.error_count = 0
        self.last_error: Optional[str] = None
        self._lock = threading.RLock()

    def subscribe(self, pattern: str):
        """Subscribe this agent's inbox to messages matching the pattern."""
        def _inbox_handler(msg: AgentMessage):
            # Don't queue messages published by this agent itself
            if msg.source == self.name:
                return
            try:
                self.inbox.put_nowait(msg)
            except queue.Full:
                print(f"[{self.name}] Warning: Inbox queue full. Dropped message on topic '{msg.topic}'")

        sub_id = self.message_bus.subscribe(pattern, _inbox_handler, subscriber_id=f"{self.name}_{pattern}")
        self._subscriptions.append(sub_id)
        return sub_id

    def publish(self, topic: str, payload: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> AgentMessage:
        """Publish a message from this agent onto the bus."""
        msg = self.message_bus.publish(
            topic=topic,
            payload=payload,
            source=self.name,
            metadata=metadata or {}
        )
        with self._lock:
            self.messages_published += 1
            self.last_active_time = time.time()
        return msg

    def start(self):
        """Start the agent's dedicated worker loop thread."""
        with self._lock:
            if self.status == AgentStatus.RUNNING:
                return
            self.status = AgentStatus.STARTING
            self._stop_event.clear()
            self._pause_event.clear()
            self.start_time = time.time()
            self.last_active_time = time.time()
            self._thread = threading.Thread(target=self._run_loop, name=f"Agent-{self.name}", daemon=True)
            self._thread.start()
            self.status = AgentStatus.RUNNING
        self.on_start()

    def stop(self, timeout: float = 2.0):
        """Signal and wait for worker thread termination."""
        with self._lock:
            if self.status == AgentStatus.STOPPED:
                return
            self._stop_event.set()
            self._pause_event.clear()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            self._thread = None

        with self._lock:
            self.status = AgentStatus.STOPPED
        self.on_stop()

    def pause(self):
        """Temporarily pause message processing."""
        self._pause_event.set()
        with self._lock:
            self.status = AgentStatus.PAUSED

    def resume(self):
        """Resume message processing."""
        self._pause_event.clear()
        with self._lock:
            self.status = AgentStatus.RUNNING

    def _run_loop(self):
        """Internal execution loop executed by worker thread."""
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(self.tick_interval)
                continue

            # Process pending messages in inbox
            try:
                msg: AgentMessage = self.inbox.get(timeout=self.tick_interval)
                try:
                    self.process_message(msg)
                    with self._lock:
                        self.messages_processed += 1
                        self.last_active_time = time.time()
                except Exception as e:
                    with self._lock:
                        self.error_count += 1
                        self.last_error = str(e)
                    print(f"[{self.name}] Error handling message {msg.topic}: {e}")
                finally:
                    self.inbox.task_done()
            except queue.Empty:
                pass

            # Execute periodic tick
            try:
                self.on_tick()
            except Exception as e:
                with self._lock:
                    self.error_count += 1
                    self.last_error = str(e)
                print(f"[{self.name}] Error in on_tick: {e}")

    def process_message(self, msg: AgentMessage):
        """Child classes implement specific message handling logic."""
        pass

    def on_tick(self):
        """Periodic background hook for heartbeats, timeouts, and state checks."""
        pass

    def on_start(self):
        """Lifecycle hook executed upon start."""
        pass

    def on_stop(self):
        """Lifecycle hook executed upon stop."""
        pass

    def get_status(self) -> Dict[str, Any]:
        """Return standardized status and health dictionary."""
        with self._lock:
            uptime = round(time.time() - self.start_time, 2) if self.start_time and self.status == AgentStatus.RUNNING else 0.0
            return {
                "name": self.name,
                "status": self.status.value,
                "uptime_seconds": uptime,
                "inbox_size": self.inbox.qsize(),
                "messages_processed": self.messages_processed,
                "messages_published": self.messages_published,
                "error_count": self.error_count,
                "last_error": self.last_error,
                "last_active": round(self.last_active_time, 2) if self.last_active_time else None
            }
