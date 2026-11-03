"""
ElevatorGuard-AI: Multi-Agent Message Bus
Thread-safe, decoupled Publish/Subscribe communication bus with topic pattern matching,
asynchronous queuing, message audit log, and metrics tracking.
"""

import fnmatch
import time
import uuid
import threading
from typing import Dict, Any, List, Callable, Optional
from collections import deque
from dataclasses import dataclass, field


@dataclass
class AgentMessage:
    topic: str
    source: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "topic": self.topic,
            "source": self.source,
            "timestamp": round(self.timestamp, 4),
            "payload": self.payload,
            "metadata": self.metadata
        }


class MessageBus:
    """
    Decoupled event broker allowing autonomous agents to exchange
    sensor telemetry, vision defect analyses, fusion forecasts, and safety alerts.
    """

    def __init__(self, history_limit: int = 500):
        self.history_limit = history_limit
        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        self._history: deque = deque(maxlen=history_limit)
        self._lock = threading.RLock()
        self._total_published = 0
        self._total_delivered = 0
        self._topic_counts: Dict[str, int] = {}

    def subscribe(self, pattern: str, handler: Callable[[AgentMessage], None], subscriber_id: Optional[str] = None) -> str:
        """
        Subscribe a callback handler to a topic pattern (supports '*' and exact match).
        Returns a unique subscription ID.
        """
        sub_id = subscriber_id or f"sub_{uuid.uuid4().hex[:8]}"
        with self._lock:
            self._subscriptions[sub_id] = {
                "pattern": pattern,
                "handler": handler
            }
        return sub_id

    def unsubscribe(self, sub_id: str) -> bool:
        """Remove a subscription by ID."""
        with self._lock:
            if sub_id in self._subscriptions:
                del self._subscriptions[sub_id]
                return True
        return False

    def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        source: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """
        Publish a message to the bus. Matched subscribers will be invoked immediately.
        """
        msg = AgentMessage(
            topic=topic,
            source=source,
            payload=payload,
            metadata=metadata or {}
        )

        with self._lock:
            self._history.append(msg)
            self._total_published += 1
            self._topic_counts[topic] = self._topic_counts.get(topic, 0) + 1
            # Copy subscribers to avoid lock contention during dispatch
            active_subs = list(self._subscriptions.values())

        # Dispatch outside the lock to prevent handler deadlocks
        for sub in active_subs:
            pattern = sub["pattern"]
            if pattern == "*" or pattern == "#" or fnmatch.fnmatch(topic, pattern):
                try:
                    sub["handler"](msg)
                    with self._lock:
                        self._total_delivered += 1
                except Exception as e:
                    print(f"[MessageBus] Error dispatching message to subscriber: {e}")

        return msg

    def get_history(self, limit: int = 50, topic_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve recent message history formatted as dictionaries."""
        with self._lock:
            messages = list(self._history)

        if topic_filter:
            messages = [m for m in messages if fnmatch.fnmatch(m.topic, topic_filter)]

        return [m.to_dict() for m in messages[-limit:]]

    def clear(self):
        """Clear subscriptions and history."""
        with self._lock:
            self._subscriptions.clear()
            self._history.clear()
            self._total_published = 0
            self._total_delivered = 0
            self._topic_counts.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """Return bus operational metrics."""
        with self._lock:
            return {
                "active_subscriptions": len(self._subscriptions),
                "total_published": self._total_published,
                "total_delivered": self._total_delivered,
                "history_size": len(self._history),
                "topic_counts": dict(self._topic_counts)
            }
