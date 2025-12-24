"""Anomaly data structures and types"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class AnomalyType(Enum):
    """Types of anomalies that can be detected for LLM agents"""
    BEHAVIORAL = "behavioral"  # Отклонения в поведении агента
    PERFORMANCE = "performance"  # Проблемы производительности
    COMMUNICATION = "communication"  # Проблемы коммуникации
    TEMPORAL = "temporal"  # Временные аномалии
    SEMANTIC = "semantic"  # Семантические ошибки
    # LLM-specific types
    TOKEN_USAGE = "token_usage"  # Аномальное использование токенов
    LATENCY = "latency"  # Аномальная задержка ответа
    COST = "cost"  # Аномальные расходы на API
    QUALITY = "quality"  # Падение качества ответов (галлюцинации, ошибки)
    RATE_LIMIT = "rate_limit"  # Превышение лимитов API
    CONTEXT_OVERFLOW = "context_overflow"  # Переполнение контекста


class Severity(Enum):
    """Severity levels for anomalies"""
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    CRITICAL = 0.9


@dataclass
class Anomaly:
    """
    Represents an anomaly detected in an agent's behavior
    
    Attributes:
        agent_id: Unique identifier of the agent
        anomaly_type: Type of anomaly detected
        severity: Severity score (0.0 to 1.0)
        timestamp: When the anomaly was detected
        metrics: Dictionary of metric values at the time of anomaly
        description: Human-readable description
        detector_name: Name of the detector that found this anomaly
        metadata: Additional metadata about the anomaly
    """
    agent_id: str
    anomaly_type: AnomalyType
    severity: float
    timestamp: datetime
    description: str
    detector_name: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate anomaly data after initialization"""
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"Severity must be between 0.0 and 1.0, got {self.severity}")
        
        if not self.agent_id:
            raise ValueError("agent_id cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert anomaly to dictionary for serialization"""
        return {
            "agent_id": self.agent_id,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "detector_name": self.detector_name,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Anomaly":
        """Create anomaly from dictionary"""
        return cls(
            agent_id=data["agent_id"],
            anomaly_type=AnomalyType(data["anomaly_type"]),
            severity=data["severity"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            description=data["description"],
            detector_name=data["detector_name"],
            metrics=data.get("metrics", {}),
            metadata=data.get("metadata", {}),
        )

