"""Abstract base class for anomaly detectors"""

from abc import ABC, abstractmethod
from typing import List, Optional

from anomaly_detection.core.anomaly import Anomaly


class AnomalyDetector(ABC):
    """
    Abstract base class for all anomaly detectors.
    
    Each detector should implement the detect method which analyzes
    agent metrics and returns a list of detected anomalies.
    """
    
    def __init__(self, name: str, enabled: bool = True):
        """
        Initialize detector
        
        Args:
            name: Unique name for this detector
            enabled: Whether this detector is enabled
        """
        self.name = name
        self.enabled = enabled
    
    @abstractmethod
    def detect(self, agent_id: str, metrics: dict, historical_data: Optional[list] = None) -> List[Anomaly]:
        """
        Detect anomalies in agent metrics
        
        Args:
            agent_id: Identifier of the agent being monitored
            metrics: Current metrics dictionary from the agent
            historical_data: Optional list of previous metric snapshots
            
        Returns:
            List of detected anomalies (empty if none found)
        """
        pass
    
    def fit(self, data: list) -> None:
        """
        Train/fit the detector on historical data (optional)
        
        Args:
            data: List of metric dictionaries for training
        """
        # Default implementation does nothing
        # Override in subclasses that need training
        pass
    
    def reset(self) -> None:
        """Reset detector state"""
        # Default implementation does nothing
        # Override in subclasses that maintain state
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, enabled={self.enabled})"

