"""Agent monitoring functionality"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Callable

from anomaly_detection.core.anomaly import Anomaly
from anomaly_detection.core.detector import AnomalyDetector


class Agent:
    """Represents an agent being monitored"""
    
    def __init__(
        self,
        agent_id: str,
        metrics_endpoint: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize agent
        
        Args:
            agent_id: Unique identifier
            metrics_endpoint: Optional endpoint to fetch metrics from
            metadata: Optional metadata about the agent
        """
        self.agent_id = agent_id
        self.metrics_endpoint = metrics_endpoint
        self.metadata = metadata or {}
        self.last_seen = datetime.now()
        self.is_active = True
    
    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, endpoint={self.metrics_endpoint})"


class AgentMonitor:
    """
    Monitors a single agent and runs anomaly detection
    
    Responsibilities:
    - Collect metrics from agent
    - Run detectors on collected metrics
    - Report detected anomalies
    """
    
    def __init__(
        self,
        agent: Agent,
        detectors: List[AnomalyDetector],
        metrics_collector: Optional[Callable] = None
    ):
        """
        Initialize agent monitor
        
        Args:
            agent: Agent instance to monitor
            detectors: List of detectors to run
            metrics_collector: Optional callable to collect metrics
        """
        self.agent = agent
        self.detectors = [d for d in detectors if d.enabled]
        self.metrics_collector = metrics_collector
        self._metrics_history: List[Dict] = []
        self._max_history_size = 1000
    
    async def collect_metrics(self) -> Dict:
        """
        Collect current metrics from the agent
        
        Returns:
            Dictionary of current metrics
        """
        if self.metrics_collector:
            try:
                metrics = await self.metrics_collector(self.agent)
                if not isinstance(metrics, dict):
                    raise ValueError("metrics_collector must return a dictionary")
                return metrics
            except Exception as e:
                # Return default metrics on error
                return {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
        
        # Default: return basic metrics
        return {
            "agent_id": self.agent.agent_id,
            "timestamp": datetime.now().isoformat(),
            "is_active": self.agent.is_active,
        }
    
    async def detect_anomalies(self, metrics: Optional[Dict] = None) -> List[Anomaly]:
        """
        Run all detectors on agent metrics
        
        Args:
            metrics: Optional metrics dictionary. If None, will collect fresh metrics.
            
        Returns:
            List of detected anomalies
        """
        if metrics is None:
            metrics = await self.collect_metrics()
        
        # Add to history
        self._metrics_history.append(metrics.copy())
        if len(self._metrics_history) > self._max_history_size:
            self._metrics_history.pop(0)
        
        # Get historical data (last N entries)
        historical_data = self._metrics_history[:-1] if len(self._metrics_history) > 1 else None
        
        # Run all enabled detectors
        all_anomalies = []
        for detector in self.detectors:
            try:
                anomalies = detector.detect(
                    agent_id=self.agent.agent_id,
                    metrics=metrics,
                    historical_data=historical_data
                )
                all_anomalies.extend(anomalies)
            except Exception as e:
                # Log error but continue with other detectors
                print(f"Error in detector {detector.name}: {e}")
        
        return all_anomalies
    
    async def report_anomaly(self, anomaly: Anomaly, callback: Optional[Callable] = None) -> None:
        """
        Report a detected anomaly
        
        Args:
            anomaly: The anomaly to report
            callback: Optional callback function to handle the anomaly
        """
        if callback:
            try:
                await callback(anomaly)
            except Exception as e:
                print(f"Error in anomaly callback: {e}")
    
    def get_metrics_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get historical metrics
        
        Args:
            limit: Optional limit on number of entries to return
            
        Returns:
            List of historical metrics
        """
        if limit:
            return self._metrics_history[-limit:]
        return self._metrics_history.copy()

