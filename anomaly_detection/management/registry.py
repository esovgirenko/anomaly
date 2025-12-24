"""Anomaly registry for managing detected anomalies"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from collections import defaultdict

from anomaly_detection.core.anomaly import Anomaly, AnomalyType


class AnomalyRegistry:
    """
    Registry for managing detected anomalies
    
    Features:
    - Storage and retrieval of anomalies
    - Deduplication of similar anomalies
    - Prioritization by severity
    - Time-based filtering
    """
    
    def __init__(
        self,
        deduplication_window: timedelta = timedelta(minutes=5),
        similarity_threshold: float = 0.8
    ):
        """
        Initialize anomaly registry
        
        Args:
            deduplication_window: Time window for deduplication
            similarity_threshold: Similarity threshold for deduplication (0-1)
        """
        self._anomalies: List[Anomaly] = []
        self._agent_anomalies: Dict[str, List[Anomaly]] = defaultdict(list)
        self._type_anomalies: Dict[AnomalyType, List[Anomaly]] = defaultdict(list)
        self.deduplication_window = deduplication_window
        self.similarity_threshold = similarity_threshold
    
    def register(self, anomaly: Anomaly) -> bool:
        """
        Register a new anomaly
        
        Returns:
            True if anomaly was registered, False if it was deduplicated
        """
        # Check for duplicates
        if self._is_duplicate(anomaly):
            return False
        
        # Add to registry
        self._anomalies.append(anomaly)
        self._agent_anomalies[anomaly.agent_id].append(anomaly)
        self._type_anomalies[anomaly.anomaly_type].append(anomaly)
        
        return True
    
    def _is_duplicate(self, anomaly: Anomaly) -> bool:
        """Check if anomaly is duplicate of existing one"""
        cutoff_time = anomaly.timestamp - self.deduplication_window
        
        # Check recent anomalies for same agent
        recent_anomalies = [
            a for a in self._agent_anomalies.get(anomaly.agent_id, [])
            if a.timestamp >= cutoff_time
            and a.anomaly_type == anomaly.anomaly_type
            and a.detector_name == anomaly.detector_name
        ]
        
        for existing in recent_anomalies:
            if self._are_similar(anomaly, existing):
                return True
        
        return False
    
    def _are_similar(self, a1: Anomaly, a2: Anomaly) -> bool:
        """Check if two anomalies are similar"""
        # Same agent, type, and detector
        if (a1.agent_id != a2.agent_id or
            a1.anomaly_type != a2.anomaly_type or
            a1.detector_name != a2.detector_name):
            return False
        
        # Check metric similarity
        common_metrics = set(a1.metrics.keys()) & set(a2.metrics.keys())
        if not common_metrics:
            return False
        
        # Compare common metric values
        similarities = []
        for metric in common_metrics:
            val1 = a1.metrics[metric]
            val2 = a2.metrics[metric]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Calculate similarity for numeric values
                if val1 == 0 and val2 == 0:
                    similarity = 1.0
                else:
                    max_val = max(abs(val1), abs(val2))
                    if max_val == 0:
                        similarity = 1.0
                    else:
                        similarity = 1.0 - min(1.0, abs(val1 - val2) / max_val)
                similarities.append(similarity)
        
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            return avg_similarity >= self.similarity_threshold
        
        return False
    
    def get_recent_anomalies(
        self,
        hours: Optional[int] = None,
        agent_id: Optional[str] = None,
        anomaly_type: Optional[AnomalyType] = None,
        min_severity: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[Anomaly]:
        """
        Get recent anomalies with optional filters
        
        Args:
            hours: Number of hours to look back (None = all time)
            agent_id: Filter by agent ID
            anomaly_type: Filter by anomaly type
            min_severity: Minimum severity threshold
            limit: Maximum number of results
            
        Returns:
            List of anomalies, sorted by severity (descending) and timestamp (descending)
        """
        cutoff_time = None
        if hours is not None:
            cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Start with appropriate list
        if agent_id:
            candidates = self._agent_anomalies.get(agent_id, [])
        elif anomaly_type:
            candidates = self._type_anomalies.get(anomaly_type, [])
        else:
            candidates = self._anomalies
        
        # Apply filters
        filtered = candidates
        if cutoff_time:
            filtered = [a for a in filtered if a.timestamp >= cutoff_time]
        if agent_id and not agent_id:
            filtered = [a for a in filtered if a.agent_id == agent_id]
        if anomaly_type:
            filtered = [a for a in filtered if a.anomaly_type == anomaly_type]
        if min_severity is not None:
            filtered = [a for a in filtered if a.severity >= min_severity]
        
        # Sort by severity (descending), then timestamp (descending)
        sorted_anomalies = sorted(
            filtered,
            key=lambda x: (x.severity, x.timestamp),
            reverse=True
        )
        
        if limit:
            return sorted_anomalies[:limit]
        
        return sorted_anomalies
    
    def get_agent_stats(self, agent_id: str, hours: Optional[int] = 24) -> Dict:
        """Get statistics for a specific agent"""
        cutoff_time = datetime.now() - timedelta(hours=hours) if hours else None
        
        anomalies = self._agent_anomalies.get(agent_id, [])
        if cutoff_time:
            anomalies = [a for a in anomalies if a.timestamp >= cutoff_time]
        
        if not anomalies:
            return {
                "agent_id": agent_id,
                "total_anomalies": 0,
                "by_type": {},
                "by_severity": {},
                "average_severity": 0.0,
            }
        
        by_type = defaultdict(int)
        by_severity = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        total_severity = 0.0
        for a in anomalies:
            by_type[a.anomaly_type.value] += 1
            total_severity += a.severity
            
            if a.severity < 0.4:
                by_severity["low"] += 1
            elif a.severity < 0.6:
                by_severity["medium"] += 1
            elif a.severity < 0.8:
                by_severity["high"] += 1
            else:
                by_severity["critical"] += 1
        
        return {
            "agent_id": agent_id,
            "total_anomalies": len(anomalies),
            "by_type": dict(by_type),
            "by_severity": by_severity,
            "average_severity": total_severity / len(anomalies),
        }
    
    def get_overall_stats(self, hours: Optional[int] = 24) -> Dict:
        """Get overall statistics"""
        cutoff_time = datetime.now() - timedelta(hours=hours) if hours else None
        
        all_anomalies = self._anomalies
        if cutoff_time:
            all_anomalies = [a for a in all_anomalies if a.timestamp >= cutoff_time]
        
        if not all_anomalies:
            return {
                "total_anomalies": 0,
                "unique_agents": 0,
                "by_type": {},
                "by_severity": {},
            }
        
        by_type = defaultdict(int)
        by_severity = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        unique_agents = set()
        
        for a in all_anomalies:
            unique_agents.add(a.agent_id)
            by_type[a.anomaly_type.value] += 1
            
            if a.severity < 0.4:
                by_severity["low"] += 1
            elif a.severity < 0.6:
                by_severity["medium"] += 1
            elif a.severity < 0.8:
                by_severity["high"] += 1
            else:
                by_severity["critical"] += 1
        
        return {
            "total_anomalies": len(all_anomalies),
            "unique_agents": len(unique_agents),
            "by_type": dict(by_type),
            "by_severity": by_severity,
        }
    
    def clear_old_anomalies(self, days: int = 30) -> int:
        """
        Remove anomalies older than specified days
        
        Returns:
            Number of anomalies removed
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        
        old_count = len(self._anomalies)
        
        self._anomalies = [a for a in self._anomalies if a.timestamp >= cutoff_time]
        
        # Rebuild indexes
        self._agent_anomalies.clear()
        self._type_anomalies.clear()
        for anomaly in self._anomalies:
            self._agent_anomalies[anomaly.agent_id].append(anomaly)
            self._type_anomalies[anomaly.anomaly_type].append(anomaly)
        
        removed = old_count - len(self._anomalies)
        return removed

