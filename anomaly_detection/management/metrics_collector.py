"""Metrics collection and storage"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict
import aiohttp

from anomaly_detection.core.monitor import Agent


class MetricsCollector:
    """
    Collects and stores metrics from agents
    
    Features:
    - Async metric collection
    - Metric aggregation and normalization
    - In-memory storage (can be extended to use databases)
    - Historical data retrieval
    """
    
    def __init__(
        self,
        storage_backend: Optional[Any] = None,
        default_collector: Optional[Callable] = None
    ):
        """
        Initialize metrics collector
        
        Args:
            storage_backend: Optional storage backend (e.g., database connection)
            default_collector: Default function to collect metrics from agents
        """
        self.storage_backend = storage_backend
        self.default_collector = default_collector or self._default_collect_metrics
        
        # In-memory storage (fallback)
        self._metrics_store: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._max_storage_size = 10000  # Max metrics per agent
    
    async def collect_from_agent(self, agent: Agent) -> Dict[str, Any]:
        """
        Collect metrics from an agent
        
        Args:
            agent: Agent to collect metrics from
            
        Returns:
            Dictionary of metrics
        """
        try:
            metrics = await self.default_collector(agent)
            metrics["agent_id"] = agent.agent_id
            metrics["collected_at"] = datetime.now().isoformat()
            
            # Store metrics
            self._store_metrics(agent.agent_id, metrics)
            
            return metrics
        except Exception as e:
            # Return error metrics
            return {
                "agent_id": agent.agent_id,
                "error": str(e),
                "collected_at": datetime.now().isoformat(),
            }
    
    async def _default_collect_metrics(self, agent: Agent) -> Dict[str, Any]:
        """Default metric collection implementation"""
        if agent.metrics_endpoint:
            # Try to fetch from endpoint
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(agent.metrics_endpoint, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            return await resp.json()
            except Exception:
                pass
        
        # Default: return basic metrics
        return {
            "timestamp": datetime.now().isoformat(),
            "is_active": agent.is_active,
        }
    
    def _store_metrics(self, agent_id: str, metrics: Dict[str, Any]) -> None:
        """Store metrics (in-memory or backend)"""
        if self.storage_backend:
            # Use backend storage
            # This would be implemented based on the backend type
            pass
        else:
            # In-memory storage
            self._metrics_store[agent_id].append(metrics)
            
            # Limit storage size
            if len(self._metrics_store[agent_id]) > self._max_storage_size:
                # Remove oldest entries
                self._metrics_store[agent_id] = self._metrics_store[agent_id][-self._max_storage_size:]
    
    def get_metrics_history(
        self,
        agent_id: str,
        hours: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical metrics for an agent
        
        Args:
            agent_id: Agent ID
            hours: Number of hours to look back (None = all)
            limit: Maximum number of entries
            
        Returns:
            List of metrics dictionaries
        """
        if agent_id not in self._metrics_store:
            return []
        
        metrics = self._metrics_store[agent_id]
        
        # Filter by time
        if hours:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            metrics = [
                m for m in metrics
                if self._parse_timestamp(m.get("collected_at") or m.get("timestamp")) >= cutoff_time
            ]
        
        # Apply limit
        if limit:
            metrics = metrics[-limit:]
        
        return metrics
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime"""
        if isinstance(timestamp_str, datetime):
            return timestamp_str
        
        try:
            return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except Exception:
            return datetime.now()
    
    def get_all_agents(self) -> List[str]:
        """Get list of all agents with stored metrics"""
        return list(self._metrics_store.keys())
    
    def get_latest_metrics(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get latest metrics for an agent"""
        history = self.get_metrics_history(agent_id, limit=1)
        return history[0] if history else None
    
    def aggregate_metrics(
        self,
        agent_id: str,
        hours: int = 1,
        aggregation: str = "average"
    ) -> Dict[str, Any]:
        """
        Aggregate metrics over time period
        
        Args:
            agent_id: Agent ID
            hours: Hours to aggregate over
            aggregation: Type of aggregation ('average', 'sum', 'min', 'max', 'latest')
            
        Returns:
            Aggregated metrics dictionary
        """
        metrics = self.get_metrics_history(agent_id, hours=hours)
        
        if not metrics:
            return {}
        
        # Extract numeric metrics
        numeric_metrics = {}
        for m in metrics:
            for key, value in m.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)
        
        # Aggregate
        aggregated = {"agent_id": agent_id, "aggregation": aggregation, "period_hours": hours}
        
        for key, values in numeric_metrics.items():
            if not values:
                continue
            
            if aggregation == "average":
                aggregated[key] = sum(values) / len(values)
            elif aggregation == "sum":
                aggregated[key] = sum(values)
            elif aggregation == "min":
                aggregated[key] = min(values)
            elif aggregation == "max":
                aggregated[key] = max(values)
            elif aggregation == "latest":
                aggregated[key] = values[-1]
        
        return aggregated
    
    def clear_old_metrics(self, days: int = 30) -> int:
        """
        Clear metrics older than specified days
        
        Returns:
            Number of metrics removed
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        removed_count = 0
        
        for agent_id in list(self._metrics_store.keys()):
            original_count = len(self._metrics_store[agent_id])
            self._metrics_store[agent_id] = [
                m for m in self._metrics_store[agent_id]
                if self._parse_timestamp(m.get("collected_at") or m.get("timestamp")) >= cutoff_time
            ]
            removed_count += original_count - len(self._metrics_store[agent_id])
            
            # Remove empty entries
            if not self._metrics_store[agent_id]:
                del self._metrics_store[agent_id]
        
        return removed_count

