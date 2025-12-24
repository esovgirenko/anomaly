"""LLM-specific anomaly detectors"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

from anomaly_detection.core.detector import AnomalyDetector
from anomaly_detection.core.anomaly import Anomaly, AnomalyType
from anomaly_detection.core.llm_agent import LLMAgentMetrics


class TokenUsageDetector(AnomalyDetector):
    """
    Detects anomalies in token usage patterns
    
    Monitors:
    - Unusually high token consumption
    - Token usage spikes
    - Inefficient token usage (high input/output ratio)
    """
    
    def __init__(
        self,
        name: str = "token_usage",
        enabled: bool = True,
        token_spike_threshold: float = 3.0,  # Z-score threshold
        max_tokens_threshold: Optional[int] = None,
        inefficient_ratio_threshold: float = 10.0,  # input/output ratio
        min_samples: int = 10
    ):
        super().__init__(name, enabled)
        self.token_spike_threshold = token_spike_threshold
        self.max_tokens_threshold = max_tokens_threshold
        self.inefficient_ratio_threshold = inefficient_ratio_threshold
        self.min_samples = min_samples
        self._token_history: List[int] = []
    
    def detect(
        self,
        agent_id: str,
        metrics: Dict[str, Any],
        historical_data: Optional[List[Dict]] = None
    ) -> List[Anomaly]:
        """Detect token usage anomalies"""
        anomalies = []
        
        total_tokens = metrics.get("total_tokens", 0)
        input_tokens = metrics.get("input_tokens", 0)
        output_tokens = metrics.get("output_tokens", 0)
        
        if total_tokens == 0:
            return anomalies
        
        # Update history
        self._token_history.append(total_tokens)
        if len(self._token_history) > 1000:
            self._token_history.pop(0)
        
        # Check max tokens threshold
        if self.max_tokens_threshold and total_tokens > self.max_tokens_threshold:
            severity = min(1.0, total_tokens / (self.max_tokens_threshold * 2))
            anomalies.append(Anomaly(
                agent_id=agent_id,
                anomaly_type=AnomalyType.TOKEN_USAGE,
                severity=severity,
                timestamp=datetime.now(),
                description=f"Token usage exceeds threshold: {total_tokens} > {self.max_tokens_threshold}",
                detector_name=self.name,
                metrics=metrics.copy(),
                metadata={
                    "total_tokens": total_tokens,
                    "threshold": self.max_tokens_threshold,
                }
            ))
        
        # Check for token spike (if we have enough history)
        if len(self._token_history) >= self.min_samples:
            historical_values = np.array(self._token_history[:-1])
            mean_tokens = np.mean(historical_values)
            std_tokens = np.std(historical_values)
            
            if std_tokens > 0:
                z_score = abs((total_tokens - mean_tokens) / std_tokens)
                if z_score > self.token_spike_threshold:
                    severity = min(1.0, z_score / (self.token_spike_threshold * 2))
                    anomalies.append(Anomaly(
                        agent_id=agent_id,
                        anomaly_type=AnomalyType.TOKEN_USAGE,
                        severity=severity,
                        timestamp=datetime.now(),
                        description=f"Token usage spike detected: {total_tokens} tokens (Z-score: {z_score:.2f}, mean: {mean_tokens:.0f})",
                        detector_name=self.name,
                        metrics=metrics.copy(),
                        metadata={
                            "z_score": z_score,
                            "mean_tokens": mean_tokens,
                            "std_tokens": std_tokens,
                        }
                    ))
        
        # Check for inefficient token usage (high input/output ratio)
        if output_tokens > 0:
            ratio = input_tokens / output_tokens
            if ratio > self.inefficient_ratio_threshold:
                severity = min(1.0, (ratio / self.inefficient_ratio_threshold) * 0.7)
                anomalies.append(Anomaly(
                    agent_id=agent_id,
                    anomaly_type=AnomalyType.TOKEN_USAGE,
                    severity=severity,
                    timestamp=datetime.now(),
                    description=f"Inefficient token usage: input/output ratio {ratio:.2f} (threshold: {self.inefficient_ratio_threshold})",
                    detector_name=self.name,
                    metrics=metrics.copy(),
                    metadata={
                        "input_output_ratio": ratio,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                ))
        
        return anomalies
    
    def reset(self) -> None:
        self._token_history.clear()


class LatencyDetector(AnomalyDetector):
    """
    Detects anomalies in response latency
    
    Monitors:
    - Unusually high latency
    - Latency spikes
    - Degrading latency trends
    """
    
    def __init__(
        self,
        name: str = "latency",
        enabled: bool = True,
        latency_threshold_ms: Optional[float] = None,
        spike_threshold: float = 3.0,
        min_samples: int = 10
    ):
        super().__init__(name, enabled)
        self.latency_threshold_ms = latency_threshold_ms
        self.spike_threshold = spike_threshold
        self.min_samples = min_samples
        self._latency_history: List[float] = []
    
    def detect(
        self,
        agent_id: str,
        metrics: Dict[str, Any],
        historical_data: Optional[List[Dict]] = None
    ) -> List[Anomaly]:
        """Detect latency anomalies"""
        anomalies = []
        
        latency_ms = metrics.get("latency_ms", 0.0)
        if latency_ms <= 0:
            return anomalies
        
        # Update history
        self._latency_history.append(latency_ms)
        if len(self._latency_history) > 1000:
            self._latency_history.pop(0)
        
        # Check absolute threshold
        if self.latency_threshold_ms and latency_ms > self.latency_threshold_ms:
            severity = min(1.0, latency_ms / (self.latency_threshold_ms * 2))
            anomalies.append(Anomaly(
                agent_id=agent_id,
                anomaly_type=AnomalyType.LATENCY,
                severity=severity,
                timestamp=datetime.now(),
                description=f"Latency exceeds threshold: {latency_ms:.0f}ms > {self.latency_threshold_ms:.0f}ms",
                detector_name=self.name,
                metrics=metrics.copy(),
                metadata={
                    "latency_ms": latency_ms,
                    "threshold_ms": self.latency_threshold_ms,
                }
            ))
        
        # Check for latency spike
        if len(self._latency_history) >= self.min_samples:
            historical_values = np.array(self._latency_history[:-1])
            mean_latency = np.mean(historical_values)
            std_latency = np.std(historical_values)
            
            if std_latency > 0:
                z_score = (latency_ms - mean_latency) / std_latency
                if z_score > self.spike_threshold:
                    severity = min(1.0, z_score / (self.spike_threshold * 2))
                    anomalies.append(Anomaly(
                        agent_id=agent_id,
                        anomaly_type=AnomalyType.LATENCY,
                        severity=severity,
                        timestamp=datetime.now(),
                        description=f"Latency spike detected: {latency_ms:.0f}ms (Z-score: {z_score:.2f}, mean: {mean_latency:.0f}ms)",
                        detector_name=self.name,
                        metrics=metrics.copy(),
                        metadata={
                            "z_score": z_score,
                            "mean_latency_ms": mean_latency,
                            "std_latency_ms": std_latency,
                        }
                    ))
        
        return anomalies
    
    def reset(self) -> None:
        self._latency_history.clear()


class CostDetector(AnomalyDetector):
    """
    Detects anomalies in API costs
    
    Monitors:
    - Unusually high costs
    - Cost spikes
    - Cost trends
    """
    
    def __init__(
        self,
        name: str = "cost",
        enabled: bool = True,
        cost_threshold_usd: Optional[float] = None,
        spike_threshold: float = 3.0,
        daily_budget_usd: Optional[float] = None,
        min_samples: int = 10
    ):
        super().__init__(name, enabled)
        self.cost_threshold_usd = cost_threshold_usd
        self.spike_threshold = spike_threshold
        self.daily_budget_usd = daily_budget_usd
        self.min_samples = min_samples
        self._cost_history: List[float] = []
        self._daily_cost: float = 0.0
        self._last_reset: datetime = datetime.now()
    
    def detect(
        self,
        agent_id: str,
        metrics: Dict[str, Any],
        historical_data: Optional[List[Dict]] = None
    ) -> List[Anomaly]:
        """Detect cost anomalies"""
        anomalies = []
        
        cost_usd = metrics.get("cost_usd", 0.0)
        if cost_usd <= 0:
            return anomalies
        
        # Reset daily cost if needed
        now = datetime.now()
        if (now - self._last_reset).days >= 1:
            self._daily_cost = 0.0
            self._last_reset = now
        
        self._daily_cost += cost_usd
        
        # Check daily budget
        if self.daily_budget_usd and self._daily_cost > self.daily_budget_usd:
            severity = min(1.0, self._daily_cost / (self.daily_budget_usd * 1.5))
            anomalies.append(Anomaly(
                agent_id=agent_id,
                anomaly_type=AnomalyType.COST,
                severity=severity,
                timestamp=datetime.now(),
                description=f"Daily budget exceeded: ${self._daily_cost:.4f} > ${self.daily_budget_usd:.4f}",
                detector_name=self.name,
                metrics=metrics.copy(),
                metadata={
                    "daily_cost_usd": self._daily_cost,
                    "budget_usd": self.daily_budget_usd,
                }
            ))
        
        # Check per-request threshold
        if self.cost_threshold_usd and cost_usd > self.cost_threshold_usd:
            severity = min(1.0, cost_usd / (self.cost_threshold_usd * 2))
            anomalies.append(Anomaly(
                agent_id=agent_id,
                anomaly_type=AnomalyType.COST,
                severity=severity,
                timestamp=datetime.now(),
                description=f"Cost per request exceeds threshold: ${cost_usd:.4f} > ${self.cost_threshold_usd:.4f}",
                detector_name=self.name,
                metrics=metrics.copy(),
                metadata={
                    "cost_usd": cost_usd,
                    "threshold_usd": self.cost_threshold_usd,
                }
            ))
        
        # Update history and check for spikes
        self._cost_history.append(cost_usd)
        if len(self._cost_history) > 1000:
            self._cost_history.pop(0)
        
        if len(self._cost_history) >= self.min_samples:
            historical_values = np.array(self._cost_history[:-1])
            mean_cost = np.mean(historical_values)
            std_cost = np.std(historical_values)
            
            if std_cost > 0:
                z_score = (cost_usd - mean_cost) / std_cost
                if z_score > self.spike_threshold:
                    severity = min(1.0, z_score / (self.spike_threshold * 2))
                    anomalies.append(Anomaly(
                        agent_id=agent_id,
                        anomaly_type=AnomalyType.COST,
                        severity=severity,
                        timestamp=datetime.now(),
                        description=f"Cost spike detected: ${cost_usd:.4f} (Z-score: {z_score:.2f}, mean: ${mean_cost:.4f})",
                        detector_name=self.name,
                        metrics=metrics.copy(),
                        metadata={
                            "z_score": z_score,
                            "mean_cost_usd": mean_cost,
                            "std_cost_usd": std_cost,
                        }
                    ))
        
        return anomalies
    
    def reset(self) -> None:
        self._cost_history.clear()
        self._daily_cost = 0.0
        self._last_reset = datetime.now()


class QualityDetector(AnomalyDetector):
    """
    Detects quality degradation in LLM responses
    
    Monitors:
    - Quality score drops
    - Hallucinations
    - Factual errors
    - Coherence and relevance scores
    """
    
    def __init__(
        self,
        name: str = "quality",
        enabled: bool = True,
        quality_threshold: float = 0.7,
        coherence_threshold: float = 0.7,
        relevance_threshold: float = 0.7,
        hallucination_severity: float = 0.9,
        factual_error_severity: float = 0.8
    ):
        super().__init__(name, enabled)
        self.quality_threshold = quality_threshold
        self.coherence_threshold = coherence_threshold
        self.relevance_threshold = relevance_threshold
        self.hallucination_severity = hallucination_severity
        self.factual_error_severity = factual_error_severity
    
    def detect(
        self,
        agent_id: str,
        metrics: Dict[str, Any],
        historical_data: Optional[List[Dict]] = None
    ) -> List[Anomaly]:
        """Detect quality issues"""
        anomalies = []
        
        quality_score = metrics.get("quality_score", 1.0)
        coherence_score = metrics.get("coherence_score", 1.0)
        relevance_score = metrics.get("relevance_score", 1.0)
        hallucination_detected = metrics.get("hallucination_detected", False)
        factual_errors = metrics.get("factual_errors", 0)
        
        # Check for hallucinations
        if hallucination_detected:
            anomalies.append(Anomaly(
                agent_id=agent_id,
                anomaly_type=AnomalyType.QUALITY,
                severity=self.hallucination_severity,
                timestamp=datetime.now(),
                description="Hallucination detected in LLM response",
                detector_name=self.name,
                metrics=metrics.copy(),
                metadata={
                    "issue_type": "hallucination",
                }
            ))
        
        # Check for factual errors
        if factual_errors > 0:
            severity = min(1.0, self.factual_error_severity + (factual_errors - 1) * 0.1)
            anomalies.append(Anomaly(
                agent_id=agent_id,
                anomaly_type=AnomalyType.QUALITY,
                severity=severity,
                timestamp=datetime.now(),
                description=f"Factual errors detected: {factual_errors} error(s)",
                detector_name=self.name,
                metrics=metrics.copy(),
                metadata={
                    "issue_type": "factual_errors",
                    "error_count": factual_errors,
                }
            ))
        
        # Check quality score
        if quality_score < self.quality_threshold:
            severity = 1.0 - quality_score
            anomalies.append(Anomaly(
                agent_id=agent_id,
                anomaly_type=AnomalyType.QUALITY,
                severity=severity,
                timestamp=datetime.now(),
                description=f"Quality score below threshold: {quality_score:.2f} < {self.quality_threshold}",
                detector_name=self.name,
                metrics=metrics.copy(),
                metadata={
                    "quality_score": quality_score,
                    "threshold": self.quality_threshold,
                }
            ))
        
        # Check coherence
        if coherence_score < self.coherence_threshold:
            severity = 1.0 - coherence_score
            anomalies.append(Anomaly(
                agent_id=agent_id,
                anomaly_type=AnomalyType.QUALITY,
                severity=severity,
                timestamp=datetime.now(),
                description=f"Coherence score below threshold: {coherence_score:.2f} < {self.coherence_threshold}",
                detector_name=self.name,
                metrics=metrics.copy(),
                metadata={
                    "coherence_score": coherence_score,
                    "threshold": self.coherence_threshold,
                }
            ))
        
        # Check relevance
        if relevance_score < self.relevance_threshold:
            severity = 1.0 - relevance_score
            anomalies.append(Anomaly(
                agent_id=agent_id,
                anomaly_type=AnomalyType.QUALITY,
                severity=severity,
                timestamp=datetime.now(),
                description=f"Relevance score below threshold: {relevance_score:.2f} < {self.relevance_threshold}",
                detector_name=self.name,
                metrics=metrics.copy(),
                metadata={
                    "relevance_score": relevance_score,
                    "threshold": self.relevance_threshold,
                }
            ))
        
        return anomalies


class RateLimitDetector(AnomalyDetector):
    """
    Detects rate limit issues
    
    Monitors:
    - Approaching rate limits
    - Rate limit exceeded
    """
    
    def __init__(
        self,
        name: str = "rate_limit",
        enabled: bool = True,
        warning_threshold: float = 0.8  # Warn at 80% of limit
    ):
        super().__init__(name, enabled)
        self.warning_threshold = warning_threshold
    
    def detect(
        self,
        agent_id: str,
        metrics: Dict[str, Any],
        historical_data: Optional[List[Dict]] = None
    ) -> List[Anomaly]:
        """Detect rate limit issues"""
        anomalies = []
        
        rate_limit_remaining = metrics.get("rate_limit_remaining")
        requests_per_minute = metrics.get("requests_per_minute", 0.0)
        
        if rate_limit_remaining is None:
            return anomalies
        
        # Calculate usage ratio (if we know the limit)
        # This is a simplified check - in practice you'd track the actual limit
        
        if rate_limit_remaining == 0:
            anomalies.append(Anomaly(
                agent_id=agent_id,
                anomaly_type=AnomalyType.RATE_LIMIT,
                severity=0.9,
                timestamp=datetime.now(),
                description="Rate limit exceeded - no remaining requests",
                detector_name=self.name,
                metrics=metrics.copy(),
                metadata={
                    "rate_limit_remaining": rate_limit_remaining,
                    "requests_per_minute": requests_per_minute,
                }
            ))
        elif rate_limit_remaining < 10:  # Arbitrary threshold for "low"
            # Try to infer limit from history or use default
            inferred_limit = rate_limit_remaining + requests_per_minute
            if inferred_limit > 0:
                usage_ratio = requests_per_minute / inferred_limit
                if usage_ratio >= self.warning_threshold:
                    severity = min(0.8, usage_ratio)
                    anomalies.append(Anomaly(
                        agent_id=agent_id,
                        anomaly_type=AnomalyType.RATE_LIMIT,
                        severity=severity,
                        timestamp=datetime.now(),
                        description=f"Approaching rate limit: {rate_limit_remaining} remaining, {requests_per_minute:.1f} req/min",
                        detector_name=self.name,
                        metrics=metrics.copy(),
                        metadata={
                            "rate_limit_remaining": rate_limit_remaining,
                            "requests_per_minute": requests_per_minute,
                            "usage_ratio": usage_ratio,
                        }
                    ))
        
        return anomalies


class ContextOverflowDetector(AnomalyDetector):
    """
    Detects context window overflow issues
    
    Monitors:
    - Context window usage approaching limits
    - Context overflow
    """
    
    def __init__(
        self,
        name: str = "context_overflow",
        enabled: bool = True,
        warning_threshold: float = 0.85,  # Warn at 85% usage
        critical_threshold: float = 0.95   # Critical at 95% usage
    ):
        super().__init__(name, enabled)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
    
    def detect(
        self,
        agent_id: str,
        metrics: Dict[str, Any],
        historical_data: Optional[List[Dict]] = None
    ) -> List[Anomaly]:
        """Detect context overflow issues"""
        anomalies = []
        
        context_usage = metrics.get("context_window_usage", 0.0)
        
        if context_usage >= self.critical_threshold:
            severity = min(1.0, context_usage)
            anomalies.append(Anomaly(
                agent_id=agent_id,
                anomaly_type=AnomalyType.CONTEXT_OVERFLOW,
                severity=severity,
                timestamp=datetime.now(),
                description=f"Context window nearly full: {context_usage*100:.1f}% usage (critical threshold: {self.critical_threshold*100}%)",
                detector_name=self.name,
                metrics=metrics.copy(),
                metadata={
                    "context_usage": context_usage,
                    "threshold": self.critical_threshold,
                }
            ))
        elif context_usage >= self.warning_threshold:
            severity = (context_usage - self.warning_threshold) / (self.critical_threshold - self.warning_threshold) * 0.7 + 0.3
            anomalies.append(Anomaly(
                agent_id=agent_id,
                anomaly_type=AnomalyType.CONTEXT_OVERFLOW,
                severity=severity,
                timestamp=datetime.now(),
                description=f"Context window usage high: {context_usage*100:.1f}% usage (warning threshold: {self.warning_threshold*100}%)",
                detector_name=self.name,
                metrics=metrics.copy(),
                metadata={
                    "context_usage": context_usage,
                    "threshold": self.warning_threshold,
                }
            ))
        
        return anomalies

