"""LLM Agent specific classes and metrics"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Any, List
from enum import Enum

from anomaly_detection.core.monitor import Agent


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    CUSTOM = "custom"


class LLMModel(Enum):
    """Common LLM models"""
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    CLAUDE_3_OPUS = "claude-3-opus"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    GEMINI_PRO = "gemini-pro"
    UNKNOWN = "unknown"


@dataclass
class LLMAgentMetrics:
    """
    Metrics specific to LLM agents
    
    Attributes:
        timestamp: When metrics were collected
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        total_tokens: Total tokens (input + output)
        latency_ms: Response latency in milliseconds
        cost_usd: Cost in USD for this request
        requests_per_minute: Current request rate
        error_rate: Rate of errors (0.0 to 1.0)
        quality_score: Quality score of the response (0.0 to 1.0)
        context_window_usage: Percentage of context window used (0.0 to 1.0)
        rate_limit_remaining: Remaining rate limit
        rate_limit_reset_at: When rate limit resets
        model: Model name used
        provider: LLM provider
        response_time_p50: 50th percentile response time
        response_time_p95: 95th percentile response time
        response_time_p99: 99th percentile response time
        hallucination_detected: Whether hallucination was detected
        factual_errors: Number of factual errors detected
        coherence_score: Coherence score (0.0 to 1.0)
        relevance_score: Relevance score (0.0 to 1.0)
    """
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Token metrics
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Performance metrics
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    requests_per_minute: float = 0.0
    
    # Quality metrics
    error_rate: float = 0.0
    quality_score: float = 1.0
    hallucination_detected: bool = False
    factual_errors: int = 0
    coherence_score: float = 1.0
    relevance_score: float = 1.0
    
    # Context and limits
    context_window_usage: float = 0.0  # 0.0 to 1.0
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset_at: Optional[datetime] = None
    
    # Model info
    model: str = "unknown"
    provider: str = "custom"
    
    # Response time percentiles
    response_time_p50: Optional[float] = None
    response_time_p95: Optional[float] = None
    response_time_p99: Optional[float] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "timestamp": self.timestamp.isoformat(),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "requests_per_minute": self.requests_per_minute,
            "error_rate": self.error_rate,
            "quality_score": self.quality_score,
            "context_window_usage": self.context_window_usage,
            "model": self.model,
            "provider": self.provider,
            "hallucination_detected": self.hallucination_detected,
            "factual_errors": self.factual_errors,
            "coherence_score": self.coherence_score,
            "relevance_score": self.relevance_score,
        }
        
        if self.rate_limit_remaining is not None:
            result["rate_limit_remaining"] = self.rate_limit_remaining
        if self.rate_limit_reset_at:
            result["rate_limit_reset_at"] = self.rate_limit_reset_at.isoformat()
        if self.response_time_p50 is not None:
            result["response_time_p50"] = self.response_time_p50
        if self.response_time_p95 is not None:
            result["response_time_p95"] = self.response_time_p95
        if self.response_time_p99 is not None:
            result["response_time_p99"] = self.response_time_p99
        
        result["metadata"] = self.metadata
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMAgentMetrics":
        """Create from dictionary"""
        rate_limit_reset_at = None
        if "rate_limit_reset_at" in data and data["rate_limit_reset_at"]:
            rate_limit_reset_at = datetime.fromisoformat(data["rate_limit_reset_at"])
        
        timestamp = datetime.now()
        if "timestamp" in data:
            timestamp = datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"]
        
        return cls(
            timestamp=timestamp,
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            latency_ms=data.get("latency_ms", 0.0),
            cost_usd=data.get("cost_usd", 0.0),
            requests_per_minute=data.get("requests_per_minute", 0.0),
            error_rate=data.get("error_rate", 0.0),
            quality_score=data.get("quality_score", 1.0),
            context_window_usage=data.get("context_window_usage", 0.0),
            rate_limit_remaining=data.get("rate_limit_remaining"),
            rate_limit_reset_at=rate_limit_reset_at,
            model=data.get("model", "unknown"),
            provider=data.get("provider", "custom"),
            hallucination_detected=data.get("hallucination_detected", False),
            factual_errors=data.get("factual_errors", 0),
            coherence_score=data.get("coherence_score", 1.0),
            relevance_score=data.get("relevance_score", 1.0),
            response_time_p50=data.get("response_time_p50"),
            response_time_p95=data.get("response_time_p95"),
            response_time_p99=data.get("response_time_p99"),
            metadata=data.get("metadata", {}),
        )


class LLMAgent(Agent):
    """
    Represents an LLM agent being monitored
    
    LLM agents are autonomous systems based on large language models
    that can perceive environment, plan actions, and achieve goals.
    """
    
    def __init__(
        self,
        agent_id: str,
        model: str = "gpt-4",
        provider: str = "openai",
        metrics_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        metadata: Optional[Dict] = None,
        context_window_size: Optional[int] = None,
        rate_limit: Optional[int] = None,
    ):
        """
        Initialize LLM agent
        
        Args:
            agent_id: Unique identifier
            model: LLM model name (e.g., "gpt-4", "claude-3-opus")
            provider: LLM provider (openai, anthropic, google, etc.)
            metrics_endpoint: Optional endpoint to fetch metrics from
            api_key: Optional API key (stored in metadata)
            metadata: Optional metadata about the agent
            context_window_size: Maximum context window size in tokens
            rate_limit: Rate limit (requests per minute)
        """
        super().__init__(agent_id, metrics_endpoint, metadata)
        self.model = model
        self.provider = provider
        self.context_window_size = context_window_size
        self.rate_limit = rate_limit
        
        if api_key:
            if metadata is None:
                self.metadata = {}
            self.metadata["api_key"] = api_key  # In production, use secure storage
    
    def __repr__(self) -> str:
        return f"LLMAgent(id={self.agent_id}, model={self.model}, provider={self.provider})"

