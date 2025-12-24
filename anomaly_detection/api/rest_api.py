"""REST API for anomaly detection system"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from anomaly_detection.system.system import AnomalyDetectionSystem
from anomaly_detection.core.anomaly import Anomaly, AnomalyType

# Prometheus integration (optional)
try:
    from anomaly_detection.api.prometheus import setup_prometheus_endpoint, PrometheusExporter
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    PrometheusExporter = None
    def setup_prometheus_endpoint(app, system): pass


app = FastAPI(title="Anomaly Detection API", version="0.1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instance (should be initialized by main)
system: Optional[AnomalyDetectionSystem] = None
prometheus_exporter: Optional[Any] = None


# Pydantic models for request/response
class AgentRegistration(BaseModel):
    agent_id: str
    metrics_endpoint: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AnomalyResponse(BaseModel):
    agent_id: str
    anomaly_type: str
    severity: float
    timestamp: str
    description: str
    detector_name: str
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]


class StatsResponse(BaseModel):
    total_anomalies: int
    unique_agents: Optional[int] = None
    by_type: Dict[str, int]
    by_severity: Dict[str, int]
    average_severity: Optional[float] = None


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global system
    # System should be set before starting the server
    if system is None:
        system = AnomalyDetectionSystem()


def set_system(sys: AnomalyDetectionSystem) -> None:
    """Set the system instance (called from main)"""
    global system, prometheus_exporter
    system = sys
    if system and PROMETHEUS_AVAILABLE and PrometheusExporter:
        prometheus_exporter = PrometheusExporter(system)
        setup_prometheus_endpoint(app, system)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Anomaly Detection API",
        "version": "0.1.0",
        "agents": len(system.list_agents()) if system else 0,
    }


@app.post("/agents/register")
async def register_agent(registration: AgentRegistration):
    """Register a new agent"""
    if not system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    agent = system.register_agent(
        agent_id=registration.agent_id,
        metrics_endpoint=registration.metrics_endpoint,
        metadata=registration.metadata
    )
    
    return {"agent_id": agent.agent_id, "status": "registered"}


@app.delete("/agents/{agent_id}")
async def unregister_agent(agent_id: str):
    """Unregister an agent"""
    if not system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    if agent_id not in system.list_agents():
        raise HTTPException(status_code=404, detail="Agent not found")
    
    system.unregister_agent(agent_id)
    return {"agent_id": agent_id, "status": "unregistered"}


@app.get("/agents")
async def list_agents():
    """List all registered agents"""
    if not system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    return {"agents": system.list_agents()}


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get agent information"""
    if not system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    agent = system.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return {
        "agent_id": agent.agent_id,
        "metrics_endpoint": agent.metrics_endpoint,
        "metadata": agent.metadata,
        "is_active": agent.is_active,
    }


@app.post("/agents/{agent_id}/check")
async def check_agent(agent_id: str):
    """Manually trigger anomaly detection for an agent"""
    if not system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    if agent_id not in system.list_agents():
        raise HTTPException(status_code=404, detail="Agent not found")
    
    anomalies = await system.check_agent(agent_id)
    
    return {
        "agent_id": agent_id,
        "anomalies_detected": len(anomalies),
        "anomalies": [anomaly.to_dict() for anomaly in anomalies],
    }


@app.get("/anomalies")
async def get_anomalies(
    hours: int = 24,
    agent_id: Optional[str] = None,
    anomaly_type: Optional[str] = None,
    min_severity: Optional[float] = None,
    limit: Optional[int] = None
):
    """Get recent anomalies"""
    if not system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        anomaly_type_enum = AnomalyType(anomaly_type) if anomaly_type else None
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid anomaly_type: {anomaly_type}")
    
    anomalies = system.get_recent_anomalies(
        hours=hours,
        agent_id=agent_id,
        anomaly_type=anomaly_type_enum,
        min_severity=min_severity,
        limit=limit
    )
    
    return {
        "count": len(anomalies),
        "anomalies": [anomaly.to_dict() for anomaly in anomalies],
    }


@app.get("/agents/{agent_id}/stats")
async def get_agent_stats(agent_id: str, hours: int = 24):
    """Get statistics for an agent"""
    if not system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    if agent_id not in system.list_agents():
        raise HTTPException(status_code=404, detail="Agent not found")
    
    stats = system.get_agent_stats(agent_id, hours)
    return stats


@app.get("/stats")
async def get_overall_stats(hours: int = 24):
    """Get overall system statistics"""
    if not system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    stats = system.get_overall_stats(hours)
    return stats


@app.post("/monitoring/start")
async def start_monitoring():
    """Start continuous monitoring"""
    if not system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    await system.start_monitoring()
    return {"status": "monitoring_started"}


@app.post("/monitoring/stop")
async def stop_monitoring():
    """Stop monitoring"""
    if not system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    await system.stop_monitoring()
    return {"status": "monitoring_stopped"}


# WebSocket endpoint for real-time anomaly notifications
@app.websocket("/ws/anomalies")
async def websocket_anomalies(websocket: WebSocket):
    """WebSocket endpoint for real-time anomaly notifications"""
    await websocket.accept()
    
    # Send initial connection message
    await websocket.send_json({"type": "connected", "message": "Connected to anomaly stream"})
    
    # This is a simplified implementation
    # In production, you'd want to use a proper pub/sub mechanism
    try:
        while True:
            # Wait for client messages (ping/pong for keepalive)
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system_initialized": system is not None,
        "agents_count": len(system.list_agents()) if system else 0,
    }

