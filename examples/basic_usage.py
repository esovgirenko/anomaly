"""Basic usage example of the anomaly detection system"""

import asyncio
from datetime import datetime
from anomaly_detection import AnomalyDetectionSystem, Agent


async def example_basic_usage():
    """Basic example of using the anomaly detection system"""
    
    # Create system with configuration
    system = AnomalyDetectionSystem(config={
        "collection_interval_seconds": 5,
        "detectors": {
            "statistical": {"enabled": True, "z_score_threshold": 3.0},
            "rule_based": {"enabled": True},
        },
        "rules": [
            {
                "name": "high_cpu",
                "description": "CPU usage above 80%",
                "anomaly_type": "performance",
                "severity": 0.7,
                "logic": "AND",
                "conditions": [
                    {
                        "metric": "cpu_usage",
                        "operator": ">",
                        "value": 80.0
                    }
                ]
            }
        ]
    })
    
    # Register an agent
    agent = system.register_agent(
        agent_id="example_agent",
        metadata={"name": "Example Agent", "environment": "test"}
    )
    
    print(f"Registered agent: {agent.agent_id}")
    
    # Start monitoring
    await system.start_monitoring()
    print("Monitoring started...")
    
    # Simulate some metrics collection (in real scenario, agents would send metrics)
    # For this example, we'll manually check the agent a few times
    for i in range(5):
        await asyncio.sleep(2)
        
        # Manually trigger detection with simulated metrics
        # In production, metrics would come from the agent's endpoint
        monitor = system._monitors.get("example_agent")
        if monitor:
            # Simulate normal metrics
            metrics = {
                "cpu_usage": 45.0 + (i * 5),  # Gradually increasing
                "memory_usage": 60.0,
                "response_time_ms": 100.0,
                "timestamp": datetime.now().isoformat(),
            }
            anomalies = await monitor.detect_anomalies(metrics)
            
            if anomalies:
                print(f"\nAnomalies detected at iteration {i}:")
                for anomaly in anomalies:
                    print(f"  - {anomaly.description}")
                    print(f"    Severity: {anomaly.severity:.2f}")
                    # Register the anomaly
                    system.registry.register(anomaly)
    
    # Get recent anomalies
    print("\n" + "="*60)
    print("Recent anomalies:")
    anomalies = system.get_recent_anomalies(hours=1)
    for anomaly in anomalies:
        print(f"  Agent: {anomaly.agent_id}")
        print(f"  Type: {anomaly.anomaly_type.value}")
        print(f"  Severity: {anomaly.severity:.2f}")
        print(f"  Description: {anomaly.description}")
        print()
    
    # Get statistics
    print("="*60)
    print("Agent statistics:")
    stats = system.get_agent_stats("example_agent", hours=1)
    print(f"  Total anomalies: {stats['total_anomalies']}")
    print(f"  Average severity: {stats['average_severity']:.2f}")
    print(f"  By type: {stats['by_type']}")
    
    # Stop monitoring
    await system.stop_monitoring()
    print("\nMonitoring stopped.")


if __name__ == "__main__":
    asyncio.run(example_basic_usage())

