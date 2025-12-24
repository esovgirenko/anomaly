"""Main entry point for the anomaly detection system"""

import asyncio
import argparse
import yaml
from pathlib import Path

from anomaly_detection.system.system import AnomalyDetectionSystem
from anomaly_detection.api.rest_api import app, set_system
import uvicorn


async def load_agents_from_config(system: AnomalyDetectionSystem, config_path: str):
    """Load agents from configuration file"""
    if not Path(config_path).exists():
        print(f"Warning: Agents config file {config_path} not found, skipping agent registration")
        return
    
    with open(config_path, 'r') as f:
        agents_config = yaml.safe_load(f)
    
    agents = agents_config.get("agents", [])
    for agent_config in agents:
        system.register_agent(
            agent_id=agent_config["agent_id"],
            metrics_endpoint=agent_config.get("metrics_endpoint"),
            metadata=agent_config.get("metadata", {})
        )
        print(f"Registered agent: {agent_config['agent_id']}")


def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection System")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--agents-config",
        type=str,
        default="config/agents.yaml",
        help="Path to agents configuration file"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind API server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind API server"
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Run without REST API server"
    )
    
    args = parser.parse_args()
    
    # Create system
    print("Initializing Anomaly Detection System...")
    system = AnomalyDetectionSystem(config_path=args.config)
    
    # Load agents
    asyncio.run(load_agents_from_config(system, args.agents_config))
    
    # Set system for API
    set_system(system)
    
    # Start monitoring
    print("Starting monitoring...")
    asyncio.create_task(system.start_monitoring())
    
    if not args.no_api:
        # Start API server
        print(f"Starting API server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        # Run monitoring loop
        print("Running in monitoring-only mode (no API)")
        async def run_monitoring():
            await system.start_monitoring()
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
                await system.stop_monitoring()
        
        asyncio.run(run_monitoring())


if __name__ == "__main__":
    main()

