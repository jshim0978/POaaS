#!/usr/bin/env python3
"""
Start all POaaS services for experiments.

This script starts the POaaS orchestrator and all worker services
in the correct order for running experiments.

Usage:
    python scripts/start_services.py
"""

import subprocess
import sys
import time
import requests
import os
from pathlib import Path

def check_service(url: str, service_name: str) -> bool:
    """Check if a service is running and healthy."""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_service(script_path: str, port: int, service_name: str):
    """Start a service and return the process."""
    print(f"Starting {service_name} on port {port}...")
    
    # Add explicit port environment and uvicorn command for workers
    env = os.environ.copy()
    env["PORT"] = str(port)
    
    if "worker" in script_path:
        # Workers use uvicorn to start
        cmd = [sys.executable, "-m", "uvicorn", f"{script_path.replace('/', '.').replace('.py', '')}:app", "--host", "0.0.0.0", "--port", str(port)]
    elif "orchestrator" in script_path:
        # Orchestrator has its own main
        cmd = [sys.executable, script_path, "--port", str(port)]
    else:
        cmd = [sys.executable, script_path]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=Path(script_path).parent.parent  # Set working directory to project root
    )
    
    # Wait a moment for service to start
    time.sleep(2)
    
    # Check if service is healthy
    if check_service(f"http://localhost:{port}", service_name):
        print(f"✓ {service_name} started successfully on port {port}")
        return process
    else:
        print(f"✗ Failed to start {service_name}")
        process.terminate()
        return None

def main():
    print("Starting POaaS services...")
    
    services = [
        ("workers/cleaner/app.py", 8002, "Cleaner Worker"),
        ("workers/paraphraser/app.py", 8003, "Paraphraser Worker"), 
        ("workers/fact_adder/app.py", 8004, "Fact-Adder Worker"),
        ("orchestrator/app.py", 8001, "POaaS Orchestrator")
    ]
    
    processes = []
    
    for script_path, port, service_name in services:
        if not Path(script_path).exists():
            print(f"✗ Service script not found: {script_path}")
            continue
            
        process = start_service(script_path, port, service_name)
        if process:
            processes.append((process, service_name))
        else:
            print(f"Failed to start {service_name}, stopping all services")
            for p, name in processes:
                p.terminate()
            return False
    
    print(f"\n✓ All services started successfully!")
    print("\nService endpoints:")
    print("  - POaaS Orchestrator: http://localhost:8001")
    print("  - Cleaner Worker: http://localhost:8002")
    print("  - Paraphraser Worker: http://localhost:8003")
    print("  - Fact-Adder Worker: http://localhost:8004")
    
    print("\nServices are running. Press Ctrl+C to stop all services.")
    
    try:
        # Keep services running
        while True:
            time.sleep(1)
            
            # Check if any process has died
            for i, (process, name) in enumerate(processes):
                if process.poll() is not None:
                    print(f"✗ Service {name} has stopped unexpectedly")
                    return False
                    
    except KeyboardInterrupt:
        print("\nStopping all services...")
        for process, name in processes:
            print(f"Stopping {name}...")
            process.terminate()
            process.wait()
        print("All services stopped.")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)