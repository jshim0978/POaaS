"""
Run artifact persistence for POaaS.

Saves artifacts under runs/{date}/{run_id}/{node}/ structure
as specified in Appendix F of the paper.

Usage:
    from poaas.common.artifacts import save_run_artifact, get_run_path
    
    # Save an artifact
    save_run_artifact(run_id="abc123", node="orchestrator", data={"prompt": "..."})
    
    # Get path for a run
    path = get_run_path(run_id="abc123")
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib


# Base directory for runs
RUNS_DIR = Path(os.getenv("POAAS_RUNS_DIR", "runs"))


def generate_run_id() -> str:
    """Generate a unique run ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:8]
    return f"{timestamp}_{unique}"


def get_run_path(run_id: str, node: Optional[str] = None) -> Path:
    """
    Get the path for a run's artifacts.
    
    Args:
        run_id: The run identifier
        node: Optional node name (orchestrator, cleaner, etc.)
        
    Returns:
        Path to the run directory or node subdirectory
    """
    date = datetime.now().strftime("%Y-%m-%d")
    
    if node:
        return RUNS_DIR / date / run_id / node
    return RUNS_DIR / date / run_id


def save_run_artifact(
    run_id: str,
    node: str,
    data: Dict[str, Any],
    artifact_name: str = "artifact.json"
) -> Path:
    """
    Save an artifact for a run.
    
    Args:
        run_id: The run identifier
        node: Node name (orchestrator, cleaner, paraphraser, fact_adder)
        data: Data to save
        artifact_name: Name of the artifact file
        
    Returns:
        Path to the saved artifact
    """
    # Create directory structure
    artifact_dir = get_run_path(run_id, node)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    data_with_meta = {
        "run_id": run_id,
        "node": node,
        "timestamp": datetime.now().isoformat(),
        "data": data
    }
    
    # Save artifact
    artifact_path = artifact_dir / artifact_name
    with open(artifact_path, 'w') as f:
        json.dump(data_with_meta, f, indent=2, default=str)
    
    return artifact_path


def save_input_artifact(run_id: str, prompt: str, config: Dict[str, Any]) -> Path:
    """Save the input prompt and configuration."""
    return save_run_artifact(
        run_id=run_id,
        node="input",
        data={
            "prompt": prompt,
            "config": config,
            "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()
        },
        artifact_name="input.json"
    )


def save_output_artifact(
    run_id: str,
    node: str,
    input_text: str,
    output_text: str,
    latency_ms: float,
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """Save an output artifact from a node."""
    data = {
        "input": input_text,
        "output": output_text,
        "latency_ms": latency_ms,
        "input_hash": hashlib.md5(input_text.encode()).hexdigest(),
        "output_hash": hashlib.md5(output_text.encode()).hexdigest()
    }
    
    if metadata:
        data["metadata"] = metadata
    
    return save_run_artifact(
        run_id=run_id,
        node=node,
        data=data,
        artifact_name="output.json"
    )


def save_final_artifact(
    run_id: str,
    original_prompt: str,
    final_prompt: str,
    workers_used: list,
    skipped: bool,
    latency_ms: float,
    ablation_mode: str = "full"
) -> Path:
    """Save the final result artifact."""
    return save_run_artifact(
        run_id=run_id,
        node="final",
        data={
            "original_prompt": original_prompt,
            "final_prompt": final_prompt,
            "workers_used": workers_used,
            "skipped": skipped,
            "total_latency_ms": latency_ms,
            "ablation_mode": ablation_mode,
            "drift": compute_drift_ratio(original_prompt, final_prompt),
            "length_ratio": len(final_prompt) / max(len(original_prompt), 1)
        },
        artifact_name="result.json"
    )


def compute_drift_ratio(original: str, modified: str) -> float:
    """Compute character-level drift ratio."""
    if not original:
        return 0.0
    
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, original.lower(), modified.lower()).ratio()
    return 1.0 - similarity


def load_run_artifact(run_id: str, node: str, artifact_name: str = "artifact.json") -> Optional[Dict[str, Any]]:
    """
    Load an artifact from a run.
    
    Args:
        run_id: The run identifier
        node: Node name
        artifact_name: Name of the artifact file
        
    Returns:
        Artifact data or None if not found
    """
    # Try today's date first
    date = datetime.now().strftime("%Y-%m-%d")
    artifact_path = RUNS_DIR / date / run_id / node / artifact_name
    
    if artifact_path.exists():
        with open(artifact_path, 'r') as f:
            return json.load(f)
    
    # Search in all dates
    for date_dir in RUNS_DIR.glob("*"):
        if date_dir.is_dir():
            artifact_path = date_dir / run_id / node / artifact_name
            if artifact_path.exists():
                with open(artifact_path, 'r') as f:
                    return json.load(f)
    
    return None


def list_runs(date: Optional[str] = None, limit: int = 100) -> list:
    """
    List runs, optionally filtered by date.
    
    Args:
        date: Date string (YYYY-MM-DD) or None for all dates
        limit: Maximum number of runs to return
        
    Returns:
        List of run IDs
    """
    runs = []
    
    if date:
        date_dir = RUNS_DIR / date
        if date_dir.exists():
            for run_dir in sorted(date_dir.iterdir(), reverse=True):
                if run_dir.is_dir():
                    runs.append(run_dir.name)
                    if len(runs) >= limit:
                        break
    else:
        for date_dir in sorted(RUNS_DIR.glob("*"), reverse=True):
            if date_dir.is_dir():
                for run_dir in sorted(date_dir.iterdir(), reverse=True):
                    if run_dir.is_dir():
                        runs.append(run_dir.name)
                        if len(runs) >= limit:
                            break
                if len(runs) >= limit:
                    break
    
    return runs


def get_run_summary(run_id: str) -> Optional[Dict[str, Any]]:
    """Get a summary of a run including all node outputs."""
    final = load_run_artifact(run_id, "final", "result.json")
    
    if not final:
        return None
    
    summary = {
        "run_id": run_id,
        "final_result": final.get("data", {}),
        "nodes": {}
    }
    
    # Load each node's output
    for node in ["input", "cleaner", "paraphraser", "fact_adder"]:
        artifact = load_run_artifact(run_id, node)
        if artifact:
            summary["nodes"][node] = artifact.get("data", {})
    
    return summary


if __name__ == "__main__":
    # Test artifact saving
    run_id = generate_run_id()
    print(f"Generated run ID: {run_id}")
    
    # Save test artifacts
    input_path = save_input_artifact(
        run_id=run_id,
        prompt="What is the captial of France?",
        config={"ablation": "full"}
    )
    print(f"Saved input artifact: {input_path}")
    
    output_path = save_output_artifact(
        run_id=run_id,
        node="cleaner",
        input_text="What is the captial of France?",
        output_text="What is the capital of France?",
        latency_ms=125.5
    )
    print(f"Saved cleaner output: {output_path}")
    
    final_path = save_final_artifact(
        run_id=run_id,
        original_prompt="What is the captial of France?",
        final_prompt="What is the capital of France?",
        workers_used=["cleaner"],
        skipped=False,
        latency_ms=150.0
    )
    print(f"Saved final result: {final_path}")
    
    # Load and display
    summary = get_run_summary(run_id)
    print(f"\nRun summary: {json.dumps(summary, indent=2)}")

