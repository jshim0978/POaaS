#!/usr/bin/env python3
"""
Aggregate POaaS experimental results into paper tables.

Reads raw result JSONs and generates:
- Summary CSV files
- LaTeX tables (matching paper format)
- Comparison plots

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --output-dir results/tables
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics


def load_result_files(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all result JSON files from the results directory."""
    results = []
    
    for json_file in results_dir.glob("real_eval_*_summary.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                data["_source_file"] = str(json_file)
                results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    
    return results


def extract_metrics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Extract metrics organized by benchmark, method, and metric.
    
    Returns:
        {benchmark: {method: {metric: value}}}
    """
    metrics = defaultdict(lambda: defaultdict(dict))
    
    for result in results:
        for benchmark, bench_data in result.items():
            if benchmark.startswith("_"):
                continue
            
            if isinstance(bench_data, dict):
                for method, method_data in bench_data.items():
                    if isinstance(method_data, dict):
                        # Extract primary metric
                        if benchmark in ["bbh", "gsm8k", "commonsenseqa"]:
                            metric_name = "accuracy"
                        elif benchmark == "halueval":
                            metric_name = "truthfulness"
                        elif benchmark == "hallulens":
                            metric_name = "consistency"
                        elif benchmark == "factscore":
                            metric_name = "fact_score"
                        else:
                            metric_name = "accuracy"
                        
                        if metric_name in method_data:
                            metrics[benchmark][method]["score"] = method_data[metric_name]
                        
                        if "avg_optimization_latency_ms" in method_data:
                            metrics[benchmark][method]["opt_latency"] = method_data["avg_optimization_latency_ms"]
                        
                        if "avg_inference_latency_ms" in method_data:
                            metrics[benchmark][method]["inf_latency"] = method_data["avg_inference_latency_ms"]
                        
                        if "num_samples" in method_data:
                            metrics[benchmark][method]["n_samples"] = method_data["num_samples"]
    
    return dict(metrics)


def generate_main_results_table(metrics: Dict) -> str:
    """Generate Table 3: Main Results (matching paper format)."""
    benchmarks = ["bbh", "gsm8k", "commonsenseqa", "halueval", "hallulens", "factscore"]
    methods = ["poaas", "evoprompt", "opro", "promptwizard", "apo"]
    
    # Header
    lines = [
        "=" * 80,
        "TABLE 3: Main Results (Clean Inputs) - Llama-3.2-3B-Instruct",
        "=" * 80,
        "",
        f"{'Method':<15} {'BBH':>8} {'GSM8K':>8} {'CSQA':>8} {'HaluEval':>10} {'HalluLens':>10} {'FActScore':>10}",
        "-" * 80
    ]
    
    for method in methods:
        row = f"{method:<15}"
        for bench in benchmarks:
            if bench in metrics and method in metrics[bench]:
                score = metrics[bench][method].get("score", 0.0) * 100
                row += f" {score:>8.1f}"
            else:
                row += f" {'N/A':>8}"
        lines.append(row)
    
    lines.append("-" * 80)
    return "\n".join(lines)


def generate_latency_table(metrics: Dict) -> str:
    """Generate Table 4: Latency Comparison."""
    benchmarks = ["bbh", "gsm8k", "commonsenseqa"]
    methods = ["poaas", "evoprompt", "opro", "promptwizard"]
    
    lines = [
        "=" * 60,
        "TABLE 4: Latency Comparison",
        "=" * 60,
        "",
        f"{'Method':<15} {'Avg Opt Latency (ms)':>25}",
        "-" * 60
    ]
    
    for method in methods:
        latencies = []
        for bench in benchmarks:
            if bench in metrics and method in metrics[bench]:
                lat = metrics[bench][method].get("opt_latency", 0)
                if lat > 0:
                    latencies.append(lat)
        
        if latencies:
            avg_lat = statistics.mean(latencies)
            lines.append(f"{method:<15} {avg_lat:>25.1f}")
        else:
            lines.append(f"{method:<15} {'N/A':>25}")
    
    lines.append("-" * 60)
    return "\n".join(lines)


def generate_latex_table(metrics: Dict) -> str:
    """Generate LaTeX table for paper."""
    benchmarks = ["bbh", "gsm8k", "commonsenseqa", "halueval", "hallulens", "factscore"]
    methods = ["poaas", "evoprompt", "opro", "promptwizard", "apo"]
    method_names = {
        "poaas": "POaaS",
        "evoprompt": "EvoPrompt",
        "opro": "OPRO",
        "promptwizard": "PromptWizard",
        "apo": "APO"
    }
    
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Main Results on Clean Inputs (Llama-3.2-3B-Instruct)}",
        "\\label{tab:main-results}",
        "\\begin{tabular}{l|ccc|ccc}",
        "\\toprule",
        "& \\multicolumn{3}{c|}{Reasoning} & \\multicolumn{3}{c}{Factuality} \\\\",
        "Method & BBH & GSM8K & CSQA & HaluEval & HalluLens & FActScore \\\\",
        "\\midrule"
    ]
    
    for method in methods:
        name = method_names.get(method, method)
        row_parts = [name]
        
        for bench in benchmarks:
            if bench in metrics and method in metrics[bench]:
                score = metrics[bench][method].get("score", 0.0) * 100
                # Bold best result (simplified - POaaS in this template)
                if method == "poaas":
                    row_parts.append(f"\\textbf{{{score:.1f}}}")
                else:
                    row_parts.append(f"{score:.1f}")
            else:
                row_parts.append("--")
        
        lines.append(" & ".join(row_parts) + " \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def generate_csv(metrics: Dict, output_path: Path):
    """Generate CSV file with all results."""
    with open(output_path, 'w') as f:
        # Header
        f.write("benchmark,method,score,opt_latency_ms,inf_latency_ms,n_samples\n")
        
        for benchmark in sorted(metrics.keys()):
            for method in sorted(metrics[benchmark].keys()):
                data = metrics[benchmark][method]
                score = data.get("score", 0.0)
                opt_lat = data.get("opt_latency", 0.0)
                inf_lat = data.get("inf_latency", 0.0)
                n_samples = data.get("n_samples", 0)
                
                f.write(f"{benchmark},{method},{score:.4f},{opt_lat:.2f},{inf_lat:.2f},{n_samples}\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate POaaS results")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Directory containing result JSON files")
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"),
                        help="Output directory for tables")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.results_dir}...")
    results = load_result_files(args.results_dir)
    print(f"Loaded {len(results)} result files")
    
    if not results:
        print("No results found. Run experiments first:")
        print("  python scripts/run_experiments.py --config test")
        return
    
    # Extract metrics
    metrics = extract_metrics(results)
    
    # Generate tables
    print("\n" + generate_main_results_table(metrics))
    print("\n" + generate_latency_table(metrics))
    
    # Save outputs
    csv_path = args.output_dir / "results.csv"
    generate_csv(metrics, csv_path)
    print(f"\nSaved CSV to {csv_path}")
    
    # Save LaTeX table
    latex_path = args.output_dir / "table3.tex"
    with open(latex_path, 'w') as f:
        f.write(generate_latex_table(metrics))
    print(f"Saved LaTeX table to {latex_path}")
    
    # Save text tables
    txt_path = args.output_dir / "summary.txt"
    with open(txt_path, 'w') as f:
        f.write(generate_main_results_table(metrics))
        f.write("\n\n")
        f.write(generate_latency_table(metrics))
    print(f"Saved text summary to {txt_path}")
    
    print("\nAggregation complete!")


if __name__ == "__main__":
    main()

