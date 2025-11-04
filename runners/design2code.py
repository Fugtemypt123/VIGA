#!/usr/bin/env python3
"""
Design2Code Runner for AgenticVerifier
Scans the Design2Code dataset and runs the dual-agent system per test case.

Each test case consists of a pair:
- HTML: data/design2code/testset_final/<id>.html
- PNG:  data/design2code/testset_final/<id>.png
"""
import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import get_model_info


def _collect_test_pairs(dataset_dir: str) -> List[Tuple[str, str, str]]:
    """
    Collect (case_id, html_path, png_path) tuples by intersecting ids from html and png files.
    """
    base = Path(dataset_dir)
    if not base.exists():
        print(f"Error: Design2Code dataset path does not exist: {dataset_dir}")
        return []

    html_files = {p.stem: str(p) for p in base.glob("*.html")}
    png_files = {p.stem: str(p) for p in base.glob("*.png")}

    common_ids = sorted(set(html_files.keys()) & set(png_files.keys()), key=lambda x: int(x) if x.isdigit() else x)
    pairs: List[Tuple[str, str, str]] = []
    for cid in common_ids:
        pairs.append((cid, html_files[cid], png_files[cid]))
    return pairs


def load_design2code_dataset(dataset_dir: str, case_id: Optional[str] = None) -> List[Dict]:
    """
    Build task list from dataset directory.

    Args:
        dataset_dir: Directory containing *.html and *.png pairs
        case_id: If provided, only run the specified id

    Returns:
        List of task configurations
    """
    tasks: List[Dict] = []
    pairs = _collect_test_pairs(dataset_dir)

    if case_id is not None:
        pairs = [p for p in pairs if p[0] == case_id]

    if not pairs:
        print(f"Warning: No valid html/png pairs found in {dataset_dir}")
        return tasks

    for cid, html_path, png_path in pairs:
        init_code_path = html_path.replace("Design2Code-HARD", "initialize")
        tasks.append({
            "case_id": cid,
            "init_code_path": init_code_path,
            "target_image_path": png_path,
        })
        print(f"Found case: id={cid}, html_path={init_code_path}, png_path={png_path}")

    return tasks


def run_design2code_task(task_config: Dict, args) -> tuple:
    """
    Run a single Design2Code test case using main.py

    Returns:
        Tuple of (case_id, success: bool, error_message: str)
    """
    case_id = task_config["case_id"]
    print(f"\n{'='*60}")
    print(f"Running Design2Code case: {case_id}")
    print(f"{'='*60}")

    # Prepare output directory for this case
    output_base = Path(args.output_dir) / case_id
    output_base.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "main.py",
        "--mode", "design2code",
        "--model", args.model,
        "--api-key", get_model_info(args.model)["api_key"],
        "--api-base-url", get_model_info(args.model)["base_url"],
        "--max-rounds", str(args.max_rounds),
        "--memory-length", str(args.memory_length),
        "--task-name", "design2code",
        "--init-code-path", str(task_config["init_code_path"]),
        "--init-image-path", str(task_config["target_image_path"]),  # not used but required by main
        "--target-image-path", str(task_config["target_image_path"]),
        "--output-dir", str(output_base),
        # Tool servers
        "--generator-tools", args.generator_tools,
        "--verifier-tools", args.verifier_tools,
        "--browser-command", args.browser_command,
        "--clear-memory"
    ]

    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd)  # no timeout
        print(f"Case completed successfully: {case_id}")
        return (case_id, True, "")
    except subprocess.CalledProcessError as e:
        error_msg = f"Case failed: {case_id}, Error: {e}"
        print(error_msg)
        return (case_id, False, str(e))
    except subprocess.TimeoutExpired:
        error_msg = f"Case timed out: {case_id}"
        print(error_msg)
        return (case_id, False, "Timeout")
    except Exception as e:
        error_msg = f"Case failed with exception: {case_id}, Error: {e}"
        print(error_msg)
        return (case_id, False, str(e))


def run_tasks_parallel(tasks: List[Dict], args, max_workers: int = 8) -> tuple:
    successful = 0
    failed = 0
    failed_details = []

    print(f"\nStarting parallel execution with max {max_workers} workers...")
    print(f"Total cases: {len(tasks)}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(run_design2code_task, task_config, args): task_config
            for task_config in tasks
        }

        for future in as_completed(future_to_task):
            task_config = future_to_task[future]
            try:
                case_id, ok, error_msg = future.result()
                if ok:
                    successful += 1
                    print(f"✅ {case_id} completed successfully")
                else:
                    failed += 1
                    failed_details.append({
                        "case_id": case_id,
                        "error": error_msg
                    })
                    print(f"❌ {case_id} failed: {error_msg}")
            except Exception as e:
                failed += 1
                case_id = task_config["case_id"]
                failed_details.append({
                    "case_id": case_id,
                    "error": str(e)
                })
                print(f"❌ {case_id} failed with exception: {e}")

    return successful, failed, failed_details


def main():
    parser = argparse.ArgumentParser(description="Design2Code Runner for AgenticVerifier")

    parser.add_argument("--dataset-path", default="data/design2code/Design2Code-HARD", help="Path to Design2Code dataset directory")
    parser.add_argument("--output-dir", default=f"output/design2code/{time.strftime('%Y%m%d_%H%M%S')}", help="Output directory for results")

    # Selection
    parser.add_argument("--case-id", default=None, help="Specific case id to run (e.g., '2')")

    # Main.py parameters
    parser.add_argument("--max-rounds", type=int, default=10, help="Maximum number of interaction rounds")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI vision model to use")
    parser.add_argument("--memory-length", type=int, default=12, help="Memory length")

    # Agent server paths
    parser.add_argument("--generator-script", default="agents/generator.py", help="Generator MCP script path")
    parser.add_argument("--verifier-script", default="agents/verifier.py", help="Verifier MCP script path")

    # Tool server scripts (comma-separated)
    parser.add_argument("--generator-tools", default="tools/exec_html.py,tools/generator_base.py", help="Comma-separated list of generator tool server scripts")
    parser.add_argument("--browser-command", default="google-chrome", help="Browser command for HTML screenshots")

    # Verifier tools
    parser.add_argument("--verifier-tools", default="tools/verifier_base.py", help="Comma-separated list of verifier tool server scripts")

    # Parallel execution parameters
    parser.add_argument("--max-workers", type=int, default=8, help="Maximum number of parallel workers")
    parser.add_argument("--sequential", action="store_true", help="Run tasks sequentially instead of in parallel")

    args = parser.parse_args()

    # Create top-level output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save args
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # Load tasks
    print(f"Loading Design2Code dataset from: {args.dataset_path}")
    tasks = load_design2code_dataset(args.dataset_path, args.case_id)

    if not tasks:
        print("No valid cases found in dataset!")
        sys.exit(1)

    print(f"Found {len(tasks)} cases")

    start_time = time.time()

    if args.sequential:
        print("\nRunning cases sequentially...")
        successful = 0
        failed = 0
        failed_details = []

        for i, task_config in enumerate(tasks, 1):
            print(f"\nCase {i}/{len(tasks)}")
            case_id, ok, error_msg = run_design2code_task(task_config, args)
            if ok:
                successful += 1
            else:
                failed += 1
                failed_details.append({
                    "case_id": case_id,
                    "error": error_msg
                })
    else:
        successful, failed, failed_details = run_tasks_parallel(tasks, args, max_workers=args.max_workers)

    end_time = time.time()
    exec_secs = end_time - start_time

    # Summary
    print(f"\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total cases: {len(tasks)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Execution time: {exec_secs:.2f} seconds")
    print(f"Output directory: {args.output_dir}")

    results = {
        "total_cases": len(tasks),
        "successful_cases": successful,
        "failed_cases": failed,
        "execution_time_seconds": exec_secs,
        "failed_case_details": failed_details,
        "execution_mode": "sequential" if args.sequential else f"parallel_{args.max_workers}_workers"
    }

    with open(os.path.join(args.output_dir, "execution_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    if successful > 0:
        print(f"\nResults saved to: {args.output_dir}")
    if failed > 0:
        print(f"\nFailed cases details saved to: {os.path.join(args.output_dir, 'execution_results.json')}")


if __name__ == "__main__":
    main()


