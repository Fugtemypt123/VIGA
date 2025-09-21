#!/usr/bin/env python3
"""
Script to adjust penalty scores in tournament results and output final scores.
Reads tournament_results.json, reduces penalty scores from 1.0 to 0.1, and outputs final statistics.
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List


def load_tournament_results(results_path: str) -> Dict:
    """
    Load tournament results from JSON file.
    
    Args:
        results_path: Path to tournament_results.json
        
    Returns:
        Dictionary containing tournament results
    """
    with open(results_path, 'r') as f:
        return json.load(f)


def get_task_specific_penalty(task_name: str) -> float:
    """
    Get task-specific penalty score based on task type.
    
    Args:
        task_name: Name of the task (e.g., 'blendshape1', 'material5')
        
    Returns:
        Penalty score for the task type
    """
    # Extract task type from task name
    task_type = ''.join([c for c in task_name if not c.isdigit()]).lower()
    
    # Define penalty scores by task type
    penalty_scores = {
        'blendshape': 0.1,
        'placement': 0.1,
        'geometry': 1,
        'lighting': 1,
        'material': 1
    }
    
    return penalty_scores.get(task_type, 1.0)


def adjust_penalty_scores(results: Dict) -> Dict:
    """
    Adjust penalty scores in tournament results based on task type.
    
    Args:
        results: Tournament results dictionary
        
    Returns:
        Adjusted results dictionary
    """
    adjusted_results = results.copy()
    
    for task in adjusted_results.get("tasks", []):
        if task.get("special_case") == "no_rounds":
            # Get task-specific penalty score
            task_name = task['task_name']
            penalty_score = get_task_specific_penalty(task_name)
            
            # Update penalty scores (pl penalty is 1/3 of clip penalty)
            pl_penalty_score = penalty_score / 3.0
            task["final_metrics"] = {
                "n_clip_render1": penalty_score,
                "n_clip_render2": penalty_score,
                "avg_n_clip": penalty_score,
                "pl_render1": pl_penalty_score,
                "pl_render2": pl_penalty_score,
                "avg_pl": pl_penalty_score
            }
            print(f"Adjusted penalty scores for {task_name}: {1.0} -> {penalty_score}")
    
    return adjusted_results


def calculate_final_statistics(results: Dict) -> Dict:
    """
    Calculate final statistics from adjusted results.
    
    Args:
        results: Tournament results dictionary
        
    Returns:
        Dictionary containing final statistics
    """
    tasks = results.get("tasks", [])
    successful_tasks = [t for t in tasks if "error" not in t]
    
    if not successful_tasks:
        return {"error": "No successful tasks found"}
    
    # Group by task type
    task_types = {}
    for task in successful_tasks:
        task_name = task["task_name"]
        task_type = ''.join([c for c in task_name if not c.isdigit()])  # Extract task type
        if task_type not in task_types:
            task_types[task_type] = []
        task_types[task_type].append(task)
    
    # Calculate statistics
    final_stats = {}
    for task_type, type_tasks in task_types.items():
        n_clip_values = [t["final_metrics"]["avg_n_clip"] for t in type_tasks]
        pl_values = [t["final_metrics"]["avg_pl"] for t in type_tasks]
        
        # Count special cases
        auto_wins = sum(1 for t in type_tasks if t.get("special_case") == "auto_win")
        penalties = sum(1 for t in type_tasks if t.get("special_case") == "no_rounds")
        tournaments = len(type_tasks) - auto_wins - penalties
        
        final_stats[task_type] = {
            "num_tasks": len(type_tasks),
            "tournaments_run": tournaments,
            "auto_wins": auto_wins,
            "penalty_scores": penalties,
            "avg_n_clip": sum(n_clip_values) / len(n_clip_values),
            "avg_pl": sum(pl_values) / len(pl_values),
            "best_n_clip": min(n_clip_values),
            "worst_n_clip": max(n_clip_values),
            "best_pl": min(pl_values),
            "worst_pl": max(pl_values)
        }
    
    return final_stats


def print_final_scores(stats: Dict):
    """
    Print final scores in a formatted way.
    
    Args:
        stats: Final statistics dictionary
    """
    if "error" in stats:
        print(f"Error: {stats['error']}")
        return
    
    print(f"\n{'='*60}")
    print("FINAL SCORES (with adjusted penalty)")
    print(f"{'='*60}")
    
    # Calculate overall averages
    all_n_clip = []
    all_pl = []
    total_tasks = 0
    total_tournaments = 0
    total_auto_wins = 0
    total_penalties = 0
    
    for task_type, type_stats in stats.items():
        all_n_clip.extend([type_stats["avg_n_clip"]] * type_stats["num_tasks"])
        all_pl.extend([type_stats["avg_pl"]] * type_stats["num_tasks"])
        total_tasks += type_stats["num_tasks"]
        total_tournaments += type_stats["tournaments_run"]
        total_auto_wins += type_stats["auto_wins"]
        total_penalties += type_stats["penalty_scores"]
    
    print(f"\nOVERALL SUMMARY:")
    print(f"  Total tasks: {total_tasks}")
    print(f"  Tournaments run: {total_tournaments}")
    print(f"  Auto-wins: {total_auto_wins}")
    print(f"  Penalty scores: {total_penalties}")
    print(f"  Overall avg n_clip: {sum(all_n_clip) / len(all_n_clip):.4f}")
    print(f"  Overall avg pl: {sum(all_pl) / len(all_pl):.4f}")
    
    print(f"\nDETAILED STATISTICS BY TASK TYPE:")
    for task_type, type_stats in stats.items():
        print(f"\n{task_type.upper()}:")
        print(f"  Tasks evaluated: {type_stats['num_tasks']}")
        print(f"  Tournaments run: {type_stats['tournaments_run']}")
        print(f"  Auto-wins: {type_stats['auto_wins']}")
        print(f"  Penalty scores: {type_stats['penalty_scores']}")
        print(f"  Average n_clip: {type_stats['avg_n_clip']}")
        print(f"  Average pl: {type_stats['avg_pl']}")
        print(f"  Best n_clip: {type_stats['best_n_clip']}")
        print(f"  Worst n_clip: {type_stats['worst_n_clip']}")
        print(f"  Best pl: {type_stats['best_pl']}")
        print(f"  Worst pl: {type_stats['worst_pl']}")


def save_adjusted_results(results: Dict, output_path: str):
    """
    Save adjusted results to a new file.
    
    Args:
        results: Adjusted results dictionary
        output_path: Path to save the adjusted results
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAdjusted results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Adjust penalty scores in tournament results based on task type")
    
    parser.add_argument("--results-file", required=True, 
                       help="Path to tournament_results.json file")
    parser.add_argument("--output-file", 
                       help="Path to save adjusted results (optional)")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading tournament results from: {args.results_file}")
    results = load_tournament_results(args.results_file)
    
    # Count original penalty tasks
    original_penalties = sum(1 for task in results.get("tasks", []) 
                           if task.get("special_case") == "no_rounds")
    print(f"Found {original_penalties} tasks with penalty scores")
    
    # Adjust penalty scores based on task type
    print("Adjusting penalty scores based on task type:")
    print("  blendshape, placement: 0.1")
    print("  geometry, lighting, material: 1.0")
    adjusted_results = adjust_penalty_scores(results)
    
    # Calculate final statistics
    print("Calculating final statistics...")
    final_stats = calculate_final_statistics(adjusted_results)
    
    # Print final scores
    print_final_scores(final_stats)
    
    # Save adjusted results if requested
    if args.output_file:
        save_adjusted_results(adjusted_results, args.output_file)
    
    # Save final statistics
    stats_output = args.results_file.replace("tournament_results.json", "final_scores.json")
    with open(stats_output, 'w') as f:
        json.dump(final_stats, f, indent=2)
    print(f"Final statistics saved to: {stats_output}")


if __name__ == "__main__":
    main()
