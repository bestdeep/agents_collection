"""
Run TextCraft Agent Evaluation

This script runs the TextCraft agent on a set of tasks and evaluates its performance.
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from textcraft_agent import TextCraftAgent, TextCraftAgentConfig
import requests


def create_env(env_url: str) -> int:
    """Create a new environment instance."""
    response = requests.post(f"{env_url}/create", json={})
    result = response.json()
    if "error" in result:
        raise RuntimeError(f"Failed to create environment: {result['error']}")
    return result["id"]


def close_env(env_url: str, env_id: int):
    """Close an environment instance."""
    try:
        requests.post(f"{env_url}/close", json={"id": env_id})
    except Exception as e:
        print(f"Warning: Failed to close environment {env_id}: {e}")


def run_evaluation(config_path: str, verbose: bool = True):
    """
    Run evaluation of the TextCraft agent.
    
    Args:
        config_path: Path to the configuration JSON file
        verbose: Whether to print detailed progress
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract configuration sections
    agent_config = TextCraftAgentConfig(**config["agent"])
    env_config = config["environment"]
    eval_config = config["evaluation"]
    
    # Create output directory
    output_dir = Path(eval_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize agent
    agent = TextCraftAgent(agent_config)
    
    # Run evaluation
    num_tasks = eval_config["num_tasks"]
    start_idx = eval_config["start_idx"]
    env_url = env_config["env_server"]
    
    print(f"\n{'='*70}")
    print(f"TextCraft Agent Evaluation")
    print(f"{'='*70}")
    print(f"Model: {agent_config.model}")
    print(f"Environment Server: {env_url}")
    print(f"Tasks: {num_tasks} (starting from index {start_idx})")
    print(f"Output Directory: {run_dir}")
    print(f"{'='*70}\n")
    
    # Check if environment server is running
    try:
        response = requests.get(f"{env_url}/", timeout=5)
        print(f"Environment server status: {response.text}\n")
    except Exception as e:
        print(f"Warning: Could not connect to environment server at {env_url}")
        print(f"Error: {e}")
        print("Please make sure the TextCraft server is running.")
        print("Start it with: textcraft --host 0.0.0.0 --port 36001\n")
        return
    
    results = []
    success_count = 0
    
    for i in range(num_tasks):
        task_idx = start_idx + i
        print(f"\n{'#'*70}")
        print(f"Task {i+1}/{num_tasks} (Data Index: {task_idx})")
        print(f"{'#'*70}")
        
        env_id = None
        try:
            # Create environment
            env_id = create_env(env_url)
            
            # Run episode
            result = agent.run_episode(env_url, env_id, task_idx, verbose=verbose)
            
            # Save conversation if requested
            if eval_config.get("save_conversations", True):
                conv_file = run_dir / f"task_{task_idx}_conversation.json"
                agent.save_conversation(str(conv_file))
            
            # Record result
            result["task_idx"] = task_idx
            results.append(result)
            
            if result["success"]:
                success_count += 1
            
            # Print summary
            print(f"\nTask {task_idx} Summary:")
            print(f"  Success: {result['success']}")
            print(f"  Reward: {result['reward']}")
            print(f"  Steps: {result['steps']}")
            print(f"  Success Rate So Far: {success_count}/{i+1} ({100*success_count/(i+1):.1f}%)")
            
        except Exception as e:
            print(f"\nError running task {task_idx}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "task_idx": task_idx,
                "success": False,
                "error": str(e)
            })
        
        finally:
            # Clean up environment
            if env_id is not None:
                close_env(env_url, env_id)
    
    # Save results summary
    summary = {
        "timestamp": timestamp,
        "config": config,
        "total_tasks": num_tasks,
        "successful_tasks": success_count,
        "success_rate": success_count / num_tasks if num_tasks > 0 else 0,
        "results": results
    }
    
    summary_file = run_dir / "evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"Evaluation Complete!")
    print(f"{'='*70}")
    print(f"Total Tasks: {num_tasks}")
    print(f"Successful: {success_count}")
    print(f"Success Rate: {100*success_count/num_tasks:.1f}%")
    print(f"\nResults saved to: {run_dir}")
    print(f"Summary file: {summary_file}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Run TextCraft Agent Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="config_example.json",
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        help="Override number of tasks to run"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        help="Override starting task index"
    )
    
    args = parser.parse_args()
    
    # Override config if command line arguments are provided
    if args.num_tasks is not None or args.start_idx is not None:
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        if args.num_tasks is not None:
            config["evaluation"]["num_tasks"] = args.num_tasks
        if args.start_idx is not None:
            config["evaluation"]["start_idx"] = args.start_idx
        
        # Save temporary config
        temp_config = "temp_config.json"
        with open(temp_config, 'w') as f:
            json.dump(config, f, indent=2)
        
        run_evaluation(temp_config, verbose=args.verbose)
        
        # Clean up
        os.remove(temp_config)
    else:
        run_evaluation(args.config, verbose=args.verbose)


if __name__ == "__main__":
    main()
