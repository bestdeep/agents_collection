"""
Run TextCraft Agent Evaluation

This script runs the TextCraft agent on a set of tasks and evaluates its performance.
"""

import os
import json
import argparse
import sys
from datetime import datetime
from pathlib import Path
from textcraft_agent import TextCraftAgent, TextCraftAgentConfig

# Import environment client from agentenv
try:
    from agentenv.controller import BaseEnvClient
    from agentenv.envs.textcraft import TextCraftEnvClient
except ImportError as e:
    print("Error: agentenv package or its dependencies not found.")
    print(f"Import error: {e}")
    print("\nPlease install agentenv and its dependencies:")
    print("  cd ../../agentenv")
    print("  pip install -e .")
    print("\nIf you see 'No module named httpx' or similar, install missing dependencies:")
    print("  pip install httpx requests tqdm")
    sys.exit(1)


def load_task_ids(file_path: str) -> list:
    """
    Load task IDs from a file.
    
    Args:
        file_path: Path to file containing task IDs (JSON array)
        
    Returns:
        List of task IDs as integers
    """
    with open(file_path, 'r') as f:
        task_ids = json.load(f)
    return task_ids


def run_evaluation(config_path: str, verbose: bool = True, task_list_file: str = None):
    """
    Run evaluation of the TextCraft agent.
    
    Args:
        config_path: Path to the configuration JSON file
        verbose: Whether to print detailed progress
        task_list_file: Optional path to file with task IDs
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract configuration sections
    agent_config = TextCraftAgentConfig(**config["agent"])
    env_config = config["environment"]
    eval_config = config["evaluation"]
    
    # Load task IDs
    if task_list_file:
        if not os.path.exists(task_list_file):
            print(f"Error: Task list file not found: {task_list_file}")
            sys.exit(1)
        task_ids = load_task_ids(task_list_file)
        if not task_ids:
            print(f"Error: No valid task IDs found in {task_list_file}")
            sys.exit(1)
        print(f"Loaded {len(task_ids)} task IDs from {task_list_file}")
    else:
        # Generate task IDs from config
        num_tasks = eval_config["num_tasks"]
        start_idx = eval_config["start_idx"]
        task_ids = list(range(start_idx, start_idx + num_tasks))
    
    # Create output directory
    output_dir = Path(eval_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize agent
    agent = TextCraftAgent(agent_config)
    
    # Initialize environment client
    env_client = TextCraftEnvClient(
        env_server_base=env_config["env_server"],
        data_len=env_config.get("data_len", 200),
        timeout=env_config.get("timeout", 300)
    )
    
    print(f"\n{'='*70}")
    print(f"TextCraft Agent Evaluation")
    print(f"{'='*70}")
    print(f"Model: {agent_config.model}")
    print(f"Environment Server: {env_config['env_server']}")
    print(f"Tasks to evaluate: {len(task_ids)}")
    if task_list_file:
        print(f"Task list file: {task_list_file}")
    else:
        print(f"Task range: {eval_config['start_idx']} to {eval_config['start_idx'] + eval_config['num_tasks'] - 1}")
    print(f"Output Directory: {run_dir}")
    print(f"{'='*70}\n")
    
    # Create environment instance
    try:
        env_idx = env_client.create()
        print(f"Environment created with ID: {env_idx}\n")
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("Make sure the TextCraft server is running:")
        print("  textcraft --host 0.0.0.0 --port 36001")
        sys.exit(1)
    
    results = []
    success_count = 0
    total_reward = 0.0
    
    try:
        for i, task_idx in enumerate(task_ids):
            
            if verbose:
                print(f"\n{'#'*70}")
                print(f"Task {i+1}/{len(task_ids)} (Data Index: {task_idx})")
                print(f"{'#'*70}")
            
            try:
                # Reset environment and conversation
                agent.reset_conversation()
                reset_response = env_client.reset(env_idx, task_idx)
                
                observation = reset_response["observation"]
                done = reset_response.get("done", False)
                reward = reset_response.get("reward", 0)
                
                if verbose:
                    print(f"\nInitial Observation:\n{observation}\n")
                
                step_count = 0
                episode_history = []
                
                # Run episode
                while not done and step_count < agent_config.max_rounds:
                    try:
                        is_first_step = (step_count == 0)
                        full_response, action = agent.generate_action(observation, include_goal=is_first_step)
                        
                        if verbose:
                            print(f"\n--- Step {step_count + 1} ---")
                            print(f"Agent Response:\n{full_response}")
                            print(f"\nExecuting Action: {action}")
                        
                        # Take step in environment
                        step_output = env_client.step(env_idx, action)
                        
                        observation = step_output.state
                        reward = step_output.reward
                        done = step_output.done
                        
                        if verbose:
                            print(f"Observation: {observation}")
                            print(f"Reward: {reward}, Done: {done}")
                        
                        # Record step
                        episode_history.append({
                            "step": step_count,
                            "thought_and_action": full_response,
                            "action": action,
                            "observation": observation,
                            "reward": reward,
                            "done": done
                        })
                        
                        step_count += 1
                        
                    except Exception as e:
                        print(f"Error during step {step_count}: {e}")
                        import traceback
                        traceback.print_exc()
                        break
                
                success = reward >= 1.0
                
                # Record result
                result = {
                    "task_idx": task_idx,
                    "success": success,
                    "reward": reward,
                    "steps": step_count,
                    "done": done
                }
                results.append(result)
                
                if success:
                    success_count += 1
                total_reward += reward
                
                # Save conversation if requested
                if eval_config.get("save_conversations", True):
                    conv_file = run_dir / f"task_{task_idx}_conversation.json"
                    with open(conv_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "task_idx": task_idx,
                            "result": result,
                            "conversation": agent.conversation_history,
                            "history": episode_history
                        }, f, indent=2, ensure_ascii=False)
                
                # Print summary
                if verbose:
                    print(f"\n{'='*70}")
                    print(f"Task {task_idx} Complete!")
                    print(f"Success: {success}, Reward: {reward}, Steps: {step_count}")
                    print(f"Success Rate So Far: {success_count}/{i+1} ({100*success_count/(i+1):.1f}%)")
                    print(f"{'='*70}")
                else:
                    print(f"Task {task_idx}: {'✓' if success else '✗'} (Steps: {step_count})")
                
            except Exception as e:
                print(f"\nError running task {task_idx}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "task_idx": task_idx,
                    "success": False,
                    "reward": 0,
                    "steps": 0,
                    "error": str(e)
                })
    
    finally:
        # Close environment
        try:
            env_client.close(env_idx)
            print("\nEnvironment closed successfully")
        except Exception as e:
            print(f"\nWarning: Error closing environment: {e}")
    
    # Save results summary
    summary = {
        "timestamp": timestamp,
        "config": config,
        "total_tasks": len(task_ids),
        "task_ids": task_ids,
        "task_list_file": task_list_file if task_list_file else None,
        "successful_tasks": success_count,
        "success_rate": success_count / len(task_ids) if task_ids else 0,
        "average_reward": total_reward / len(task_ids) if task_ids else 0,
        "results": results
    }
    
    summary_file = run_dir / "evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"Evaluation Complete!")
    print(f"{'='*70}")
    print(f"Total Tasks: {len(task_ids)}")
    print(f"Successful: {success_count}")
    print(f"Success Rate: {100*success_count/len(task_ids):.1f}%")
    print(f"Average Reward: {total_reward/len(task_ids):.3f}")
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
        "--task-list",
        type=str,
        default=None,
        help="Path to file containing list of task IDs (JSON array)"
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
        
        run_evaluation(temp_config, verbose=args.verbose, task_list_file=args.task_list)
        
        # Clean up
        os.remove(temp_config)
    else:
        run_evaluation(args.config, verbose=args.verbose, task_list_file=args.task_list)


if __name__ == "__main__":
    main()
