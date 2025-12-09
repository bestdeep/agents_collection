"""
Example script to run the BabyAI LLM Agent with the BabyAI environment.

This script demonstrates how to:
1. Set up the BabyAI environment client
2. Initialize the BabyAI agent
3. Run evaluation on multiple tasks
4. Save results

Usage:
    python run_babyai_agent.py --env_server http://localhost:36001 --api_key YOUR_API_KEY
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
import sys

# Import the agent
from babyai_agent import BabyAIAgent, BabyAIAgentConfig

# Import environment client from agentenv
try:
    from agentenv.envs.babyai import BabyAIEnvClient
except ImportError as e:
    print("Error: agentenv package or its dependencies not found.")
    print(f"Import error: {e}")
    print("\nPlease install agentenv and its dependencies:")
    print("  cd ../../agentenv")
    print("  pip install -e .")
    print("\nIf you see 'No module named httpx' or similar, install missing dependencies:")
    print("  pip install httpx requests tqdm")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run BabyAI Agent evaluation")
    
    # Environment settings
    parser.add_argument("--env_server", type=str, default="http://localhost:36001",
                        help="BabyAI environment server base URL")
    parser.add_argument("--data_len", type=int, default=200,
                        help="Number of tasks in the dataset")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout for environment requests in seconds")
    
    # API settings
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1",
                        help="API base URL")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        help="Model name to use")
    
    # Agent settings
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation")
    parser.add_argument("--max_rounds", type=int, default=20,
                        help="Maximum rounds per episode")
    
    # Evaluation settings
    parser.add_argument("--num_tasks", type=int, default=10,
                        help="Number of tasks to evaluate")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting task index")
    parser.add_argument("--output_dir", type=str, default="./babyai_results",
                        help="Directory to save results")
    parser.add_argument("--save_conversations", action="store_true",
                        help="Save conversation histories")
    
    return parser.parse_args()


def main():
    """Main evaluation loop."""
    args = parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: API key not provided. Set --api_key or OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_conversations:
        (output_dir / "conversations").mkdir(exist_ok=True)
    
    print("=" * 60)
    print("BabyAI Agent Evaluation")
    print("=" * 60)
    print(f"Environment server: {args.env_server}")
    print(f"Model: {args.model}")
    print(f"Max rounds: {args.max_rounds}")
    print(f"Tasks to evaluate: {args.num_tasks}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Initialize agent
    agent_config = BabyAIAgentConfig(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_rounds=args.max_rounds
    )
    agent = BabyAIAgent(agent_config)
    
    # Initialize environment client
    env_client = BabyAIEnvClient(
        env_server_base=args.env_server,
        data_len=args.data_len,
        timeout=args.timeout
    )
    
    # Create environment instance
    try:
        env_idx = env_client.create()
        print(f"Environment created with ID: {env_idx}")
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("Make sure the BabyAI server is running:")
        print("  babyai --host 0.0.0.0 --port 36001")
        sys.exit(1)
    
    # Run evaluation
    results = []
    total_success = 0
    total_reward = 0.0
    
    try:
        for i in tqdm(range(args.num_tasks), desc="Evaluating tasks"):
            data_idx = args.start_idx + i
            
            try:
                # Run episode
                result = agent.run_episode(env_client, env_idx, data_idx)
                
                # Update statistics
                total_success += int(result["success"])
                total_reward += result["reward"]
                
                # Save result
                result_summary = {
                    "task_idx": data_idx,
                    "success": result["success"],
                    "reward": result["reward"],
                    "steps": result["steps"],
                    "done": result["done"]
                }
                results.append(result_summary)
                
                # Save conversation if requested
                if args.save_conversations:
                    conv_path = output_dir / "conversations" / f"task_{data_idx}.json"
                    with open(conv_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            "task_idx": data_idx,
                            "result": result_summary,
                            "conversation": result["conversation"],
                            "history": result["history"]
                        }, f, indent=2, ensure_ascii=False)
                
                # Print progress
                if (i + 1) % 5 == 0:
                    current_success_rate = total_success / (i + 1)
                    current_avg_reward = total_reward / (i + 1)
                    print(f"\nProgress: {i+1}/{args.num_tasks}")
                    print(f"  Success rate: {current_success_rate:.2%}")
                    print(f"  Average reward: {current_avg_reward:.3f}")
                
            except Exception as e:
                print(f"\nError on task {data_idx}: {e}")
                results.append({
                    "task_idx": data_idx,
                    "success": False,
                    "reward": 0.0,
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
        "config": {
            "model": args.model,
            "max_rounds": args.max_rounds,
            "num_tasks": args.num_tasks,
            "start_idx": args.start_idx
        },
        "results": {
            "total_tasks": len(results),
            "total_success": total_success,
            "success_rate": total_success / len(results) if results else 0,
            "average_reward": total_reward / len(results) if results else 0,
            "tasks": results
        }
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print final results
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Total tasks: {len(results)}")
    print(f"Successful tasks: {total_success}")
    print(f"Success rate: {summary['results']['success_rate']:.2%}")
    print(f"Average reward: {summary['results']['average_reward']:.3f}")
    print(f"\nResults saved to: {summary_path}")
    if args.save_conversations:
        print(f"Conversations saved to: {output_dir / 'conversations'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
