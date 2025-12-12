#!/usr/bin/env python3
"""
CLI for Affine Agent

Provides a command-line interface for running SAT, ABD, and DED tasks.
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path

from agent import AffineAgentConfig, create_agent
from env_integration import AffineEnvironmentAgent, evaluate_agent
from config_utils import load_config, create_agent_config_from_dict, get_env_configs
from dataset_generator import DatasetGenerator


async def cmd_evaluate(args):
    """Evaluate a single task from the dataset."""
    try:
        config_dict = load_config(args.config)
        agent_config = create_agent_config_from_dict(config_dict)
        agent_config.verbose = args.verbose
        env_configs = get_env_configs(config_dict)
    except FileNotFoundError:
        agent_config = AffineAgentConfig(
            api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
            model=args.model or "gpt-4o",
            base_url=args.base_url or os.getenv("BASE_URL") or "https://api.openai.com/v1",
            temperature=args.temperature or 0.7,
            verbose=args.verbose
        )
        env_configs = {}
    
    # Override with CLI args
    if args.api_key:
        agent_config.api_key = args.api_key
    if args.model:
        agent_config.model = args.model
    if args.base_url:
        agent_config.base_url = args.base_url
    if args.temperature is not None:
        agent_config.temperature = args.temperature
    
    env_agent = AffineEnvironmentAgent(agent_config, env_configs)
    
    print(f"\nEvaluating task {args.task_id} from {args.env.upper()} environment...")
    
    result = await env_agent.solve_and_evaluate(
        env=args.env,
        task_id=args.task_id,
        save_results=args.save,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*70)
    print("EVALUATION RESULT")
    print("="*70)
    print(f"Environment: {args.env.upper()}")
    print(f"Task ID: {result['task_id']}")
    print(f"\nPrompt:\n{result['challenge'].prompt[:200]}...")
    print(f"\n{'-'*70}")
    print(f"\nResponse:\n{result['response'][:500]}...")
    print(f"\n{'-'*70}")
    print(f"\nScore: {result['score']}")
    
    # Show extracted answer if available
    if result.get('extracted_answer'):
        print(f"\nExtracted Answer: {result['extracted_answer'][:200]}...")
    
    # Show if saved
    if args.save:
        print(f"\n✓ Results saved to: {args.output_dir}")
    
    print("="*70)


def cmd_solve(args):
    """Solve a single task with a custom prompt."""
    # Load config
    try:
        config_dict = load_config(args.config)
        agent_config = create_agent_config_from_dict(config_dict)
        agent_config.verbose = args.verbose
    except FileNotFoundError:
        agent_config = AffineAgentConfig(
            api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
            model=args.model or "gpt-4o",
            base_url=args.base_url or os.getenv("BASE_URL") or "https://api.openai.com/v1",
            temperature=args.temperature or 0.7,
            verbose=args.verbose
        )
    
    # Create agent
    agent = create_agent(
        api_key=agent_config.api_key,
        base_url=agent_config.base_url,
        model=agent_config.model,
        temperature=agent_config.temperature,
        verbose=args.verbose
    )
    
    # Solve
    print(f"\nSolving task in {args.env.upper()} environment...")
    print(f"Prompt: {args.prompt}\n")
    
    solution = agent.solve(args.prompt, env=args.env)
    
    print("="*70)
    print("SOLUTION")
    print("="*70)
    print(solution)
    print("="*70)
    
    # Extract answer if applicable
    if args.env == 'sat':
        answer = agent.extract_sat_assignment(solution)
        if answer:
            print(f"\nExtracted Assignment: {answer}")
    elif args.env == 'abd':
        answer = agent.extract_input(solution)
        if answer:
            print(f"\nExtracted Input:\n{answer}")
    elif args.env == 'ded':
        code = agent.extract_code(solution)
        print(f"\nExtracted Code:\n{code}")


def cmd_interactive(args):
    """Interactive mode for solving tasks."""
    # Load config
    try:
        config_dict = load_config(args.config)
        agent_config = create_agent_config_from_dict(config_dict)
    except FileNotFoundError:
        agent_config = AffineAgentConfig(
            api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
            model=args.model or "gpt-4o",
            base_url=args.base_url or os.getenv("BASE_URL") or "https://api.openai.com/v1"
        )
    
    # Override with CLI args
    if args.api_key:
        agent_config.api_key = args.api_key
    if args.model:
        agent_config.model = args.model
    if args.base_url:
        agent_config.base_url = args.base_url
    
    agent = create_agent(
        api_key=agent_config.api_key,
        base_url=agent_config.base_url,
        model=agent_config.model,
        temperature=args.temperature or agent_config.temperature,
        verbose=args.verbose
    )
    
    print("\n" + "="*70)
    print(f"Interactive Mode - {args.env.upper()} Environment")
    print("="*70)
    print("Type your prompts and press Enter. Type 'exit' or 'quit' to exit.")
    print("Type 'reset' to reset conversation history.")
    print("="*70 + "\n")
    
    while True:
        try:
            prompt = input(f"{args.env.upper()}> ").strip()
            
            if prompt.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
            
            if prompt.lower() == 'reset':
                agent.reset_conversation(args.env)
                print("✓ Conversation history reset.\n")
                continue
            
            if not prompt:
                continue
            
            print("\nGenerating solution...\n")
            solution = agent.solve(prompt, env=args.env)
            
            print("-"*70)
            print(solution)
            print("-"*70)
            
            # Show extracted answer if applicable
            if args.env == 'sat':
                answer = agent.extract_sat_assignment(solution)
                if answer:
                    print(f"\n✓ Assignment: {answer}")
            elif args.env == 'abd':
                answer = agent.extract_input(solution)
                if answer and len(answer) < len(solution) * 0.9:
                    print(f"\n✓ Extracted Input:\n{answer[:200]}{'...' if len(answer) > 200 else ''}")
            elif args.env == 'ded':
                code = agent.extract_code(solution)
                if code and len(code) < len(solution) * 0.9:
                    print(f"\n✓ Extracted Code:\n{code[:200]}{'...' if len(code) > 200 else ''}")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}\n")



async def run_benchmark(args):
    """Run benchmark on multiple tasks."""
    # Load config
    try:
        config_dict = load_config(args.config)
        agent_config = create_agent_config_from_dict(config_dict)
        agent_config.verbose = args.verbose
        env_configs = get_env_configs(config_dict)
    except FileNotFoundError:
        agent_config = AffineAgentConfig(
            api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
            model=args.model or "gpt-4o",
            base_url=args.base_url or os.getenv("BASE_URL") or "https://api.openai.com/v1",
            temperature=args.temperature or 0.7,
            verbose=args.verbose
        )
        env_configs = {}
    
    # Override with CLI args
    if args.api_key:
        agent_config.api_key = args.api_key
    if args.model:
        agent_config.model = args.model
    if args.base_url:
        agent_config.base_url = args.base_url
    if args.temperature is not None:
        agent_config.temperature = args.temperature
    
    # Run benchmark
    results = await evaluate_agent(
        env=args.env,
        num_tasks=args.num_tasks,
        task_ids=args.task_ids,
        agent_config=agent_config,
        save_results=args.save,
        output_dir=args.output_dir,
        save_interval=getattr(args, 'save_interval', 10),
        verbose=args.verbose
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Benchmark Summary")
    print(f"{'='*60}")
    print(f"Environment: {args.env}")
    print(f"Tasks evaluated: {results['num_tasks']}")
    print(f"Average score: {results.get('average_score', results.get('avg_score', 0)):.3f}")
    if 'success_rate' in results:
        print(f"Success rate: {results['success_rate']:.2%}")
    if 'max_score' in results:
        print(f"Max score: {results['max_score']:.3f}")
        print(f"Min score: {results['min_score']:.3f}")
    print(f"{'='*60}")
    
    return results


async def cmd_generate_dataset(args):
    """Generate training dataset from successful agent conversations."""
    from datetime import datetime
    
    # Load config
    try:
        config_dict = load_config(args.config)
        agent_config = create_agent_config_from_dict(config_dict)
        agent_config.verbose = args.verbose
        env_configs = get_env_configs(config_dict)
    except FileNotFoundError:
        agent_config = AffineAgentConfig(
            api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
            model=args.model or "gpt-4o",
            base_url=args.base_url or os.getenv("BASE_URL") or "https://api.openai.com/v1",
            temperature=args.temperature or 0.7,
            verbose=args.verbose
        )
        env_configs = {}
    
    # Override with CLI args
    if args.api_key:
        agent_config.api_key = args.api_key
    if args.model:
        agent_config.model = args.model
    if args.base_url:
        agent_config.base_url = args.base_url
    if args.temperature is not None:
        agent_config.temperature = args.temperature
    
    # Create generator
    generator = DatasetGenerator(agent_config, env_configs)
    
    # Generate task IDs
    task_ids = list(range(args.start_id, args.end_id + 1))
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{args.output_dir}/{args.env}_dataset_{args.start_id}_{args.end_id}_{timestamp}.json"
    
    # Generate dataset
    results = await generator.generate_dataset(
        env=args.env,
        task_ids=task_ids,
        output_file=output_file,
        verbose=args.verbose,
        save_interval=args.save_interval
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="CLI for Affine Agent (SAT, ABD, and DED tasks)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single ABD task
  python cli.py evaluate --env abd --task-id 0 --api-key YOUR_KEY
  
  # Run single DED task with verbose output
  python cli.py evaluate --env ded --task-id 5 --verbose
  
  # Run benchmark on 10 DED tasks
  python cli.py benchmark --env ded --num-tasks 10 --save
  
  # Solve custom prompt
  python cli.py solve --env sat --prompt "Find satisfying assignment for x1 ∨ x2..."
  
  # Interactive mode
  python cli.py interactive --env ded
  
  # Generate dataset from Score 1.0 conversations (ABD tasks 20000-20100)
  python cli.py generate-dataset --env abd --start-id 20000 --end-id 20100 --verbose
  
  # Generate dataset for DED tasks 20000-23302 with API key
  python cli.py generate-dataset --env ded --start-id 20000 --end-id 23302 --api-key YOUR_KEY
  
  # Run with custom model
  python cli.py benchmark --env abd --num-tasks 5 --model deepseek-ai/DeepSeek-V3
        """
    )
    
    # Use parent parser for shared arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--config", default="config.json", help="Config file path")
    parent_parser.add_argument("--api-key", help="API key (overrides config)")
    parent_parser.add_argument("--base-url", help="API base URL (overrides config)")
    parent_parser.add_argument("--model", help="Model name (overrides config)")
    parent_parser.add_argument("--temperature", type=float, help="Temperature (overrides config)")
    parent_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parent_parser.add_argument("--save", action="store_true", help="Save results to files")
    parent_parser.add_argument("--output-dir", default="results", help="Output directory for results")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Evaluate command (single task from dataset)
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a single task from dataset", parents=[parent_parser])
    eval_parser.add_argument("--env", required=True, choices=["sat", "abd", "ded"], help="Environment name")
    eval_parser.add_argument("--task-id", type=int, required=True, help="Task ID to evaluate")
    
    # Solve command (custom prompt)
    solve_parser = subparsers.add_parser("solve", help="Solve a custom prompt", parents=[parent_parser])
    solve_parser.add_argument("--env", required=True, choices=["sat", "abd", "ded"], help="Environment name")
    solve_parser.add_argument("--prompt", required=True, help="Task prompt")
    
    # Benchmark command (multiple tasks)
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark on multiple tasks", parents=[parent_parser])
    bench_parser.add_argument("--env", required=True, choices=["sat", "abd", "ded"], help="Environment name")
    bench_parser.add_argument("--num-tasks", type=int, default=10, help="Number of tasks to evaluate")
    bench_parser.add_argument("--task-ids", type=int, nargs="+", help="Specific task IDs to evaluate")
    bench_parser.add_argument("--save-interval", type=int, default=10, help="Save intermediate results every N tasks")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode", parents=[parent_parser])
    interactive_parser.add_argument("--env", default="ded", choices=["sat", "abd", "ded"], help="Environment type (default: ded)")
    
    # Generate dataset command
    dataset_parser = subparsers.add_parser("generate-dataset", help="Generate training dataset from Score 1.0 tasks", parents=[parent_parser])
    dataset_parser.add_argument("--env", required=True, choices=["abd", "ded"], help="Environment name (sat not supported for dataset generation)")
    dataset_parser.add_argument("--start-id", type=int, required=True, help="Starting task ID (inclusive)")
    dataset_parser.add_argument("--end-id", type=int, required=True, help="Ending task ID (inclusive)")
    dataset_parser.add_argument("--save-interval", type=int, default=10, help="Save intermediate results every N tasks")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Check for API key
    if not args.api_key and not os.getenv("OPENAI_API_KEY") and args.base_url is None:
        print("Error: No API key provided. Set OPENAI_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Run async command
    if args.command == "evaluate":
        asyncio.run(cmd_evaluate(args))
    elif args.command == "solve":
        cmd_solve(args)
    elif args.command == "benchmark":
        asyncio.run(run_benchmark(args))
    elif args.command == "interactive":
        cmd_interactive(args)
    elif args.command == "generate-dataset":
        asyncio.run(cmd_generate_dataset(args))


if __name__ == "__main__":
    main()
