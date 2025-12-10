#!/usr/bin/env python3
"""
Command-line interface for PrimeIntellect Agent

Usage:
    python cli.py solve --env mth --prompt "What is 2+2?" 
    python cli.py benchmark --env mth --num-tasks 10
    python cli.py interactive --env mth
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from agent import create_agent, PrimeIntellectAgentConfig
from config_utils import load_config, create_agent_config_from_dict, get_env_configs


def cmd_solve(args):
    """Solve a single task."""
    # Load config
    try:
        config_dict = load_config(args.config)
        agent_config = create_agent_config_from_dict(config_dict)
        agent_config.verbose = args.verbose
    except FileNotFoundError:
        agent_config = PrimeIntellectAgentConfig(
            model=args.model,
            verbose=args.verbose
        )
    
    # Create agent
    agent = create_agent(
        api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
        model=args.model or agent_config.model,
        temperature=args.temperature or agent_config.temperature,
        verbose=args.verbose,
        base_url=args.base_url or os.getenv("BASE_URL") or "https://api.openai.com/v1"
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
    if args.env in ['mth', 'sci']:
        answer = agent.extract_boxed_answer(solution)
        if answer:
            print(f"\nExtracted Answer: {answer}")
    elif args.env == 'cde':
        code = agent.extract_code(solution)
        print(f"\nExtracted Code:\n{code}")


async def cmd_benchmark(args):
    """Run benchmark on multiple tasks."""
    from env_integration import evaluate_agent
    
    print(f"\nRunning benchmark on {args.num_tasks} tasks from {args.env.upper()} environment...")
    
    results = await evaluate_agent(
        env=args.env,
        num_tasks=args.num_tasks,
        api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
        model=args.model or "gpt-4o",
        verbose=args.verbose,
        temperature=args.temperature or 0.7,
        base_url=args.base_url or os.getenv("BASE_URL") or "https://api.openai.com/v1",
        save_results=args.save,
        output_dir=args.output_dir if hasattr(args, 'output_dir') else "results"
    )
    
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    print(f"Environment: {args.env.upper()}")
    print(f"Tasks Evaluated: {results['num_tasks']}")
    print(f"Average Score: {results['average_score']:.2%}")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Individual Scores: {results['scores']}")
    
    if 'saved_to' in results:
        print(f"\n✓ Results saved to: {results['saved_to']}")
    
    print("="*70)


async def cmd_evaluate(args):
    """Evaluate a single task from the dataset."""
    from env_integration import PrimeIntellectEnvironmentAgent
    
    try:
        config_dict = load_config(args.config)
        agent_config = create_agent_config_from_dict(config_dict)
        agent_config.verbose = args.verbose
        env_configs = get_env_configs(config_dict)
    except FileNotFoundError:
        agent_config = PrimeIntellectAgentConfig(
            model=args.model or "gpt-4o",
            verbose=args.verbose,
            base_url=args.base_url or "https://api.openai.com/v1"
        )
        env_configs = {}
    
    # Override with CLI args
    if args.api_key:
        agent_config.api_key = args.api_key
    if args.model:
        agent_config.model = args.model
    if args.temperature is not None:
        agent_config.temperature = args.temperature
    
    env_agent = PrimeIntellectEnvironmentAgent(agent_config, env_configs)
    
    print(f"\nEvaluating task {args.task_id} from {args.env.upper()} environment...")
    
    result = await env_agent.solve_and_evaluate(
        env=args.env,
        task_id=args.task_id,
        save_results=args.save,
        output_dir=args.output_dir if hasattr(args, 'output_dir') else "results"
    )
    
    print("\n" + "="*70)
    print("EVALUATION RESULT")
    print("="*70)
    print(f"Environment: {args.env.upper()}")
    print(f"Task ID: {result['task_id']}")
    print(f"\nPrompt:\n{result['challenge'].prompt}")
    print(f"\n{'-'*70}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\n{'-'*70}")
    print(f"\nScore: {result['score']}")
    
    # Show extracted answer if available
    if result.get('extracted_answer'):
        print(f"\nExtracted Answer: {result['extracted_answer']}")
    
    # Show ground truth if available
    if hasattr(result['challenge'], 'extra') and 'answer' in result['challenge'].extra:
        print(f"Ground Truth: {result['challenge'].extra['answer']}")
    
    # Show if saved
    if 'saved_to' in result:
        print(f"\n✓ Results saved to: {result['saved_to']}")
    
    print("="*70)


def cmd_interactive(args):
    """Interactive mode for solving tasks."""
    # Load config
    try:
        config_dict = load_config(args.config)
        agent_config = create_agent_config_from_dict(config_dict)
    except FileNotFoundError:
        agent_config = PrimeIntellectAgentConfig(
            model=args.model or "gpt-4o"
        )
    
    # Override with CLI args
    if args.api_key:
        agent_config.api_key = args.api_key
    if args.model:
        agent_config.model = args.model
    
    agent = create_agent(
        api_key=agent_config.api_key,
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
            if args.env in ['mth', 'sci']:
                answer = agent.extract_boxed_answer(solution)
                if answer:
                    print(f"\n✓ Answer: {answer}")
            elif args.env == 'cde':
                code = agent.extract_code(solution)
                if code and len(code) < len(solution) * 0.9:  # Only show if extraction worked
                    print(f"\n✓ Extracted Code:\n{code[:200]}{'...' if len(code) > 200 else ''}")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="PrimeIntellect Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global arguments
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (default: config.json)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='OpenAI API key (default: OPENAI_API_KEY env var)')
    parser.add_argument('--base-url', type=str, default=None,
                        help='LLM endpoint (default: BASE_URL env var)')
    parser.add_argument('--model', type=str, default=None,
                       help='Model to use (default: from config or gpt-4o)')
    parser.add_argument('--temperature', type=float, default=None,
                       help='Sampling temperature (default: from config or 0.7)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--save', action='store_true',
                       help='Save results to file')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results (default: results)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Solve command
    solve_parser = subparsers.add_parser('solve', help='Solve a single task')
    solve_parser.add_argument('--env', type=str, required=True,
                             choices=['cde', 'lgc', 'mth', 'sci'],
                             help='Environment type')
    solve_parser.add_argument('--prompt', type=str, required=True,
                             help='Task prompt')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmark on multiple tasks')
    benchmark_parser.add_argument('--env', type=str, required=True,
                                 choices=['cde', 'lgc', 'mth', 'sci'],
                                 help='Environment type')
    benchmark_parser.add_argument('--num-tasks', type=int, default=10,
                                 help='Number of tasks to evaluate (default: 10)')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a single task from dataset')
    eval_parser.add_argument('--env', type=str, required=True,
                           choices=['cde', 'lgc', 'mth', 'sci'],
                           help='Environment type')
    eval_parser.add_argument('--task-id', type=int, default=0,
                           help='Task ID to evaluate (default: 0)')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive mode')
    interactive_parser.add_argument('--env', type=str, default='mth',
                                   choices=['cde', 'lgc', 'mth', 'sci'],
                                   help='Environment type (default: mth)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Check for API key
    if not args.api_key and not os.getenv("OPENAI_API_KEY"):
        print("Error: No API key provided. Set OPENAI_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'solve':
            cmd_solve(args)
        elif args.command == 'benchmark':
            asyncio.run(cmd_benchmark(args))
        elif args.command == 'evaluate':
            asyncio.run(cmd_evaluate(args))
        elif args.command == 'interactive':
            cmd_interactive(args)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
