"""
Example: Using Result Saver to Save Conversation History and Scores

This example demonstrates how to save evaluation results including:
- Conversation history
- Scores
- Extracted answers
- Full evaluation details
"""

import asyncio
from env_integration import PrimeIntellectEnvironmentAgent, PrimeIntellectAgentConfig
from result_saver import ResultSaver


async def example_save_single_evaluation():
    """Example: Save a single evaluation result."""
    print("\n" + "="*70)
    print("Example 1: Save Single Evaluation Result")
    print("="*70)
    
    # Create agent
    config = PrimeIntellectAgentConfig(
        model="gpt-4o-mini",
        verbose=False
    )
    env_agent = PrimeIntellectEnvironmentAgent(config)
    
    # Evaluate with save enabled
    result = await env_agent.solve_and_evaluate(
        env="mth",
        task_id=0,
        save_results=True,
        output_dir="results"
    )
    
    print(f"✓ Task {result['task_id']} evaluated")
    print(f"  Score: {result['score']}")
    print(f"  Extracted Answer: {result.get('extracted_answer')}")
    print(f"  Saved to: {result.get('saved_to')}")


async def example_save_benchmark():
    """Example: Save benchmark results."""
    print("\n" + "="*70)
    print("Example 2: Save Benchmark Results")
    print("="*70)
    
    # Create agent
    config = PrimeIntellectAgentConfig(
        model="gpt-4o-mini",
        verbose=False
    )
    env_agent = PrimeIntellectEnvironmentAgent(config)
    
    # Run benchmark with save enabled
    benchmark = await env_agent.run_benchmark(
        env="mth",
        num_tasks=3,
        save_results=True,
        output_dir="results"
    )
    
    print(f"✓ Benchmark completed")
    print(f"  Average Score: {benchmark['average_score']:.2%}")
    print(f"  Success Rate: {benchmark['success_rate']:.2%}")
    print(f"  Saved to: {benchmark.get('saved_to')}")


def example_load_results():
    """Example: Load and inspect saved results."""
    print("\n" + "="*70)
    print("Example 3: Load Saved Results")
    print("="*70)
    
    saver = ResultSaver("results")
    
    # List saved files
    files = saver.list_saved_results()
    print(f"\nFound {len(files)} saved result files:")
    for f in files[:5]:  # Show first 5
        print(f"  - {f}")
    
    # Load and inspect a result
    if files:
        print(f"\nLoading: {files[0]}")
        data = saver.load_evaluation(files[0])
        
        print("\nResult Summary:")
        print(f"  Environment: {data['metadata']['env']}")
        print(f"  Task ID: {data['metadata']['task_id']}")
        print(f"  Score: {data['score']}")
        print(f"  Extracted Answer: {data.get('extracted_answer')}")
        print(f"  Ground Truth: {data.get('ground_truth')}")
        print(f"  Conversation Turns: {len(data.get('conversation_history', []))}")


def example_statistics():
    """Example: Get statistics from saved results."""
    print("\n" + "="*70)
    print("Example 4: Get Statistics")
    print("="*70)
    
    saver = ResultSaver("results")
    
    # Overall statistics
    stats = saver.get_statistics()
    print("\nOverall Statistics:")
    print(f"  Total Evaluations: {stats.get('total_evaluations', 0)}")
    print(f"  Average Score: {stats.get('average_score', 0):.2%}")
    print(f"  Success Rate: {stats.get('success_rate', 0):.2%}")
    
    # Per-environment statistics
    for env in ['cde', 'lgc', 'mth', 'sci']:
        env_stats = saver.get_statistics(env)
        if env_stats.get('total_evaluations', 0) > 0:
            print(f"\n{env.upper()} Statistics:")
            print(f"  Total: {env_stats['total_evaluations']}")
            print(f"  Avg Score: {env_stats['average_score']:.2%}")
            print(f"  Success Rate: {env_stats['success_rate']:.2%}")


async def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Result Saver Examples")
    print("="*70)
    
    # Example 1: Save single evaluation
    # await example_save_single_evaluation()
    
    # Example 2: Save benchmark
    # await example_save_benchmark()
    
    # Example 3: Load results
    example_load_results()
    
    # Example 4: Statistics
    example_statistics()
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
