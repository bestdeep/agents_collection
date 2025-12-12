#!/usr/bin/env python3
"""
Example usage of Affine Agent

Demonstrates how to use the agent for ABD and DED tasks.
"""

import asyncio
import os
from agent import create_agent, AffineAgentConfig
from env_integration import AffineEnvironmentAgent, evaluate_agent


async def example_basic_usage():
    """Example 1: Basic agent usage without environment integration."""
    print("\n" + "="*70)
    print("Example 1: Basic Agent Usage")
    print("="*70)
    
    # Create an agent
    agent = create_agent(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o",
        temperature=0.7,
        verbose=False
    )
    
    # Example ABD task
    print("\nABD Task: Reverse engineering program input")
    print("-"*70)
    abd_prompt = """You are given this Python program:
```python
n = int(input())
total = 0
for i in range(n):
    total += int(input())
print(f"Sum: {total}")
```

And this expected output:
```
Sum: 30
```

What input would produce this output?"""
    
    abd_solution = agent.solve(abd_prompt, env="abd")
    extracted_input = agent.extract_input(abd_solution)
    
    print(f"Extracted Input:\n{extracted_input}")
    
    # Example DED task
    print("\n\nDED Task: Code generation")
    print("-"*70)
    ded_prompt = """Write a Python program that:
1. Reads an integer n from stdin
2. Reads n more integers from stdin (one per line)
3. Outputs the sum of those n integers

Example:
Input:
3
10
20
30

Output:
60
"""
    
    ded_solution = agent.solve(ded_prompt, env="ded")
    extracted_code = agent.extract_code(ded_solution)
    
    print(f"Generated Code:\n{extracted_code}")


async def example_environment_integration():
    """Example 2: Using environment integration for automatic evaluation."""
    print("\n" + "="*70)
    print("Example 2: Environment Integration with Evaluation")
    print("="*70)
    
    # Create agent configuration
    config = AffineAgentConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o",
        temperature=0.7,
        verbose=True
    )
    
    # Create environment agent
    env_agent = AffineEnvironmentAgent(config)
    
    # Solve and evaluate an ABD task
    print("\nEvaluating ABD task...")
    print("-"*70)
    abd_result = await env_agent.solve_and_evaluate(
        env="abd",
        task_id=0,
        save_results=True,
        output_dir="example_results"
    )
    
    print(f"\nABD Results:")
    print(f"  Task ID: {abd_result['task_id']}")
    print(f"  Score: {abd_result['score']}")
    print(f"  Extracted Input:\n{abd_result['extracted_answer']}")
    
    # Solve and evaluate a DED task
    print("\n\nEvaluating DED task...")
    print("-"*70)
    ded_result = await env_agent.solve_and_evaluate(
        env="ded",
        task_id=0,
        save_results=True,
        output_dir="example_results"
    )
    
    print(f"\nDED Results:")
    print(f"  Task ID: {ded_result['task_id']}")
    print(f"  Score: {ded_result['score']}")
    print(f"  Code extracted: {'Yes' if ded_result['extracted_answer'] else 'No'}")


async def example_batch_evaluation():
    """Example 3: Batch evaluation on multiple tasks."""
    print("\n" + "="*70)
    print("Example 3: Batch Evaluation")
    print("="*70)
    
    # Create configuration
    config = AffineAgentConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o",
        temperature=0.7,
        verbose=False  # Less verbose for batch processing
    )
    
    # Evaluate 5 DED tasks
    print("\nEvaluating 5 DED tasks...")
    print("-"*70)
    results = await evaluate_agent(
        env="ded",
        num_tasks=5,
        agent_config=config,
        save_results=True,
        output_dir="example_results",
        verbose=True
    )
    
    print(f"\n\nBatch Evaluation Summary:")
    print(f"  Environment: {results['env']}")
    print(f"  Total tasks: {results['num_tasks']}")
    print(f"  Average score: {results['avg_score']:.3f}")
    print(f"  Max score: {results['max_score']:.3f}")
    print(f"  Min score: {results['min_score']:.3f}")
    print(f"  Individual scores: {results['scores']}")


async def example_specific_tasks():
    """Example 4: Evaluate specific task IDs."""
    print("\n" + "="*70)
    print("Example 4: Evaluating Specific Task IDs")
    print("="*70)
    
    # Create configuration
    config = AffineAgentConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o",
        temperature=0.7,
        verbose=True
    )
    
    # Evaluate specific ABD tasks
    task_ids = [0, 5, 10]
    print(f"\nEvaluating ABD tasks: {task_ids}")
    print("-"*70)
    
    results = await evaluate_agent(
        env="abd",
        task_ids=task_ids,
        agent_config=config,
        save_results=True,
        output_dir="example_results",
        verbose=True
    )
    
    print(f"\n\nResults for specific tasks:")
    for i, task_id in enumerate(task_ids):
        score = results['scores'][i]
        print(f"  Task {task_id}: {score}")


async def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Affine Agent - Usage Examples")
    print("="*70)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠ WARNING: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        print("\nExamples will not run without an API key.\n")
        return
    
    # Run examples
    try:
        # Basic usage
        await example_basic_usage()
        
        # Environment integration
        await example_environment_integration()
        
        # Batch evaluation
        await example_batch_evaluation()
        
        # Specific task IDs
        await example_specific_tasks()
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
