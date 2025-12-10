"""
Example usage and testing script for PrimeIntellect Agent
"""

import os
import asyncio
from agent import create_agent, PrimeIntellectAgentConfig


def example_basic_usage():
    """Demonstrate basic agent usage without environment integration."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Agent Usage (No Environment Integration)")
    print("="*70)
    
    # Create agent
    agent = create_agent(
        model="gpt-4o-mini",  # Use mini for faster/cheaper testing
        temperature=0.7,
        verbose=False
    )
    
    # Example 1: Math problem
    print("\n--- Math Problem ---")
    math_prompt = "Calculate the area of a circle with radius 5 meters. Show your work and put the final answer in \\boxed{}."
    math_solution = agent.solve(math_prompt, env="mth")
    print(f"Prompt: {math_prompt}")
    print(f"Solution:\n{math_solution}")
    
    boxed_answer = agent.extract_boxed_answer(math_solution)
    if boxed_answer:
        print(f"\n✓ Extracted Answer: {boxed_answer}")
    
    # Example 2: Code problem
    print("\n--- Code Problem ---")
    code_prompt = """Write a Python function that takes a list of numbers and returns the median value.
Handle both odd and even length lists."""
    code_solution = agent.solve(code_prompt, env="cde")
    print(f"Prompt: {code_prompt}")
    print(f"Solution:\n{code_solution}")
    
    extracted_code = agent.extract_code(code_solution)
    print(f"\n✓ Extracted Code:\n{extracted_code}")
    
    # Example 3: Logic problem
    print("\n--- Logic Problem ---")
    logic_prompt = """Five houses are painted in five different colors. In each house lives a person of a different nationality.
These five people drink different beverages, smoke different brands of cigars, and keep different pets.

Given these clues:
1. The Brit lives in the red house
2. The Swede keeps dogs as pets
3. The Dane drinks tea

Who owns the fish? (This is a simplified version - provide your reasoning)"""
    logic_solution = agent.solve(logic_prompt, env="lgc")
    print(f"Prompt: {logic_prompt}")
    print(f"Solution:\n{logic_solution}")
    
    # Example 4: Science problem
    print("\n--- Science Problem ---")
    sci_prompt = """A ball is dropped from a height of 45 meters. Assuming no air resistance and g = 10 m/s²,
how long does it take to hit the ground? Put your answer in \\boxed{} in seconds."""
    sci_solution = agent.solve(sci_prompt, env="sci")
    print(f"Prompt: {sci_prompt}")
    print(f"Solution:\n{sci_solution}")
    
    sci_answer = agent.extract_boxed_answer(sci_solution)
    if sci_answer:
        print(f"\n✓ Extracted Answer: {sci_answer}")


async def example_environment_integration():
    """Demonstrate full environment integration with task generation and evaluation."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Full Environment Integration")
    print("="*70)
    print("\nNOTE: This requires the PrimeIntellect environments to be properly set up.")
    print("Skipping if environments are not available...\n")
    
    try:
        from env_integration import PrimeIntellectEnvironmentAgent, PrimeIntellectAgentConfig
        
        # Create agent with configuration
        agent_config = PrimeIntellectAgentConfig(
            model="gpt-4o-mini",
            temperature=0.7,
            verbose=False
        )
        
        env_agent = PrimeIntellectEnvironmentAgent(agent_config)
        
        # Test a single task from each environment
        environments = ["mth", "sci", "lgc", "cde"]
        
        for env in environments:
            print(f"\n--- Testing {env.upper()} Environment ---")
            try:
                result = await env_agent.solve_and_evaluate(env, task_id=0)
                
                print(f"Task ID: {result['task_id']}")
                print(f"Prompt: {result['challenge'].prompt[:200]}...")
                print(f"Response: {result['response'][:300]}...")
                print(f"✓ Score: {result['score']}")
                
            except Exception as e:
                print(f"✗ Error testing {env}: {e}")
        
        print("\n" + "="*70)
        
    except ImportError as e:
        print(f"Environment integration not available: {e}")
        print("Make sure the PrimeIntellect environments are set up correctly.")


async def example_benchmark():
    """Demonstrate running a benchmark on multiple tasks."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Benchmark Evaluation")
    print("="*70)
    print("\nNOTE: This requires the PrimeIntellect environments to be properly set up.")
    print("Running a small benchmark on Math tasks...\n")
    
    try:
        from env_integration import evaluate_agent
        
        # Run benchmark on 3 math tasks
        results = await evaluate_agent(
            env="mth",
            num_tasks=3,
            model="gpt-4o-mini",
            verbose=True
        )
        
        print("\n" + "="*70)
        print("BENCHMARK RESULTS")
        print("="*70)
        print(f"Environment: MTH")
        print(f"Tasks Evaluated: {results['num_tasks']}")
        print(f"Average Score: {results['average_score']:.2%}")
        print(f"Success Rate: {results['success_rate']:.2%}")
        print(f"Individual Scores: {results['scores']}")
        
    except ImportError as e:
        print(f"Environment integration not available: {e}")
        print("Make sure the PrimeIntellect environments are set up correctly.")
    except Exception as e:
        print(f"Error running benchmark: {e}")


def example_batch_solving():
    """Demonstrate batch solving of multiple challenges."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Solving")
    print("="*70)
    
    agent = create_agent(
        model="gpt-4o-mini",
        temperature=0.7,
        verbose=False
    )
    
    # Create multiple challenges
    challenges = [
        {
            "env": "mth",
            "prompt": "What is 15 + 27? Put your answer in \\boxed{}."
        },
        {
            "env": "mth",
            "prompt": "Calculate 8 × 9. Put your answer in \\boxed{}."
        },
        {
            "env": "sci",
            "prompt": "What is the chemical formula for water? Put your answer in \\boxed{}."
        }
    ]
    
    print(f"\nSolving {len(challenges)} challenges...")
    solutions = agent.batch_solve(challenges, reset_between=False)
    
    for i, (challenge, solution) in enumerate(zip(challenges, solutions), 1):
        print(f"\n--- Challenge {i} ({challenge['env'].upper()}) ---")
        print(f"Prompt: {challenge['prompt']}")
        print(f"Solution: {solution[:200]}...")
        
        if challenge['env'] in ['mth', 'sci']:
            answer = agent.extract_boxed_answer(solution)
            if answer:
                print(f"✓ Answer: {answer}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("PrimeIntellect Agent - Examples and Testing")
    print("="*70)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  WARNING: OPENAI_API_KEY environment variable not set!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("Or pass api_key parameter when creating the agent.\n")
        return
    
    # Run synchronous examples
    example_basic_usage()
    example_batch_solving()
    
    # Run asynchronous examples
    print("\n" + "="*70)
    print("Running async examples...")
    print("="*70)
    
    asyncio.run(example_environment_integration())
    
    # Uncomment to run benchmark (takes longer)
    # asyncio.run(example_benchmark())
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()
