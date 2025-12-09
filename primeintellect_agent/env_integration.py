"""
PrimeIntellect Agent - Environment Integration

This module provides integration with the PrimeIntellect task environments,
allowing the agent to work seamlessly with CDE, LGC, MTH, and SCI tasks.
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any

# Add environment paths
sys.path.insert(0, '/home/120/affinetes/environments/primeintellect/cde')
sys.path.insert(0, '/home/120/affinetes/environments/primeintellect/lgc')
sys.path.insert(0, '/home/120/affinetes/environments/primeintellect/mth')
sys.path.insert(0, '/home/120/affinetes/environments/primeintellect/sci')

from agent import PrimeIntellectAgent, PrimeIntellectAgentConfig


class PrimeIntellectEnvironmentAgent:
    """
    Agent wrapper that integrates with PrimeIntellect task environments.
    Handles task generation and evaluation across all environment types.
    """
    
    def __init__(
        self,
        agent_config: Optional[PrimeIntellectAgentConfig] = None,
        env_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initialize the environment-integrated agent.
        
        Args:
            agent_config: Configuration for the LLM agent
            env_configs: Environment-specific configurations for task generators
        """
        self.agent = PrimeIntellectAgent(agent_config)
        self.env_configs = env_configs or {}
        self.task_generators = {}
        
    async def initialize_environment(self, env: str):
        """
        Initialize a specific environment's task generator.
        
        Args:
            env: Environment name (cde, lgc, mth, sci)
        """
        if env in self.task_generators:
            return
        
        env_config = self.env_configs.get(env, {})
        
        if env == "cde":
            from code_task import CodeTask
            self.task_generators[env] = CodeTask(**env_config)
        elif env == "lgc":
            from logic_task import LogicTask
            self.task_generators[env] = LogicTask(**env_config)
        elif env == "mth":
            from math_task import MathTask
            self.task_generators[env] = MathTask(**env_config)
        elif env == "sci":
            from sci_task import ScienceTask
            self.task_generators[env] = ScienceTask(**env_config)
        else:
            raise ValueError(f"Unknown environment: {env}")
    
    async def generate_task(self, env: str, task_id: Optional[int] = None):
        """
        Generate a task from the specified environment.
        
        Args:
            env: Environment name
            task_id: Optional task ID for deterministic selection
            
        Returns:
            Challenge object
        """
        await self.initialize_environment(env)
        task_gen = self.task_generators[env]
        return await task_gen.generate(task_id=task_id)
    
    async def solve_and_evaluate(
        self, 
        env: str, 
        task_id: Optional[int] = None,
        **eval_kwargs
    ) -> Dict[str, Any]:
        """
        Generate a task, solve it with the agent, and evaluate the solution.
        
        Args:
            env: Environment name
            task_id: Optional task ID
            **eval_kwargs: Additional evaluation arguments
            
        Returns:
            Dictionary with challenge, response, and evaluation results
        """
        # Generate task
        challenge = await self.generate_task(env, task_id)
        
        # Solve with agent
        response = self.agent.solve_challenge({
            "env": env,
            "prompt": challenge.prompt,
            "extra": challenge.extra
        })
        
        # Evaluate response
        task_gen = self.task_generators[env]
        
        if env == "cde":
            # Code evaluation doesn't need extra kwargs by default
            score = await task_gen.evaluate(response, challenge, **eval_kwargs)
        elif env in ["mth", "sci"]:
            # Math and science may use judge models
            score = await task_gen.evaluate(response, challenge, **eval_kwargs)
        else:  # lgc
            score = await task_gen.evaluate(response, challenge)
        
        return {
            "env": env,
            "challenge": challenge,
            "response": response,
            "score": score,
            "task_id": task_id
        }
    
    async def run_benchmark(
        self,
        env: str,
        num_tasks: int = 10,
        start_task_id: int = 0,
        **eval_kwargs
    ) -> Dict[str, Any]:
        """
        Run a benchmark on multiple tasks.
        
        Args:
            env: Environment name
            num_tasks: Number of tasks to evaluate
            start_task_id: Starting task ID
            **eval_kwargs: Additional evaluation arguments
            
        Returns:
            Dictionary with results and statistics
        """
        results = []
        scores = []
        
        for i in range(num_tasks):
            task_id = start_task_id + i
            print(f"Evaluating task {task_id} ({i+1}/{num_tasks})...")
            
            try:
                result = await self.solve_and_evaluate(env, task_id, **eval_kwargs)
                results.append(result)
                scores.append(result["score"])
                print(f"Task {task_id}: Score = {result['score']}")
            except Exception as e:
                print(f"Error on task {task_id}: {e}")
                results.append({
                    "env": env,
                    "task_id": task_id,
                    "error": str(e),
                    "score": 0.0
                })
                scores.append(0.0)
        
        return {
            "env": env,
            "num_tasks": num_tasks,
            "results": results,
            "scores": scores,
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "success_rate": sum(1 for s in scores if s > 0) / len(scores) if scores else 0.0
        }
    
    def reset_agent(self, env: Optional[str] = None):
        """
        Reset agent conversation history.
        
        Args:
            env: If specified, reset only this environment. Otherwise reset all.
        """
        if env:
            self.agent.reset_conversation(env)
        else:
            self.agent.clear_all_conversations()


# Convenience function for running evaluations
async def evaluate_agent(
    env: str = "mth",
    num_tasks: int = 10,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick function to evaluate an agent on a specific environment.
    
    Args:
        env: Environment name (cde, lgc, mth, sci)
        num_tasks: Number of tasks to evaluate
        api_key: OpenAI API key
        model: Model name
        verbose: Print progress
        **kwargs: Additional configuration
        
    Returns:
        Benchmark results dictionary
    """
    agent_config = PrimeIntellectAgentConfig(
        api_key=api_key,
        model=model,
        verbose=verbose,
        **kwargs
    )
    
    env_agent = PrimeIntellectEnvironmentAgent(agent_config)
    return await env_agent.run_benchmark(env, num_tasks)


# Example usage
if __name__ == "__main__":
    async def main():
        # Create agent with custom configuration
        agent_config = PrimeIntellectAgentConfig(
            model="gpt-4o",
            temperature=0.7,
            verbose=True
        )
        
        env_agent = PrimeIntellectEnvironmentAgent(agent_config)
        
        # Test on a single task from each environment
        print("\n" + "="*60)
        print("Testing Math Environment (MTH)")
        print("="*60)
        result_mth = await env_agent.solve_and_evaluate("mth", task_id=0)
        print(f"Score: {result_mth['score']}")
        
        print("\n" + "="*60)
        print("Testing Science Environment (SCI)")
        print("="*60)
        result_sci = await env_agent.solve_and_evaluate("sci", task_id=0)
        print(f"Score: {result_sci['score']}")
        
        print("\n" + "="*60)
        print("Testing Logic Environment (LGC)")
        print("="*60)
        result_lgc = await env_agent.solve_and_evaluate("lgc", task_id=0)
        print(f"Score: {result_lgc['score']}")
        
        print("\n" + "="*60)
        print("Testing Code Environment (CDE)")
        print("="*60)
        result_cde = await env_agent.solve_and_evaluate("cde", task_id=0)
        print(f"Score: {result_cde['score']}")
        
        # Run a small benchmark
        print("\n" + "="*60)
        print("Running Math Benchmark (5 tasks)")
        print("="*60)
        benchmark = await env_agent.run_benchmark("mth", num_tasks=5)
        print(f"\nResults:")
        print(f"Average Score: {benchmark['average_score']:.2%}")
        print(f"Success Rate: {benchmark['success_rate']:.2%}")
    
    # Run the async main function
    asyncio.run(main())
