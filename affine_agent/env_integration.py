"""
Affine Agent - Environment Integration

This module provides integration with the Affine task environments,
allowing the agent to work seamlessly with ABD and DED tasks.
"""

import sys
import gc
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import json

# Add environment paths
sys.path.insert(0, '/home/120/affinetes/environments/affine')

from agent import AffineAgent, AffineAgentConfig


# Global task generator cache to avoid reloading datasets
_TASK_GENERATOR_CACHE = {}


class AffineEnvironmentAgent:
    """
    Agent wrapper that integrates with Affine task environments.
    Handles task generation and evaluation for ABD and DED tasks.
    """
    
    def __init__(
        self,
        agent_config: Optional[AffineAgentConfig] = None,
        env_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initialize the environment-integrated agent.
        
        Args:
            agent_config: Configuration for the LLM agent
            env_configs: Environment-specific configurations for task generators
        """
        self.agent = AffineAgent(agent_config)
        self.env_configs = env_configs or {}
        self.task_generators = {}
        
    async def initialize_environment(self, env: str):
        """
        Initialize a specific environment's task generator.
        Uses global cache to avoid reloading datasets.
        
        Args:
            env: Environment name (sat, abd, ded)
        """
        # Check global cache first
        if env in _TASK_GENERATOR_CACHE:
            self.task_generators[env] = _TASK_GENERATOR_CACHE[env]
            return
            
        if env in self.task_generators:
            return
        
        env_config = self.env_configs.get(env, {})
        
        # Filter config to only include valid parameters for each task type
        if env == "sat":
            from sat import SATTask
            # SATTask accepts: n, k
            sat_config = {k: v for k, v in env_config.items() if k in ['n', 'k']}
            task_gen = SATTask(**sat_config)
        elif env == "abd":
            from abd import ABDTask
            # ABDTask accepts: dataset, dataset_name (not split)
            abd_config = {k: v for k, v in env_config.items() if k in ['dataset', 'dataset_name']}
            task_gen = ABDTask(**abd_config)
        elif env == "ded":
            from ded import DEDTask
            # DEDTask accepts: dataset, dataset_name (not split)
            ded_config = {k: v for k, v in env_config.items() if k in ['dataset', 'dataset_name']}
            task_gen = DEDTask(**ded_config)
        else:
            raise ValueError(f"Unknown environment: {env}")
        
        # Cache globally and locally
        _TASK_GENERATOR_CACHE[env] = task_gen
        self.task_generators[env] = task_gen
    
    async def generate_task(self, env: str, task_id: Optional[int] = None):
        """
        Generate a task from the specified environment.
        
        Args:
            env: Environment name (sat, abd, ded)
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
        save_results: bool = False,
        output_dir: str = "results",
        **eval_kwargs
    ) -> Dict[str, Any]:
        """
        Generate a task, solve it with the agent, and evaluate the solution.
        
        Args:
            env: Environment name (sat, abd, ded)
            task_id: Optional task ID
            save_results: Whether to save results to file
            output_dir: Directory to save results
            **eval_kwargs: Additional evaluation arguments
            
        Returns:
            Dictionary with challenge, response, evaluation results, conversation history, and extracted answer
        """
        # Generate task
        challenge = await self.generate_task(env, task_id)
        
        # Solve with agent
        response = self.agent.solve_challenge({
            "env": env,
            "prompt": challenge.prompt,
            "extra": challenge.extra
        })
        
        # Get conversation history
        conversation_history = self.agent.get_conversation_history(env)
        
        # Extract answer based on environment type
        extracted_answer = None
        if env == "sat":
            extracted_answer = self.agent.extract_sat_assignment(response)
        elif env == "abd":
            extracted_answer = self.agent.extract_input(response)
        elif env == "ded":
            extracted_answer = self.agent.extract_code(response)
        
        # Evaluate response
        task_gen = self.task_generators[env]
        
        try:
            print(f"[DEBUG] Starting evaluation for {env} task {task_id}...")
            
            # Add a safety timeout wrapper
            async def eval_with_safety():
                try:
                    return await asyncio.wait_for(
                        task_gen.evaluate(response, challenge, **eval_kwargs),
                        timeout=180  # 3 minute timeout for evaluation
                    )
                except asyncio.TimeoutError:
                    print(f"[WARNING] Evaluation timed out after 180s")
                    return (0.0, "0/0 (timeout)")
            
            score = await eval_with_safety()
            
            print(f"[DEBUG] Evaluation completed with score: {score}")
        except Exception as e:
            print(f"[DEBUG] Evaluation failed with error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            score = (0.0, "0/0")  # Default failure score
        
        result = {
            "env": env,
            "challenge": challenge,
            "response": response,
            "score": score,
            "task_id": task_id,
            "conversation_history": conversation_history,
            "extracted_answer": extracted_answer
        }
        
        # Save results if requested
        if save_results:
            await self.save_result(result, output_dir)
        
        return result
    
    async def save_result(self, result: Dict[str, Any], output_dir: str):
        """
        Save evaluation result to a JSON file.
        
        Args:
            result: Result dictionary from solve_and_evaluate
            output_dir: Directory to save results
        """
        from result_saver import save_result
        await save_result(result, output_dir)
    
    def reset_conversation(self, env: str):
        """Reset the conversation for a specific environment."""
        self.agent.reset_conversation(env)
    
    async def run_benchmark(
        self,
        env: str,
        num_tasks: int = 10,
        start_task_id: int = 0,
        save_results: bool = False,
        output_dir: str = "results",
        save_interval: int = 10,
        skip_on_error: bool = True,
        **eval_kwargs
    ) -> Dict[str, Any]:
        """
        Run a benchmark on multiple tasks.
        
        Args:
            env: Environment name (sat, abd, ded)
            num_tasks: Number of tasks to evaluate
            start_task_id: Starting task ID
            save_results: Whether to save results to file
            output_dir: Directory to save results
            save_interval: Save intermediate results every N tasks (default: 10)
            skip_on_error: Skip tasks that cause errors and continue (default: True)
            **eval_kwargs: Additional evaluation arguments
            
        Returns:
            Dictionary with results, statistics, conversation histories, and extracted answers
        """
        results = []
        scores = []
        skipped_tasks = []
        
        # Initialize saver if saving results
        saver = None
        if save_results:
            from result_saver import ResultSaver
            saver = ResultSaver(output_dir)
        
        tasks_completed = 0
        task_id = start_task_id
        
        while tasks_completed < num_tasks:
            print(f"Evaluating task {task_id} ({tasks_completed+1}/{num_tasks})...")
            
            try:
                result = await self.solve_and_evaluate(env, task_id, save_results=False, **eval_kwargs)
                
                # Extract numeric score (handle both single values and tuples)
                score = result["score"]
                numeric_score = score[0] if isinstance(score, tuple) else score
                scores.append(numeric_score)
                
                # Store minimal result info
                minimal_result = {
                    "env": result["env"],
                    "task_id": result["task_id"],
                    "score": result["score"],
                    "extracted_answer": result.get("extracted_answer"),
                    "error": result.get("error")
                }
                results.append(minimal_result)
                
                # Save individual task if enabled
                if save_results and saver:
                    task_filepath = saver.save_evaluation(
                        result,
                        result.get("conversation_history", []),
                        result.get("extracted_answer")
                    )
                
                print(f"Task {task_id}: Score = {result['score']}")
                tasks_completed += 1
                
                # Clear conversation history and garbage collect
                if env in self.agent.conversation_history:
                    self.agent.conversation_history[env] = []
                gc.collect()
                
                # Progress update
                if save_results and saver and tasks_completed % save_interval == 0:
                    summary = {
                        "env": env,
                        "num_tasks": tasks_completed,
                        "average_score": sum(scores) / len(scores) if scores else 0.0,
                        "success_rate": sum(1 for s in scores if s > 0) / len(scores) if scores else 0.0,
                        "scores": scores.copy()
                    }
                    print(f"✓ Progress: {tasks_completed}/{num_tasks} tasks | Avg Score: {summary['average_score']:.2%} | Success: {summary['success_rate']:.2%}")
                
                task_id += 1
                
            except Exception as e:
                error_msg = str(e)[:100]
                print(f"⚠ Task {task_id} failed: {error_msg}")
                
                if skip_on_error:
                    skipped_tasks.append(task_id)
                    print(f"  → Skipping task {task_id} and continuing...")
                    task_id += 1  # Skip this task
                    continue
                else:
                    # Add error result
                    results.append({
                        "env": env,
                        "task_id": task_id,
                        "error": error_msg,
                        "score": 0.0
                    })
                    scores.append(0.0)
                    tasks_completed += 1
                    task_id += 1
        
        benchmark_results = {
            "env": env,
            "num_tasks": tasks_completed,
            "results": results,
            "scores": scores,
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "success_rate": sum(1 for s in scores if s > 0) / len(scores) if scores else 0.0,
            "skipped_tasks": skipped_tasks
        }
        
        if save_results:
            print(f"\n✓ All {tasks_completed} task results completed")
            if skipped_tasks:
                print(f"  ⚠ Skipped {len(skipped_tasks)} problematic task(s): {skipped_tasks}")
            print(f"  Average Score: {benchmark_results['average_score']:.2%}")
            print(f"  Success Rate: {benchmark_results['success_rate']:.2%}")
        
        return benchmark_results


async def evaluate_agent(
    env: str,
    num_tasks: int = 10,
    task_ids: Optional[list] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    agent_config: Optional[AffineAgentConfig] = None,
    save_results: bool = False,
    output_dir: str = "results",
    save_interval: int = 10,
    verbose: bool = True,
    **eval_kwargs
) -> Dict[str, Any]:
    """
    Evaluate the agent on multiple tasks.
    
    Args:
        env: Environment name (sat, abd, ded)
        num_tasks: Number of tasks to evaluate
        task_ids: Optional list of specific task IDs to evaluate
        api_key: API key for LLM service
        base_url: Base URL for API
        model: Model name
        temperature: Sampling temperature
        agent_config: Configuration for the agent (overrides other params if provided)
        save_results: Whether to save individual results
        output_dir: Directory to save results
        save_interval: Save intermediate results every N tasks
        verbose: Print progress
        **eval_kwargs: Additional evaluation arguments
        
    Returns:
        Dictionary with aggregate results
    """
    if agent_config is None:
        agent_config = AffineAgentConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            verbose=verbose
        )
    
    env_agent = AffineEnvironmentAgent(agent_config)
    
    # If task_ids specified, use benchmark with those IDs
    if task_ids:
        # Run each task ID
        results = []
        scores = []
        
        for i, task_id in enumerate(task_ids):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Evaluating task {i+1}/{len(task_ids)} (ID: {task_id})")
                print(f"{'='*60}")
            
            try:
                result = await env_agent.solve_and_evaluate(
                    env=env,
                    task_id=task_id,
                    save_results=save_results,
                    output_dir=output_dir,
                    **eval_kwargs
                )
                results.append(result)
                
                # Extract numeric score
                score_value = result['score']
                if isinstance(score_value, tuple):
                    score_value = score_value[0]
                scores.append(score_value)
                
                if verbose:
                    print(f"Score: {result['score']}")
                    
            except Exception as e:
                print(f"Error evaluating task {task_id}: {e}")
                import traceback
                traceback.print_exc()
                scores.append(0.0)
                continue
            
            # Clear memory periodically
            if (i + 1) % 5 == 0:
                gc.collect()
        
        # Calculate aggregate statistics
        avg_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0
        
        aggregate = {
            "env": env,
            "num_tasks": len(task_ids),
            "results": results,
            "scores": scores,
            "avg_score": avg_score,
            "average_score": avg_score,  # Alias for compatibility
            "max_score": max_score,
            "min_score": min_score,
            "success_rate": sum(1 for s in scores if s > 0) / len(scores) if scores else 0.0
        }
    else:
        # Use run_benchmark for sequential task IDs
        aggregate = await env_agent.run_benchmark(
            env=env,
            num_tasks=num_tasks,
            save_results=save_results,
            output_dir=output_dir,
            save_interval=save_interval,
            **eval_kwargs
        )
        
        # Add aliases for compatibility
        if 'average_score' in aggregate:
            aggregate['avg_score'] = aggregate['average_score']
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluation Complete")
        print(f"{'='*60}")
        print(f"Environment: {env}")
        print(f"Tasks evaluated: {aggregate.get('num_tasks', 0)}")
        print(f"Average score: {aggregate.get('average_score', 0):.3f}")
        if 'success_rate' in aggregate:
            print(f"Success rate: {aggregate['success_rate']:.2%}")
    
    return aggregate


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Create agent configuration
        config = AffineAgentConfig(
            model="gpt-4o",
            temperature=0.7,
            verbose=True
        )
        
        # Create environment agent
        env_agent = AffineEnvironmentAgent(config)
        
        # Test ABD task
        print("Testing ABD (Algorithm By Deduction) task...")
        abd_result = await env_agent.solve_and_evaluate(
            env="abd",
            task_id=0
        )
        print(f"ABD Score: {abd_result['score']}")
        print(f"Extracted Input: {abd_result['extracted_answer']}")
        
        # Test DED task
        print("\n" + "="*60)
        print("Testing DED (Direct Execution Debug) task...")
        ded_result = await env_agent.solve_and_evaluate(
            env="ded",
            task_id=0
        )
        print(f"DED Score: {ded_result['score']}")
        print(f"Extracted Code: {ded_result['extracted_answer'][:100] if ded_result['extracted_answer'] else None}...")
        
        # Run benchmark on multiple tasks
        print("\n" + "="*60)
        print("Running benchmark on 5 DED tasks...")
        benchmark_results = await evaluate_agent(
            env="ded",
            num_tasks=5,
            agent_config=config,
            verbose=True
        )
        print(f"\nBenchmark Average Score: {benchmark_results['avg_score']:.3f}")
    
    asyncio.run(main())
