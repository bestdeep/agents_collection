"""
PrimeIntellect Agent - Environment Integration

This module provides integration with the PrimeIntellect task environments,
allowing the agent to work seamlessly with CDE, LGC, MTH, and SCI tasks.
"""

import sys
import gc
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any

# Add environment paths
sys.path.insert(0, '/home/120/affinetes/environments/primeintellect/cde')
sys.path.insert(0, '/home/120/affinetes/environments/primeintellect/lgc')
sys.path.insert(0, '/home/120/affinetes/environments/primeintellect/mth')
sys.path.insert(0, '/home/120/affinetes/environments/primeintellect/sci')

from agent import PrimeIntellectAgent, PrimeIntellectAgentConfig

# Apply safety patches to limit resource usage
try:
    from safe_eval_patch import apply_patch
    apply_patch()
except Exception as e:
    print(f"Warning: Could not apply safety patches: {e}")


# Global task generator cache to avoid reloading datasets
_TASK_GENERATOR_CACHE = {}

# Known problematic task IDs that cause system kills
_PROBLEMATIC_TASKS = {
    "cde": [5],  # Task 5 causes system kill during evaluation
}


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
        Uses global cache to avoid reloading datasets.
        
        Args:
            env: Environment name (cde, lgc, mth, sci)
        """
        # Check global cache first
        if env in _TASK_GENERATOR_CACHE:
            self.task_generators[env] = _TASK_GENERATOR_CACHE[env]
            return
            
        if env in self.task_generators:
            return
        
        env_config = self.env_configs.get(env, {})
        
        if env == "cde":
            from code_task import CodeTask
            task_gen = CodeTask(**env_config)
        elif env == "lgc":
            from logic_task import LogicTask
            task_gen = LogicTask(**env_config)
        elif env == "mth":
            from math_task import MathTask
            task_gen = MathTask(**env_config)
        elif env == "sci":
            from sci_task import ScienceTask
            task_gen = ScienceTask(**env_config)
        else:
            raise ValueError(f"Unknown environment: {env}")
        
        # Cache globally and locally
        _TASK_GENERATOR_CACHE[env] = task_gen
        self.task_generators[env] = task_gen
    
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
        save_results: bool = False,
        output_dir: str = "results",
        **eval_kwargs
    ) -> Dict[str, Any]:
        """
        Generate a task, solve it with the agent, and evaluate the solution.
        
        Args:
            env: Environment name
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
        # final_answer = challenge.extra["answer"]
        # response = f"Final Answer: The final answer is $\\boxed{final_answer}$"
        
        # Get conversation history
        conversation_history = self.agent.get_conversation_history(env)
        
        # Extract answer based on environment type
        extracted_answer = None
        if env in ["mth", "sci"]:
            extracted_answer = self.agent.extract_boxed_answer(response)
        elif env == "cde":
            extracted_answer = self.agent.extract_code(response)
        
        # Evaluate response
        task_gen = self.task_generators[env]
        
        try:
            print(f"[DEBUG] Starting evaluation for {env} task {task_id}...")
            
            # For CDE tasks, add a safety timeout wrapper
            if env == "cde":
                async def eval_with_safety():
                    try:
                        return await asyncio.wait_for(
                            task_gen.evaluate(response, challenge, **eval_kwargs),
                            timeout=60  # 60 second timeout for entire evaluation
                        )
                    except asyncio.TimeoutError:
                        print(f"[WARNING] Evaluation timed out after 60s")
                        return (0.0, "0/0 (timeout)")
                
                score = await eval_with_safety()
            elif env in ["mth", "sci"]:
                # Math and science may use judge models
                score = await task_gen.evaluate(response, challenge, **eval_kwargs)
            else:  # lgc
                score = await task_gen.evaluate(response, challenge)
            
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
            from result_saver import save_result
            filepath = save_result(result, conversation_history, extracted_answer, output_dir)
            result["saved_to"] = filepath
        
        return result
    
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
            env: Environment name
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
            # Check if this is a known problematic task
            if env in _PROBLEMATIC_TASKS and task_id in _PROBLEMATIC_TASKS[env]:
                print(f"⚠ Task {task_id} is known to cause issues - automatically skipping")
                skipped_tasks.append(task_id)
                task_id += 1
                continue
            
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
    base_url:Optional[str] = None,
    model: str = "gpt-4o",
    verbose: bool = True,
    save_results: bool = False,
    output_dir: str = "results",
    save_interval: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick function to evaluate an agent on a specific environment.
    
    Args:
        env: Environment name (cde, lgc, mth, sci)
        num_tasks: Number of tasks to evaluate
        api_key: OpenAI API key
        base_url: OpenAI Endpoint url
        model: Model name
        verbose: Print progress
        save_results: Whether to save results to file
        output_dir: Directory to save results
        save_interval: Save intermediate results every N tasks
        **kwargs: Additional configuration
        
    Returns:
        Benchmark results dictionary
    """
    agent_config = PrimeIntellectAgentConfig(
        api_key=api_key,
        model=model,
        verbose=verbose,
        base_url=base_url,
        **kwargs
    )
    env_agent = PrimeIntellectEnvironmentAgent(agent_config)
    return await env_agent.run_benchmark(
        env, 
        num_tasks, 
        save_results=save_results, 
        output_dir=output_dir,
        save_interval=save_interval
    )


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
