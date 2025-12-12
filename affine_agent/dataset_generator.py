#!/usr/bin/env python3
"""
Dataset Generator for Affine Agent

Generates training datasets from successful agent conversations (Score 1.0).
Saves conversations in the format compatible with training datasets.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from agent import AffineAgentConfig, create_agent
from env_integration import AffineEnvironmentAgent
from config_utils import load_config, create_agent_config_from_dict, get_env_configs


class DatasetGenerator:
    """Generate datasets from successful agent conversations."""
    
    def __init__(self, agent_config: AffineAgentConfig, env_configs: Dict[str, Any] = None):
        self.agent_config = agent_config
        self.env_configs = env_configs or {}
        self.env_agent = AffineEnvironmentAgent(agent_config, env_configs)
        
    async def generate_single_entry(self, env: str, task_id: int, verbose: bool = False) -> Dict[str, Any] | None:
        """
        Generate a single dataset entry from a task.
        Returns None if the score is not 1.0, otherwise returns the dataset entry.
        
        Args:
            env: Environment type (abd, ded, sat)
            task_id: Task ID to evaluate
            verbose: Whether to print verbose output
            
        Returns:
            Dataset entry dict or None if score != 1.0
        """
        try:
            # CRITICAL: Reset conversation history to ensure clean evaluation
            # Without this, conversation history from previous tasks pollutes the context
            self.env_agent.agent.reset_conversation(env)
            
            # Evaluate the task
            result = await self.env_agent.solve_and_evaluate(
                env=env,
                task_id=task_id,
                save_results=False
            )
            
            score = result.get('score', 0)
            
            if verbose:
                print(f"Task {task_id}: Score = {score}")
            
            # Only include perfect scores
            if score != 1.0:
                return None
            
            # Build dataset entry in the format matching dataset_sample.json
            challenge = result['challenge']
            response = result['response']
            
            # Create conversation format
            entry = {
                "conversations": [
                    {
                        "from": "human",
                        "loss": None,
                        "value": challenge.prompt
                    },
                    {
                        "from": "gpt",
                        "loss": True,
                        "value": response
                    }
                ],
                "item_id": f"{env}_task_{task_id}"
            }
            
            return entry
            
        except Exception as e:
            if verbose:
                print(f"Error processing task {task_id}: {e}")
            return None
    
    async def generate_dataset(
        self,
        env: str,
        task_ids: List[int],
        output_file: str,
        verbose: bool = False,
        save_interval: int = 10
    ) -> Dict[str, Any]:
        """
        Generate dataset from multiple tasks.
        
        Args:
            env: Environment type (abd, ded, sat)
            task_ids: List of task IDs to process
            output_file: Path to save the dataset JSON file
            verbose: Whether to print verbose output
            save_interval: Save intermediate results every N tasks
            
        Returns:
            Summary statistics
        """
        dataset = []
        failed_task_ids = []
        error_task_ids = []
        successful_count = 0
        failed_count = 0
        error_count = 0
        
        print(f"\n{'='*70}")
        print(f"Generating {env.upper()} Dataset")
        print(f"{'='*70}")
        print(f"Task range: {min(task_ids)} - {max(task_ids)}")
        print(f"Total tasks: {len(task_ids)}")
        print(f"Output file: {output_file}")
        print(f"{'='*70}\n")
        
        for i, task_id in enumerate(task_ids, 1):
            try:
                entry = await self.generate_single_entry(env, task_id, verbose)
                
                if entry is not None:
                    dataset.append(entry)
                    successful_count += 1
                    if verbose:
                        print(f"  ✓ Task {task_id}: Added to dataset (Score: 1.0)")
                else:
                    failed_count += 1
                    failed_task_ids.append(task_id)
                    if verbose:
                        print(f"  ✗ Task {task_id}: Skipped (Score < 1.0)")
                
                # Save intermediate results
                if i % save_interval == 0:
                    self._save_dataset(dataset, output_file)
                    self._save_failed_ids(failed_task_ids, error_task_ids, output_file)
                    print(f"\n[{i}/{len(task_ids)}] Intermediate save: {successful_count} entries saved")
                    print(f"  Success: {successful_count}, Failed: {failed_count}, Errors: {error_count}\n")
                    
            except Exception as e:
                error_count += 1
                error_task_ids.append(task_id)
                if verbose:
                    print(f"  ✗ Task {task_id}: Error - {e}")
        
        # Final save
        self._save_dataset(dataset, output_file)
        self._save_failed_ids(failed_task_ids, error_task_ids, output_file)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"Dataset Generation Complete")
        print(f"{'='*70}")
        print(f"Total processed: {len(task_ids)}")
        print(f"Successful (Score 1.0): {successful_count}")
        print(f"Failed (Score < 1.0): {failed_count}")
        print(f"Errors: {error_count}")
        print(f"Success rate: {successful_count / len(task_ids) * 100:.2f}%")
        print(f"Dataset saved to: {output_file}")
        if failed_task_ids:
            failed_file = output_file.replace('.json', '_failed_ids.json')
            print(f"Failed task IDs saved to: {failed_file}")
        if error_task_ids:
            error_file = output_file.replace('.json', '_error_ids.json')
            print(f"Error task IDs saved to: {error_file}")
        print(f"{'='*70}\n")
        
        return {
            "total_processed": len(task_ids),
            "successful": successful_count,
            "failed": failed_count,
            "errors": error_count,
            "success_rate": successful_count / len(task_ids),
            "output_file": output_file,
            "dataset_size": len(dataset),
            "failed_task_ids": failed_task_ids,
            "error_task_ids": error_task_ids
        }
    
    def _save_dataset(self, dataset: List[Dict[str, Any]], output_file: str):
        """Save dataset to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    def _save_failed_ids(self, failed_ids: List[int], error_ids: List[int], output_file: str):
        """Save failed and error task IDs to separate JSON files."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save failed task IDs (score < 1.0)
        if failed_ids:
            failed_file = str(output_path).replace('.json', '_failed_ids.json')
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "description": "Task IDs that were processed but did not achieve Score 1.0",
                    "count": len(failed_ids),
                    "task_ids": sorted(failed_ids)
                }, f, indent=2)
        
        # Save error task IDs (exceptions during processing)
        if error_ids:
            error_file = str(output_path).replace('.json', '_error_ids.json')
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "description": "Task IDs that encountered errors during processing",
                    "count": len(error_ids),
                    "task_ids": sorted(error_ids)
                }, f, indent=2)


async def generate_dataset_for_range(
    env: str,
    start_id: int,
    end_id: int,
    config_path: str = "config.json",
    output_dir: str = "generated_datasets",
    api_key: str = None,
    model: str = None,
    verbose: bool = False,
    save_interval: int = 10
) -> Dict[str, Any]:
    """
    Generate dataset for a range of task IDs.
    
    Args:
        env: Environment type (abd, ded, sat)
        start_id: Starting task ID (inclusive)
        end_id: Ending task ID (inclusive)
        config_path: Path to config file
        output_dir: Directory to save generated datasets
        api_key: Optional API key override
        model: Optional model override
        verbose: Whether to print verbose output
        save_interval: Save intermediate results every N tasks
        
    Returns:
        Summary statistics
    """
    # Load config
    try:
        config_dict = load_config(config_path)
        agent_config = create_agent_config_from_dict(config_dict)
        env_configs = get_env_configs(config_dict)
    except FileNotFoundError:
        agent_config = AffineAgentConfig(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            model=model or "gpt-4o",
            base_url=os.getenv("BASE_URL") or "https://api.openai.com/v1",
            verbose=verbose
        )
        env_configs = {}
    
    # Override with provided values
    if api_key:
        agent_config.api_key = api_key
    if model:
        agent_config.model = model
    
    # Create generator
    generator = DatasetGenerator(agent_config, env_configs)
    
    # Generate task IDs
    task_ids = list(range(start_id, end_id + 1))
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/{env}_dataset_{start_id}_{end_id}_{timestamp}.json"
    
    # Generate dataset
    results = await generator.generate_dataset(
        env=env,
        task_ids=task_ids,
        output_file=output_file,
        verbose=verbose,
        save_interval=save_interval
    )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate datasets from agent conversations")
    parser.add_argument("--env", required=True, choices=["abd", "ded", "sat"], help="Environment type")
    parser.add_argument("--start-id", type=int, required=True, help="Starting task ID")
    parser.add_argument("--end-id", type=int, required=True, help="Ending task ID")
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--output-dir", default="generated_datasets", help="Output directory")
    parser.add_argument("--api-key", help="API key override")
    parser.add_argument("--model", help="Model override")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--save-interval", type=int, default=10, help="Save interval")
    
    args = parser.parse_args()
    
    asyncio.run(generate_dataset_for_range(
        env=args.env,
        start_id=args.start_id,
        end_id=args.end_id,
        config_path=args.config,
        output_dir=args.output_dir,
        api_key=args.api_key,
        model=args.model,
        verbose=args.verbose,
        save_interval=args.save_interval
    ))
