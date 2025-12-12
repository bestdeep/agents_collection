"""
Result Saver for Affine Agent

Utilities for saving evaluation results to JSON files.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


class ResultSaver:
    """Save and manage evaluation results."""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize result saver.
        
        Args:
            output_dir: Directory to save results (default: results/)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_evaluation(
        self,
        result: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        extracted_answer: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Save a single evaluation result.
        
        Args:
            result: Evaluation result dictionary
            conversation_history: Optional conversation history
            extracted_answer: Optional extracted answer
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        env = result.get("env", "unknown")
        # Handle env with prefix like "affine:ded" -> "ded"
        if ":" in env:
            env = env.split(":")[-1]
        
        task_id = result.get("task_id", 0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename is None:
            filename = f"{env}_task{task_id}_{timestamp}.json"
        
        # Build save data
        save_data = {
            "metadata": {
                "env": env,
                "task_id": task_id,
                "timestamp": timestamp,
                "datetime": datetime.now().isoformat()
            },
            "challenge": {
                "prompt": result["challenge"].prompt if hasattr(result["challenge"], "prompt") else str(result["challenge"]),
                "extra": result["challenge"].extra if hasattr(result["challenge"], "extra") else {}
            },
            "response": result["response"],
            "score": str(result["score"]),  # Convert to string for serialization
            "extracted_answer": extracted_answer,
            "conversation_history": conversation_history or []
        }
        
        # Add ground truth if available
        if hasattr(result["challenge"], "extra"):
            extra = result["challenge"].extra
            if "solution" in extra:
                save_data["ground_truth"] = extra["solution"]
            elif "expected_output" in extra:
                save_data["ground_truth"] = extra["expected_output"]
        
        # Save to file
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def save_benchmark(
        self,
        benchmark_results: Dict[str, Any],
        all_conversations: Optional[Dict[int, List[Dict[str, str]]]] = None,
        all_extracted_answers: Optional[Dict[int, str]] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Save benchmark results.
        
        Args:
            benchmark_results: Benchmark result dictionary
            all_conversations: Optional dict of task_id -> conversation history
            all_extracted_answers: Optional dict of task_id -> extracted answer
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        env = benchmark_results.get("env", "unknown")
        if ":" in env:
            env = env.split(":")[-1]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename is None:
            filename = f"{env}_benchmark_{timestamp}.json"
        
        # Build save data
        save_data = {
            "metadata": {
                "env": env,
                "num_tasks": benchmark_results["num_tasks"],
                "timestamp": timestamp,
                "datetime": datetime.now().isoformat()
            },
            "summary": {
                "average_score": benchmark_results.get("average_score", 0.0),
                "success_rate": benchmark_results.get("success_rate", 0.0),
                "scores": benchmark_results.get("scores", [])
            },
            "results": []
        }
        
        # Add detailed results
        for i, result in enumerate(benchmark_results.get("results", [])):
            task_id = result.get("task_id", i)
            
            task_data = {
                "task_id": task_id,
                "score": str(result.get("score", 0.0)),
                "error": result.get("error")
            }
            
            # Add challenge if available
            if "challenge" in result:
                task_data["challenge"] = {
                    "prompt": result["challenge"].prompt if hasattr(result["challenge"], "prompt") else "",
                    "extra": result["challenge"].extra if hasattr(result["challenge"], "extra") else {}
                }
            
            # Add response if available
            if "response" in result:
                task_data["response"] = result["response"]
            
            # Add conversation history if available
            if all_conversations and task_id in all_conversations:
                task_data["conversation_history"] = all_conversations[task_id]
            
            # Add extracted answer if available
            if all_extracted_answers and task_id in all_extracted_answers:
                task_data["extracted_answer"] = all_extracted_answers[task_id]
            
            save_data["results"].append(task_data)
        
        # Save to file
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def load_evaluation(self, filepath: str) -> Dict[str, Any]:
        """
        Load a saved evaluation result.
        
        Args:
            filepath: Path to saved file
            
        Returns:
            Loaded evaluation data
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_saved_results(self, env: Optional[str] = None) -> List[str]:
        """
        List all saved result files.
        
        Args:
            env: Optional environment filter
            
        Returns:
            List of result file paths
        """
        pattern = f"{env}_*.json" if env else "*.json"
        return sorted([str(f) for f in self.output_dir.glob(pattern)])
    
    def get_statistics(self, env: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics from saved results.
        
        Args:
            env: Optional environment filter
            
        Returns:
            Statistics dictionary
        """
        files = self.list_saved_results(env)
        
        if not files:
            return {
                "total_files": 0,
                "environments": {},
                "overall_stats": {}
            }
        
        stats = {
            "total_files": len(files),
            "environments": {},
            "overall_stats": {
                "total_tasks": 0,
                "total_score": 0.0,
                "scores": []
            }
        }
        
        for filepath in files:
            try:
                data = self.load_evaluation(filepath)
                file_env = data.get("metadata", {}).get("env", "unknown")
                score_str = data.get("score", "0.0")
                
                # Parse score
                try:
                    if isinstance(score_str, str):
                        if score_str.startswith("("):
                            score_str = score_str.split(",")[0].strip("(")
                        score = float(score_str)
                    else:
                        score = float(score_str)
                except:
                    score = 0.0
                
                # Update environment stats
                if file_env not in stats["environments"]:
                    stats["environments"][file_env] = {
                        "count": 0,
                        "total_score": 0.0,
                        "scores": []
                    }
                
                stats["environments"][file_env]["count"] += 1
                stats["environments"][file_env]["total_score"] += score
                stats["environments"][file_env]["scores"].append(score)
                
                # Update overall stats
                stats["overall_stats"]["total_tasks"] += 1
                stats["overall_stats"]["total_score"] += score
                stats["overall_stats"]["scores"].append(score)
                
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
        
        # Calculate averages
        if stats["overall_stats"]["total_tasks"] > 0:
            stats["overall_stats"]["average_score"] = (
                stats["overall_stats"]["total_score"] / stats["overall_stats"]["total_tasks"]
            )
        
        for env_stats in stats["environments"].values():
            if env_stats["count"] > 0:
                env_stats["average_score"] = env_stats["total_score"] / env_stats["count"]
        
        return stats


async def save_result(result: Dict[str, Any], output_dir: str):
    """
    Save evaluation result to a JSON file (legacy function for backward compatibility).
    
    Args:
        result: Result dictionary with challenge, response, score, etc.
        output_dir: Directory to save results
    """
    saver = ResultSaver(output_dir)
    filepath = saver.save_evaluation(
        result,
        result.get("conversation_history", []),
        result.get("extracted_answer")
    )
    print(f"Result saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    # Get statistics from saved results
    saver = ResultSaver("results")
    stats = saver.get_statistics()
    
    print("Result Statistics:")
    print(json.dumps(stats, indent=2))

