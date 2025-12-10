"""
Result Saver - Save conversation history, scores, and answers

This module provides utilities to save evaluation results including:
- Conversation history
- Scores
- Extracted answers
- Full evaluation details
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


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
            "score": result["score"],
            "extracted_answer": extracted_answer,
            "conversation_history": conversation_history or []
        }
        
        # Add ground truth answer if available
        if hasattr(result["challenge"], "extra") and "answer" in result["challenge"].extra:
            save_data["ground_truth"] = result["challenge"].extra["answer"]
        
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
                "average_score": benchmark_results["average_score"],
                "success_rate": benchmark_results["success_rate"],
                "scores": benchmark_results["scores"]
            },
            "results": []
        }
        
        # Add detailed results
        for i, result in enumerate(benchmark_results["results"]):
            task_id = result.get("task_id", i)
            
            task_data = {
                "task_id": task_id,
                "score": result.get("score", 0.0),
                "challenge": {
                    "prompt": result["challenge"].prompt if "challenge" in result and hasattr(result["challenge"], "prompt") else "",
                    "extra": result["challenge"].extra if "challenge" in result and hasattr(result["challenge"], "extra") else {}
                },
                "response": result.get("response", ""),
                "error": result.get("error")
            }
            
            # Add conversation history if available
            if all_conversations and task_id in all_conversations:
                task_data["conversation_history"] = all_conversations[task_id]
            
            # Add extracted answer if available
            if all_extracted_answers and task_id in all_extracted_answers:
                task_data["extracted_answer"] = all_extracted_answers[task_id]
            
            # Add ground truth if available
            if "challenge" in result and hasattr(result["challenge"], "extra") and "answer" in result["challenge"].extra:
                task_data["ground_truth"] = result["challenge"].extra["answer"]
            
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
            return {"total_evaluations": 0}
        
        total_score = 0.0
        total_count = 0
        success_count = 0
        
        for filepath in files:
            try:
                data = self.load_evaluation(filepath)
                if "score" in data:
                    score = data["score"]
                    total_score += score
                    total_count += 1
                    if score > 0:
                        success_count += 1
                elif "summary" in data:  # Benchmark file
                    # Skip benchmark files in single result stats
                    continue
            except:
                continue
        
        if total_count == 0:
            return {"total_evaluations": 0}
        
        return {
            "total_evaluations": total_count,
            "average_score": total_score / total_count,
            "success_rate": success_count / total_count,
            "total_successes": success_count
        }


# Convenience function
def save_result(
    result: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    extracted_answer: Optional[str] = None,
    output_dir: str = "results"
) -> str:
    """
    Quick function to save a result.
    
    Args:
        result: Evaluation result
        conversation_history: Optional conversation history
        extracted_answer: Optional extracted answer
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    saver = ResultSaver(output_dir)
    return saver.save_evaluation(result, conversation_history, extracted_answer)
