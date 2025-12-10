"""
PrimeIntellect LLM Agent Implementation

This module provides a unified LLM-based agent for all PrimeIntellect environments:
- CDE (Code): Programming challenges with execution-based evaluation
- LGC (Logic): Logic puzzles and reasoning tasks
- MTH (Math): Mathematical problem solving with verification
- SCI (Science): Science questions with grading

The agent uses an OpenAI-compatible API to generate solutions based on task prompts.
"""

import os
import re
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import requests


@dataclass
class PrimeIntellectAgentConfig:
    """Configuration for the PrimeIntellect LLM Agent"""
    
    # API Configuration
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str = field(default="https://api.openai.com/v1")
    model: str = field(default="gpt-4o")
    
    # Generation Parameters
    max_tokens: int = field(default=4096)
    temperature: float = field(default=0.7)
    top_p: float = field(default=1.0)
    timeout: float = field(default=120.0)
    
    # Agent Parameters
    max_retries: int = field(default=3)
    retry_delay: float = field(default=2.0)
    
    # Environment-specific settings
    use_thinking: bool = field(default=True)  # Use <think> tags for reasoning
    verbose: bool = field(default=False)


class PrimeIntellectAgent:
    """
    Unified LLM-based agent for PrimeIntellect environments.
    
    This agent handles multiple task types:
    - Code (CDE): Generates Python code solutions
    - Logic (LGC): Solves logic puzzles and reasoning tasks
    - Math (MTH): Solves mathematical problems with boxed answers
    - Science (SCI): Answers science questions with boxed answers
    """
    
    # Environment-specific system prompts
    SYSTEM_PROMPTS = {
        "cde": """You are an expert Python programmer. You will be given programming challenges that you need to solve by writing Python code.

Your response should contain:
1. A brief thought process about the problem
2. Clean, efficient Python code in a markdown code block

Important guidelines:
- Write complete, executable Python code
- Use proper variable names and comments
- Handle edge cases appropriately
- Optimize for readability and correctness
- Ensure your code is well-structured and follows best practices

Format your code in a Python markdown code block:
```python
# Your code here
```""",
        
        "lgc": """You are an expert in logic, reasoning, and problem-solving. You will be given logic puzzles and reasoning tasks that require careful analysis.

Your response should:
1. Analyze the problem systematically
2. Consider all constraints and conditions
3. Apply logical reasoning step by step
4. Provide a clear, well-reasoned answer

Be precise and methodical in your reasoning. Show your work when helpful.""",
        
        "mth": """You are an expert mathematician. You will be given mathematical problems that you need to solve with detailed reasoning.

Your response should:
1. Understand the problem carefully
2. Explain your reasoning step by step
3. Show all work and calculations
4. Put the final answer in \\boxed{} format

Important:
- Use \\boxed{answer} for your final answer
- Be precise with mathematical notation
- Show intermediate steps
- Verify your answer makes sense

Example format:
Let me solve this step by step...
[Your reasoning]
Therefore, the answer is \\boxed{42}""",
        
        "sci": """You are an expert in science (physics, chemistry, biology, earth science, etc.). You will be given science problems that require deep understanding and problem-solving.

Your response should:
1. Identify relevant scientific concepts
2. Apply appropriate formulas or principles
3. Show your reasoning step by step
4. Put the final answer in \\boxed{} format

Important:
- Use \\boxed{answer} for your final answer
- Be precise with scientific notation and units
- Show calculations when needed
- Verify your answer is reasonable

Example format:
This problem involves...
[Your reasoning and calculations]
Therefore, the answer is \\boxed{9.8 m/s^2}"""
    }
    
    # Default system prompt (uses math as baseline)
    DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPTS["mth"]
    
    def __init__(self, config: Optional[PrimeIntellectAgentConfig] = None):
        """
        Initialize the PrimeIntellect Agent.
        
        Args:
            config: Configuration for the agent. If None, uses default configuration.
        """
        self.config = config or PrimeIntellectAgentConfig()
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}
        
    def reset_conversation(self, env: str = "mth"):
        """
        Reset the conversation history for a specific environment.
        
        Args:
            env: Environment name (cde, lgc, mth, sci)
        """
        system_prompt = self.SYSTEM_PROMPTS.get(env, self.DEFAULT_SYSTEM_PROMPT)
        self.conversation_history[env] = [
            {"role": "system", "content": system_prompt}
        ]
    
    def _make_api_call(self, messages: List[Dict[str, str]]) -> str:
        """
        Make an API call to the LLM service.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Generated response text
            
        Raises:
            Exception: If API call fails after all retries
        """
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        
        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"
        
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                if self.config.verbose:
                    print(f"API call attempt {attempt + 1}/{self.config.max_retries}...")
                
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout
                )
                
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return content.strip()
                
            except requests.exceptions.HTTPError as e:
                last_error = e
                if response.status_code == 429:  # Rate limit
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    if self.config.verbose:
                        print(f"Rate limited, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                elif response.status_code >= 500:  # Server error
                    if attempt < self.config.max_retries - 1:
                        if self.config.verbose:
                            print(f"Server error, retrying...")
                        time.sleep(self.config.retry_delay)
                    else:
                        raise
                else:
                    raise
                    
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt == self.config.max_retries - 1:
                    raise
                if self.config.verbose:
                    print(f"Request error on attempt {attempt + 1}: {e}")
                time.sleep(self.config.retry_delay)
                
            except Exception as e:
                last_error = e
                if attempt == self.config.max_retries - 1:
                    raise
                if self.config.verbose:
                    print(f"Unexpected error on attempt {attempt + 1}: {e}")
                time.sleep(self.config.retry_delay)
        
        # If we get here, all retries failed
        raise Exception(f"All API call attempts failed. Last error: {last_error}")
    
    def solve(
        self, 
        prompt: str, 
        env: str = "mth",
        extra: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a solution for the given prompt.
        
        Args:
            prompt: The problem/task prompt
            env: Environment type (cde, lgc, mth, sci)
            extra: Optional extra information about the task
            
        Returns:
            Generated solution as a string
        """
        # Initialize conversation history for this environment if needed
        if env not in self.conversation_history:
            self.reset_conversation(env)
        
        # Add user prompt to conversation
        user_message = {"role": "user", "content": prompt}
        self.conversation_history[env].append(user_message)
        
        # Generate response
        response = self._make_api_call(self.conversation_history[env])
        
        # Add assistant response to conversation
        assistant_message = {"role": "assistant", "content": response}
        self.conversation_history[env].append(assistant_message)
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Environment: {env.upper()}")
            print(f"{'='*60}")
            print(f"Prompt:\n{prompt}")
            print(f"\n{'-'*60}")
            print(f"Response:\n{response}")
            print(f"{'='*60}\n")
        
        return response
    
    def solve_challenge(self, challenge: Dict[str, Any]) -> str:
        """
        Solve a challenge from the PrimeIntellect environment.
        
        Args:
            challenge: Challenge dictionary with 'env', 'prompt', and optional 'extra' fields
            
        Returns:
            Generated solution as a string
        """
        env = challenge.get("env", "mth")
        prompt = challenge.get("prompt", "")
        extra = challenge.get("extra", {})
        
        return self.solve(prompt, env, extra)
    
    def batch_solve(
        self, 
        challenges: List[Dict[str, Any]], 
        reset_between: bool = False
    ) -> List[str]:
        """
        Solve multiple challenges in batch.
        
        Args:
            challenges: List of challenge dictionaries
            reset_between: Whether to reset conversation between challenges
            
        Returns:
            List of generated solutions
        """
        solutions = []
        
        for i, challenge in enumerate(challenges):
            if self.config.verbose:
                print(f"\nSolving challenge {i+1}/{len(challenges)}...")
            
            solution = self.solve_challenge(challenge)
            solutions.append(solution)
            
            if reset_between:
                env = challenge.get("env", "mth")
                self.reset_conversation(env)
        
        return solutions
    
    def extract_code(self, response: str) -> str:
        """
        Extract Python code from a response (for CDE environment).
        
        Args:
            response: LLM response containing code
            
        Returns:
            Extracted code string
        """
        # Try to find ```python blocks first
        pattern = r"```python\s*(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # Fall back to any ``` blocks
        pattern = r"```\s*(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # If no code blocks, return the response as-is
        return response.strip()
    
    def extract_boxed_answer(self, response: str) -> Optional[str]:
        """
        Extract answer from \\boxed{} format (for MTH and SCI environments).
        
        Args:
            response: LLM response containing boxed answer
            
        Returns:
            Extracted answer or None if not found
        """
        # Match \boxed{...} with proper nesting
        pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
        matches = re.findall(pattern, response)
        if matches:
            return matches[-1].strip()  # Return last match (final answer)
        return None
    
    def clear_all_conversations(self):
        """Clear all conversation histories for all environments."""
        self.conversation_history.clear()
    
    def get_conversation_history(self, env: str) -> List[Dict[str, str]]:
        """
        Get conversation history for a specific environment.
        
        Args:
            env: Environment name
            
        Returns:
            List of conversation messages
        """
        return self.conversation_history.get(env, [])


# Convenience functions for quick usage
def create_agent(
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    base_url: str = "https://api.openai.com/v1",
    **kwargs
) -> PrimeIntellectAgent:
    """
    Create a PrimeIntellect agent with custom configuration.
    
    Args:
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        model: Model name to use
        base_url: API base URL
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured PrimeIntellectAgent instance
    """
    config = PrimeIntellectAgentConfig(
        api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
        model=model,
        base_url=base_url,
        **kwargs
    )
    return PrimeIntellectAgent(config)


def solve_task(
    prompt: str,
    env: str = "mth",
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    **kwargs
) -> str:
    """
    Quick function to solve a single task.
    
    Args:
        prompt: Task prompt
        env: Environment type (cde, lgc, mth, sci)
        api_key: OpenAI API key
        model: Model name
        **kwargs: Additional agent configuration
        
    Returns:
        Generated solution
    """
    agent = create_agent(api_key=api_key, model=model, **kwargs)
    return agent.solve(prompt, env)


# Example usage
if __name__ == "__main__":
    # Example: Solve a math problem
    agent = create_agent(verbose=True)
    
    # Math example
    math_prompt = "What is the sum of the first 10 prime numbers?"
    math_solution = agent.solve(math_prompt, env="mth")
    print("\nMath Solution:")
    print(math_solution)
    print("\nExtracted Answer:", agent.extract_boxed_answer(math_solution))
    
    # Code example
    code_prompt = """Write a Python function that takes a list of integers and returns the second largest number."""
    code_solution = agent.solve(code_prompt, env="cde")
    print("\nCode Solution:")
    print(code_solution)
    print("\nExtracted Code:")
    print(agent.extract_code(code_solution))
    
    # Logic example
    logic_prompt = """Three people (Alice, Bob, and Carol) are sitting in a row. 
    Alice is not on the left. Bob is not in the middle. Carol is not on the right.
    What is the seating arrangement from left to right?"""
    logic_solution = agent.solve(logic_prompt, env="lgc")
    print("\nLogic Solution:")
    print(logic_solution)
    
    # Science example
    sci_prompt = "What is the acceleration due to gravity on Earth in m/s^2?"
    sci_solution = agent.solve(sci_prompt, env="sci")
    print("\nScience Solution:")
    print(sci_solution)
    print("\nExtracted Answer:", agent.extract_boxed_answer(sci_solution))
