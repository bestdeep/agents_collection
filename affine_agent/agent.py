"""
Affine LLM Agent Implementation

This module provides a unified LLM-based agent for Affine environments:
- ABD (Algorithm By Deduction): Reverse engineering program inputs from outputs
- DED (Direct Execution Debug): Python program generation from requirements

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
class AffineAgentConfig:
    """Configuration for the Affine LLM Agent"""
    
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
    verbose: bool = field(default=False)


class AffineAgent:
    """
    Unified LLM-based agent for Affine environments.
    
    This agent handles multiple task types:
    - ABD (Algorithm By Deduction): Reverse engineering - given a program and its output, deduce the input
    - DED (Direct Execution Debug): Code generation - write Python programs that solve given requirements
    """
    
    # Environment-specific system prompts
    SYSTEM_PROMPTS = {
        "sat": """You are an expert in propositional logic and satisfiability problems. You will be given k-SAT formulas and need to find satisfying assignments.

Your task is to:
1. Understand the SAT formula with its variables and clauses
2. Find a valid assignment of True/False values to all variables
3. Verify that the assignment satisfies all clauses
4. Provide the complete assignment

Important guidelines:
- Each clause must evaluate to True (at least one literal must be True)
- Use systematic reasoning or trial-and-error to find valid assignments
- Double-check your assignment against all clauses
- Provide the complete assignment for all variables

Format your answer as comma-separated assignments:
x1=True, x2=False, x3=True, ...

If you determine the formula is unsatisfiable, respond with: UNSAT""",
        
        "abd": """You are a programming expert specialized in reverse engineering and program analysis. You will be given a Python program and its expected output, and you need to determine the exact input that would produce this output.

Your task is to:
1. Carefully analyze the program's logic and control flow
2. Understand what input format the program expects from stdin
3. Work backwards from the output to deduce the input
4. Provide the exact input data that would produce the given output

Important guidelines:
- Trace through the program logic step by step
- Consider all input operations (input(), sys.stdin, etc.)
- Think about data types, formatting, and edge cases
- The input should be exactly what would be fed to stdin
- Each line of input should be on a separate line

You MUST provide your final answer within <INPUT> </INPUT> tags with the exact input data:
<INPUT>
[input data here - each line on a separate line as the program expects]
</INPUT>

Example:
Given a program that reads two numbers and outputs their sum=15, you might deduce:
<INPUT>
7
8
</INPUT>""",
        
        "ded": """You are an expert Python programmer. You will be given programming challenges that describe requirements, and you need to write complete Python programs that solve them.

Your task is to:
1. Read and understand the problem requirements carefully
2. Identify what inputs the program needs to read from STDIN
3. Determine what outputs need to be written to STDOUT
4. Write clean, efficient, correct Python code
5. Ensure the program handles all test cases properly

Important guidelines:
- Write complete, executable Python 3 code
- Read ALL input from STDIN using input() or sys.stdin
- Write ONLY the required output to STDOUT using print()
- Do NOT include any debug output, prompts, or extra text
- Handle edge cases and follow the requirements exactly
- Use clear variable names and comments when helpful
- Test your logic mentally before finalizing

You MUST provide your code in a Python markdown code block:
```python
# Your complete solution here
```

The code will be extracted and executed with test cases to verify correctness."""
    }
    
    # Default system prompt
    DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPTS["ded"]
    
    def __init__(self, config: Optional[AffineAgentConfig] = None):
        """
        Initialize the Affine Agent.
        
        Args:
            config: Configuration for the agent. If None, uses default configuration.
        """
        self.config = config or AffineAgentConfig()
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}
        
    def reset_conversation(self, env: str = "ded"):
        """
        Reset the conversation history for a specific environment.
        
        Args:
            env: Environment name (sat, abd, ded)
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
                    if self.config.verbose:
                        print(f"Rate limited, waiting {self.config.retry_delay}s...")
                    time.sleep(self.config.retry_delay)
                    continue
                elif response.status_code >= 500:  # Server error
                    if self.config.verbose:
                        print(f"Server error, retrying in {self.config.retry_delay}s...")
                    time.sleep(self.config.retry_delay)
                    continue
                else:
                    raise
                    
            except requests.exceptions.Timeout as e:
                last_error = e
                if self.config.verbose:
                    print(f"Timeout, retrying in {self.config.retry_delay}s...")
                time.sleep(self.config.retry_delay)
                continue
                
            except Exception as e:
                last_error = e
                if self.config.verbose:
                    print(f"Error: {e}, retrying in {self.config.retry_delay}s...")
                time.sleep(self.config.retry_delay)
                continue
        
        raise Exception(f"API call failed after {self.config.max_retries} attempts: {last_error}")
    
    def solve(self, prompt: str, env: str = "ded") -> str:
        """
        Solve a challenge using the LLM.
        
        Args:
            prompt: Challenge prompt/description
            env: Environment name (sat, abd, ded)
            
        Returns:
            Generated solution as text
        """
        # Initialize conversation if needed
        if env not in self.conversation_history:
            self.reset_conversation(env)
        
        # Add user message
        self.conversation_history[env].append({
            "role": "user",
            "content": prompt
        })
        
        # Get response
        response = self._make_api_call(self.conversation_history[env])
        
        # Add assistant response to history
        self.conversation_history[env].append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def solve_challenge(self, challenge: Dict[str, Any]) -> str:
        """
        Solve a challenge object.
        
        Args:
            challenge: Challenge dictionary with 'prompt' and 'env' keys
            
        Returns:
            Generated solution
        """
        env = challenge.get("env", "ded")
        # Extract base environment name (e.g., "affine:ded" -> "ded")
        if ":" in env:
            env = env.split(":")[-1]
        
        prompt = challenge.get("prompt", "")
        return self.solve(prompt, env)
    
    def extract_sat_assignment(self, response: str) -> Optional[str]:
        """
        Extract SAT assignment from response.
        
        Args:
            response: Agent's response text
            
        Returns:
            Extracted assignment string or None if not found
        """
        # Check for UNSAT
        if "UNSAT" in response.upper():
            return "UNSAT"
        
        # Look for assignment pattern like "x1=True, x2=False, ..."
        # Find all variable assignments
        import re
        assignments = re.findall(r'x\d+=(?:True|False)', response)
        
        if assignments:
            return ", ".join(assignments)
        
        return None
    
    def extract_input(self, response: str) -> Optional[str]:
        """
        Extract input data from ABD response within <INPUT></INPUT> tags.
        
        Args:
            response: Agent's response text
            
        Returns:
            Extracted input data or None if not found
        """
        # Look for content between <INPUT> and </INPUT> tags
        pattern = r'<INPUT>(.*?)</INPUT>'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # Return the first match, stripped of leading/trailing whitespace
            return matches[0].strip()
        
        return None
    
    def extract_code(self, response: str) -> Optional[str]:
        """
        Extract Python code from DED response in markdown code blocks.
        
        Args:
            response: Agent's response text
            
        Returns:
            Extracted code or None if not found
        """
        # Look for Python code blocks
        patterns = [
            r'```python\n(.*?)```',
            r'```Python\n(.*?)```',
            r'```\n(.*?)```',  # Generic code block
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return matches[0].strip()
        
        return None
    
    def get_conversation_history(self, env: str) -> List[Dict[str, str]]:
        """
        Get the conversation history for a specific environment.
        
        Args:
            env: Environment name
            
        Returns:
            List of message dictionaries
        """
        return self.conversation_history.get(env, [])
    
    def clear_all_conversations(self):
        """Clear all conversation histories."""
        self.conversation_history.clear()


def create_agent(
    api_key: Optional[str] = None,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o",
    temperature: float = 0.7,
    verbose: bool = False,
    **kwargs
) -> AffineAgent:
    """
    Convenience function to create an Affine agent with common parameters.
    
    Args:
        api_key: API key for LLM service
        base_url: Base URL for API endpoint
        model: Model name to use
        temperature: Sampling temperature
        verbose: Enable verbose logging
        **kwargs: Additional config parameters
        
    Returns:
        Configured AffineAgent instance
    """
    config = AffineAgentConfig(
        api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
        base_url=base_url,
        model=model,
        temperature=temperature,
        verbose=verbose,
        **kwargs
    )
    return AffineAgent(config)


if __name__ == "__main__":
    # Example usage
    agent = create_agent(verbose=True)
    
    # Example ABD task
    abd_prompt = """You are given this Python program:
```python
x = int(input())
y = int(input())
print(x + y)
```

And this expected output:
```
15
```

What input would produce this output?"""
    
    abd_solution = agent.solve(abd_prompt, env="abd")
    print("ABD Solution:")
    print(abd_solution)
    print("\nExtracted Input:")
    print(agent.extract_input(abd_solution))
    
    # Example DED task
    ded_prompt = """Write a Python program that reads two integers from stdin and outputs their sum."""
    
    ded_solution = agent.solve(ded_prompt, env="ded")
    print("\nDED Solution:")
    print(ded_solution)
    print("\nExtracted Code:")
    print(agent.extract_code(ded_solution))
