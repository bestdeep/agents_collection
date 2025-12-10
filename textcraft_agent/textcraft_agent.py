"""
TextCraft LLM Agent Implementation

This module provides a simple LLM-based agent for the TextCraft environment.
It uses an OpenAI-compatible API to generate actions for crafting items in a Minecraft-like environment.
"""

import os
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import requests


@dataclass
class TextCraftAgentConfig:
    """Configuration for the TextCraft LLM Agent"""
    
    # API Configuration
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str = field(default="https://api.openai.com/v1")
    model: str = field(default="gpt-3.5-turbo")
    
    # Generation Parameters
    max_tokens: int = field(default=512)
    temperature: float = field(default=0.7)
    top_p: float = field(default=1.0)
    timeout: float = field(default=60.0)
    
    # Agent Parameters
    max_rounds: int = field(default=50)
    max_retries: int = field(default=3)
    retry_delay: float = field(default=1.0)


class TextCraftAgent:
    """
    LLM-based agent for TextCraft environment.
    
    This agent uses an OpenAI-compatible API to generate actions based on
    observations from the TextCraft environment. It maintains conversation history
    and generates thoughtful actions to craft items using given recipes.
    """
    
    SYSTEM_PROMPT = """You are a crafting expert in a Minecraft-like environment. Your goal is to craft items using available recipes and materials. Every round I will give you an observation, and you have to respond with an action and your thought based on the observation to finish the given task.

You can use the following actions:

1. **get <amount> <item>**: Get a basic/raw item that doesn't need to be crafted. For example:
   - get 1 oak log
   - get 2 iron ingot

2. **craft <output_item> using <input_items>**: Craft an item using the specified ingredients. For example:
   - craft 4 oak planks using 1 oak log
   - craft 1 crafting table using 4 oak planks
   - craft 1 iron sword using 2 iron ingots, 1 stick

3. **inventory**: Check your current inventory to see what items you have.

Important rules:
- You can only get items that are basic/raw materials (items that don't appear as outputs in any crafting recipe).
- You can only craft items if you have the exact ingredients required by the recipe.
- Follow the crafting commands provided to you - they specify what items can be crafted and what ingredients are needed.
- Plan your actions carefully. Some items require crafting intermediate items first.
- Check your inventory when needed to track your progress.

Your response should use the following format:

Thought:
<Your reasoning about what to do next>

Action:
<Your action command>"""
    
    def __init__(self, config: Optional[TextCraftAgentConfig] = None):
        """
        Initialize the TextCraft Agent.
        
        Args:
            config: Configuration for the agent. If None, uses default configuration.
        """
        self.config = config or TextCraftAgentConfig()
        self.conversation_history: List[Dict[str, str]] = []
        self.reset_conversation()
        
    def reset_conversation(self):
        """Reset the conversation history to initial state."""
        self.conversation_history = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "assistant", "content": "OK. I'll follow your instructions and try my best to craft the target item."}
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
                    print(f"Rate limited, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                elif attempt == self.config.max_retries - 1:
                    raise
                else:
                    time.sleep(self.config.retry_delay)
                    
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt == self.config.max_retries - 1:
                    raise
                print(f"Request error on attempt {attempt + 1}: {e}")
                time.sleep(self.config.retry_delay)
                
            except Exception as e:
                last_error = e
                if attempt == self.config.max_retries - 1:
                    raise
                print(f"Unexpected error on attempt {attempt + 1}: {e}")
                time.sleep(self.config.retry_delay)
        
        raise RuntimeError(f"Failed after {self.config.max_retries} attempts. Last error: {last_error}")
    
    def parse_action(self, response: str) -> str:
        """
        Extract the action from the agent's response.
        
        Args:
            response: The full response from the LLM
            
        Returns:
            The action command to execute
        """
        # Try to extract action after "Action:" marker
        lines = response.strip().split('\n')
        action_found = False
        action_lines = []
        
        for line in lines:
            if line.strip().lower().startswith('action:'):
                action_found = True
                # Get content after "Action:"
                action_content = line.split(':', 1)[1].strip()
                if action_content:
                    action_lines.append(action_content)
            elif action_found and line.strip():
                # Continue collecting action lines until we hit another section
                if line.strip().lower().startswith(('thought:', 'observation:', 'goal:')):
                    break
                action_lines.append(line.strip())
        
        if action_lines:
            return ' '.join(action_lines)
        
        # Fallback: return the last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        
        return response.strip()
    
    def generate_action(self, observation: str, include_goal: bool = False, max_retries: int = 10) -> tuple[str, str]:
        """
        Generate an action based on the current observation.
        
        Args:
            observation: Current observation from the environment
            include_goal: Whether the observation includes the goal (first step)
            max_retries: Maximum number of retries if action generation fails
            
        Returns:
            Tuple of (full_response, action_command)
        """
        # Add observation to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": observation
        })
        
        # Generate response with retry logic
        for attempt in range(max_retries):
            try:
                response = self._make_api_call(self.conversation_history)
                
                # Validate that response is not empty
                if not response or not response.strip():
                    if attempt < max_retries - 1:
                        print(f"Warning: Empty response from LLM (attempt {attempt + 1}/{max_retries}), retrying...")
                        continue
                    else:
                        # Fallback to a safe default action
                        response = "Thought:\nI need to check my inventory first.\n\nAction:\ninventory"
                        print("Warning: Using fallback action after multiple empty responses")
                
                # Extract action
                action = self.parse_action(response)
                
                # Validate that action was extracted
                if not action or not action.strip():
                    if attempt < max_retries - 1:
                        print(f"Warning: Could not extract action from response (attempt {attempt + 1}/{max_retries}), retrying...")
                        # Remove the bad response and try again
                        self.conversation_history.pop()
                        continue
                    else:
                        # Use fallback action
                        action = "inventory"
                        print("Warning: Using fallback action 'inventory' after failed extraction")
                
                # Add response to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                return response, action
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error generating action (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(self.config.retry_delay)
                else:
                    # Last attempt failed, use fallback
                    print(f"Error generating action after {max_retries} attempts: {e}")
                    response = "Thought:\nI need to check my inventory first.\n\nAction:\ninventory"
                    action = "inventory"
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    return response, action
    
    def save_conversation(self, filepath: str):
        """
        Save the conversation history to a JSON file.
        
        Args:
            filepath: Path to save the conversation history
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TextCraftAgent':
        """
        Create a TextCraft agent from a configuration dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            Initialized TextCraftAgent instance
        """
        config = TextCraftAgentConfig(**config_dict)
        return cls(config)


def main():
    """Example usage of the TextCraft agent."""
    # Example configuration
    config = TextCraftAgentConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model="gpt-3.5-turbo",
        max_tokens=512,
        temperature=0.7,
        max_rounds=50
    )
    
    agent = TextCraftAgent(config)
    
    print("TextCraft Agent initialized successfully!")
    print(f"Model: {config.model}")
    print(f"Max rounds: {config.max_rounds}")
    
    # Example of generating a single action
    example_observation = """Crafting commands:
craft 4 oak planks using 1 oak log
craft 1 crafting table using 4 oak planks
craft 4 sticks using 2 oak planks
craft 1 wooden pickaxe using 3 oak planks, 2 sticks

Goal: craft wooden pickaxe."""
    
    print("\nExample observation:")
    print(example_observation)
    
    print("\nGenerating action...")
    try:
        full_response, action = agent.generate_action(example_observation, include_goal=True)
        print(f"\nGenerated response:\n{full_response}")
        print(f"\nExtracted action: {action}")
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure to set OPENAI_API_KEY environment variable or provide api_key in config")


if __name__ == "__main__":
    main()
