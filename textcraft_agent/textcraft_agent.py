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
    
    def generate_action(self, observation: str, include_goal: bool = False) -> tuple[str, str]:
        """
        Generate an action based on the current observation.
        
        Args:
            observation: Current observation from the environment
            include_goal: Whether the observation includes the goal (first step)
            
        Returns:
            Tuple of (full_response, action_command)
        """
        # Add observation to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": observation
        })
        
        # Generate response
        response = self._make_api_call(self.conversation_history)
        
        # Add response to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        # Extract action
        action = self.parse_action(response)
        
        return response, action
    
    def run_episode(self, env_url: str, env_id: int, data_idx: int, verbose: bool = True) -> Dict:
        """
        Run a complete episode in the TextCraft environment.
        
        Args:
            env_url: URL of the TextCraft environment server
            env_id: Environment instance ID
            data_idx: Data index to reset to
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing episode results including:
            - success: Whether the task was completed
            - reward: Final reward
            - steps: Number of steps taken
            - conversation: Full conversation history
        """
        # Reset environment and conversation
        self.reset_conversation()
        
        # Make reset request
        reset_response = requests.post(
            f"{env_url}/reset",
            json={"id": env_id, "data_idx": data_idx}
        ).json()
        
        observation = reset_response["observation"]
        done = reset_response.get("done", False)
        reward = reset_response.get("reward", 0)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Episode {data_idx} - Environment {env_id}")
            print(f"{'='*60}")
            print(f"\nInitial Observation:\n{observation}\n")
        
        step_count = 0
        episode_history = []
        
        # Run episode
        while not done and step_count < self.config.max_rounds:
            # Generate action
            try:
                is_first_step = (step_count == 0)
                full_response, action = self.generate_action(observation, include_goal=is_first_step)
                
                if verbose:
                    print(f"\n--- Step {step_count + 1} ---")
                    print(f"Agent Response:\n{full_response}")
                    print(f"\nExecuting Action: {action}")
                
                # Take step in environment
                step_response = requests.post(
                    f"{env_url}/step",
                    json={"id": env_id, "action": action}
                ).json()
                
                if "error" in step_response:
                    print(f"Error: {step_response['error']}")
                    observation = f"Error: {step_response['error']}"
                    reward = step_response.get("reward", 0)
                    done = step_response.get("done", False)
                else:
                    observation = step_response["observation"]
                    reward = step_response.get("reward", 0)
                    done = step_response.get("done", False)
                
                if verbose:
                    print(f"Observation: {observation}")
                    print(f"Reward: {reward}, Done: {done}")
                
                # Record step
                episode_history.append({
                    "step": step_count,
                    "thought_and_action": full_response,
                    "action": action,
                    "observation": observation,
                    "reward": reward,
                    "done": done
                })
                
                step_count += 1
                
            except Exception as e:
                print(f"Error during episode: {e}")
                import traceback
                traceback.print_exc()
                break
        
        success = reward >= 1.0
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Episode Complete!")
            print(f"Success: {success}, Reward: {reward}, Steps: {step_count}")
            print(f"{'='*60}\n")
        
        # Return episode results
        return {
            "success": success,
            "reward": reward,
            "steps": step_count,
            "done": done,
            "conversation": self.conversation_history,
            "history": episode_history
        }
    
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
