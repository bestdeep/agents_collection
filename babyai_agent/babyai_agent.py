"""
BabyAI LLM Agent Implementation

This module provides a simple LLM-based agent for the BabyAI environment.
It uses an OpenAI-compatible API to generate actions based on observations.
"""

import os
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import requests


@dataclass
class BabyAIAgentConfig:
    """Configuration for the BabyAI LLM Agent"""
    
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
    max_rounds: int = field(default=20)
    max_retries: int = field(default=3)
    retry_delay: float = field(default=1.0)


class BabyAIAgent:
    """
    LLM-based agent for BabyAI environment.
    
    This agent uses an OpenAI-compatible API to generate actions based on
    observations from the BabyAI environment. It maintains conversation history
    and generates thoughtful actions to accomplish given tasks.
    """
    
    SYSTEM_PROMPT = """You are an exploration master that wants to finish every goal you are given. Every round I will give you an observation, and you have to respond an action and your thought based on the observation to finish the given task. You are placed in a room and you need to accomplish the given goal with actions.

You can use the following actions:

- turn right
- turn left
- move forward
- go to <obj> <id>
- pick up <obj> <id>
- go through <door> <id>: <door> must be an open door.
- toggle and go through <door> <id>: <door> can be a closed door or a locked door. If you want to open a locked door, you need to carry a key that is of the same color as the locked door.
- toggle: there is a closed or locked door right in front of you and you can toggle it.

Your response should use the following format:

Thought:
<Your Thought>

Action:
<Your Action>"""
    
    def __init__(self, config: Optional[BabyAIAgentConfig] = None):
        """
        Initialize the BabyAI Agent.
        
        Args:
            config: Configuration for the agent. If None, uses default configuration.
        """
        self.config = config or BabyAIAgentConfig()
        self.conversation_history: List[Dict[str, str]] = []
        self.reset_conversation()
        
    def reset_conversation(self):
        """Reset the conversation history to initial state."""
        self.conversation_history = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "assistant", "content": "OK. I'll follow your instructions and try my best to solve the task."}
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
    
    def generate_action(self, observation: str, goal: Optional[str] = None) -> str:
        """
        Generate an action based on the current observation and goal.
        
        Args:
            observation: Current observation from the environment
            goal: Optional goal description (included in first observation)
            
        Returns:
            Generated action string in the format:
            Thought: <thought>
            Action: <action>
        """
        # Construct the user message
        if goal:
            user_message = f"Goal: {goal}\n\nObservation: {observation}"
        else:
            user_message = f"Observation: {observation}"
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Generate response
        response = self._make_api_call(self.conversation_history)
        
        # Add response to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def run_episode(self, env_client, env_idx: str, data_idx: int) -> Dict:
        """
        Run a complete episode in the BabyAI environment.
        
        Args:
            env_client: BabyAI environment client
            env_idx: Environment instance ID
            data_idx: Data index to reset to
            
        Returns:
            Dictionary containing episode results including:
            - success: Whether the task was completed
            - reward: Final reward
            - steps: Number of steps taken
            - conversation: Full conversation history
        """
        # Reset environment and conversation
        self.reset_conversation()
        reset_response = env_client.reset(env_idx, data_idx)
        
        observation = reset_response["observation"]
        done = reset_response["done"]
        reward = reset_response["score"]
        
        # Extract goal from observation if present
        goal = None
        if "Goal:" in observation or "goal:" in observation:
            # Goal is typically included in the first observation
            goal = observation
        
        step_count = 0
        episode_history = []
        
        # Run episode
        while not done and step_count < self.config.max_rounds:
            # Generate action
            try:
                if step_count == 0 and goal:
                    action_response = self.generate_action(observation, goal=goal)
                else:
                    action_response = self.generate_action(observation)
                
                # Take step in environment
                step_output = env_client.step(env_idx, action_response)
                
                # Record step
                episode_history.append({
                    "step": step_count,
                    "observation": observation,
                    "action": action_response,
                    "reward": step_output.reward,
                    "done": step_output.done
                })
                
                # Update state
                observation = step_output.state
                reward = step_output.reward
                done = step_output.done
                step_count += 1
                
            except Exception as e:
                print(f"Error during episode: {e}")
                break
        
        # Return episode results
        return {
            "success": reward >= 0.0,
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
    def from_dict(cls, config_dict: Dict) -> 'BabyAIAgent':
        """
        Create a BabyAI agent from a configuration dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            Initialized BabyAIAgent instance
        """
        config = BabyAIAgentConfig(**config_dict)
        return cls(config)


def main():
    """Example usage of the BabyAI agent."""
    # Example configuration
    config = BabyAIAgentConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model="gpt-3.5-turbo",
        max_tokens=512,
        temperature=0.7,
        max_rounds=20
    )
    
    agent = BabyAIAgent(config)
    
    print("BabyAI Agent initialized successfully!")
    print(f"Model: {config.model}")
    print(f"Max rounds: {config.max_rounds}")
    
    # Example of generating a single action
    example_observation = """Goal: Go to the red ball.

In front of you in this room, you can see several objects: There is a red ball 1 right in front of you 3 steps away."""
    
    print("\nExample observation:")
    print(example_observation)
    
    print("\nGenerating action...")
    try:
        action = agent.generate_action(example_observation, goal="Go to the red ball")
        print(f"\nGenerated action:\n{action}")
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure to set OPENAI_API_KEY environment variable or provide api_key in config")


if __name__ == "__main__":
    main()
