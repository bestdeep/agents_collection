"""
Human-in-the-Loop BabyAI Agent

This agent allows human intervention when the LLM fails to generate valid actions.
The human can provide actions via command line input.
"""

import os
import json
import sys
from typing import Dict, Optional
from pathlib import Path
from dotenv import load_dotenv
# Import the base agent
from babyai_agent import BabyAIAgent, BabyAIAgentConfig

print(Path(__file__).parent.parent / ".env")
load_dotenv(Path(__file__).parent.parent / ".env")

class HumanBabyAIAgent(BabyAIAgent):
    """
    BabyAI agent with human-in-the-loop capability.
    
    When the LLM fails to generate a valid action, the agent prompts
    the human to provide an action via command line input.
    """
    
    def __init__(self, config: Optional[BabyAIAgentConfig] = None, human_mode: str = "fallback"):
        """
        Initialize the Human-in-the-Loop BabyAI Agent.
        
        Args:
            config: Configuration for the agent
            human_mode: When to use human input:
                - "fallback": Only when LLM fails (default)
                - "always": Always ask human for confirmation
                - "hybrid": LLM generates, human can override
        """
        super().__init__(config)
        self.human_mode = human_mode
        self.human_actions_count = 0
        self.llm_actions_count = 0
        
    def get_human_action(self, observation: str, llm_action: Optional[str] = None) -> str:
        """
        Get action from human via command line input.
        
        Args:
            observation: Current observation
            llm_action: LLM's proposed action (if any)
            
        Returns:
            Human-provided action
        """
        print("\n" + "="*70)
        print("HUMAN INPUT REQUIRED")
        print("="*70)
        print(f"\nCurrent Observation:\n{observation}\n")
        
        if llm_action:
            print(f"LLM's Proposed Action:\n{llm_action}\n")
        
        print("Available Actions:")
        print("  - turn right")
        print("  - turn left")
        print("  - move forward")
        print("  - go to <obj> <id>")
        print("  - pick up <obj> <id>")
        print("  - go through <door> <id>")
        print("  - toggle and go through <door> <id>")
        print("  - toggle")
        print("\nSpecial Commands:")
        print("  - 'accept' (accept LLM's action if shown)")
        print("  - 'skip' (use fallback action: move forward)")
        print("  - 'quit' (end episode)")
        print("="*70)
        
        while True:
            try:
                user_input = input("\nYour action: ").strip()
                
                if not user_input:
                    print("Please enter an action.")
                    continue
                
                if user_input.lower() == 'quit':
                    print("Episode terminated by user.")
                    return "QUIT"
                
                if user_input.lower() == 'skip':
                    print("Using fallback action: move forward")
                    return "Thought:\nHuman chose to skip.\n\nAction:\nmove forward"
                
                if user_input.lower() == 'accept' and llm_action:
                    print("Accepting LLM's action.")
                    return llm_action
                
                if user_input.lower() == 'accept' and not llm_action:
                    print("No LLM action to accept. Please enter an action.")
                    continue
                
                # Format the action
                formatted_action = f"Thought:\nHuman intervention.\n\nAction:\n{user_input}"
                
                confirm = input(f"\nConfirm action '{user_input}'? (y/n): ").strip().lower()
                if confirm == 'y' or confirm == 'yes':
                    self.human_actions_count += 1
                    return formatted_action
                else:
                    print("Action cancelled. Please try again.")
                    
            except (EOFError, KeyboardInterrupt):
                print("\n\nInterrupted by user. Using fallback action.")
                return "Thought:\nInterrupted.\n\nAction:\nmove forward"
    
    def generate_action(self, observation: str, goal: Optional[str] = None, llm_action_failed: bool = False) -> str:
        """
        Generate an action with human-in-the-loop capability.
        
        Args:
            observation: Current observation from the environment
            goal: Optional goal description
            llm_action_failed: Whether the previous LLM action got reward 0
            
        Returns:
            Generated action string
        """
        if self.human_mode == "always":
            # Always ask human
            return self.get_human_action(observation)
        
        if self.human_mode == "fallback" and llm_action_failed:
            # LLM action failed (reward 0), ask human
            print(f"\n[LLM Action Failed - Reward 0]")
            print("Requesting human input...")
            return self.get_human_action(observation)
        
        # Generate LLM action
        try:
            llm_action = super().generate_action(observation, goal, max_retries=3)
            self.llm_actions_count += 1
            
            if self.human_mode == "hybrid":
                # Show LLM action and allow override
                print(f"\n[LLM Action Generated]")
                print(f"Action: {llm_action}")
                override = input("Override? (y/n, default=n): ").strip().lower()
                
                if override == 'y' or override == 'yes':
                    return self.get_human_action(observation, llm_action)
            
            return llm_action
            
        except Exception as e:
            # API call failed, ask human
            print(f"\n[LLM API Error: {e}]")
            print("Requesting human input...")
            return self.get_human_action(observation)
    
    def run_episode(self, env_client, env_idx: str, data_idx: int) -> Dict:
        """
        Run a complete episode with human-in-the-loop capability.
        
        Args:
            env_client: BabyAI environment client
            env_idx: Environment instance ID
            data_idx: Data index to reset to
            
        Returns:
            Dictionary containing episode results
        """
        # Reset counters
        self.human_actions_count = 0
        self.llm_actions_count = 0
        
        # Reset environment and conversation
        self.reset_conversation()
        reset_response = env_client.reset(env_idx, data_idx)
        
        observation = reset_response["observation"]
        done = reset_response["done"]
        reward = reset_response["score"]
        
        # Extract goal from observation if present
        goal = None
        if "Goal:" in observation or "goal:" in observation:
            goal = observation
        
        step_count = 0
        episode_history = []
        previous_reward = 0
        
        # Run episode
        while not done and step_count < self.config.max_rounds:
            try:
                print(f"Observation:\n{observation}\n")
                is_first_step = (step_count == 0)
                llm_action_failed = (step_count > 0 and previous_reward == 0)
                
                if is_first_step and goal:
                    action_response = self.generate_action(observation, goal=goal, llm_action_failed=False)
                else:
                    action_response = self.generate_action(observation, llm_action_failed=llm_action_failed)
                
                if "QUIT" in action_response:
                    print("Episode terminated by user.")
                    break
                
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
                previous_reward = step_output.reward
                reward = step_output.reward
                done = step_output.done
                step_count += 1
                
            except Exception as e:
                print(f"Error during episode: {e}")
                break
        
        # Return episode results
        result = {
            "success": reward >= 0.0,
            "reward": reward,
            "steps": step_count,
            "done": done,
            "conversation": self.conversation_history,
            "history": episode_history,
            "human_actions": self.human_actions_count,
            "llm_actions": self.llm_actions_count,
            "human_mode": self.human_mode
        }
        
        print(f"\n[Episode Statistics]")
        print(f"Human Actions: {self.human_actions_count}")
        print(f"LLM Actions: {self.llm_actions_count}")
        
        return result


def main():
    """Interactive demo of the human-in-the-loop agent."""
    print("="*70)
    print("Human-in-the-Loop BabyAI Agent - Interactive Demo")
    print("="*70)
    
    # Check for agentenv
    try:
        from agentenv.envs.babyai import BabyAIEnvClient
    except ImportError:
        print("\nError: agentenv not found. Please install it:")
        print("  cd ../../agentenv && pip install -e .")
        sys.exit(1)
    
    # Get configuration
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nWarning: OPENAI_API_KEY not set.")
        print("LLM features will not work. Agent will operate in human-only mode.")
        api_key = "dummy-key"
    
    # Select mode
    print("\nSelect mode:")
    print("1. Fallback (default) - Human only when LLM fails")
    print("2. Hybrid - LLM generates, human can override")
    print("3. Human-only - Always ask human")
    
    mode_choice = input("\nChoice (1-3, default=1): ").strip()
    
    mode_map = {"1": "fallback", "2": "hybrid", "3": "always", "": "fallback"}
    human_mode = mode_map.get(mode_choice, "fallback")
    
    print(f"\nMode selected: {human_mode}")
    
    # Create agent
    config = BabyAIAgentConfig(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        max_rounds=20
    )
    
    agent = HumanBabyAIAgent(config, human_mode=human_mode)
    
    # Connect to environment
    env_server = os.getenv("BABYAI_SERVER", "http://localhost:36001")
    print(f"\nConnecting to BabyAI server at {env_server}...")
    
    try:
        env_client = BabyAIEnvClient(
            env_server_base=env_server,
            data_len=200,
            timeout=300
        )
        
        env_idx = env_client.create()
        print(f"Environment created with ID: {env_idx}")
        
        # Get task index
        data_idx = input("\nEnter task index (default=0): ").strip()
        data_idx = int(data_idx) if data_idx else 0
        
        # Run episode
        result = agent.run_episode(env_client, env_idx, data_idx)
        
        # Show results
        print("\n" + "="*70)
        print("EPISODE COMPLETE")
        print("="*70)
        print(f"Success: {result['success']}")
        print(f"Reward: {result['reward']}")
        print(f"Steps: {result['steps']}")
        print(f"Human Actions: {result['human_actions']}")
        print(f"LLM Actions: {result['llm_actions']}")
        print("="*70)
        
        # Save conversation
        save = input("\nSave conversation? (y/n): ").strip().lower()
        if save == 'y' or save == 'yes':
            filename = f"human_episode_{data_idx}.json"
            agent.save_conversation(filename)
            print(f"Conversation saved to: {filename}")
        
        # Close environment
        env_client.close(env_idx)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
