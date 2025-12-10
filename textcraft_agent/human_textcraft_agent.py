"""
Human-in-the-Loop TextCraft Agent

This agent allows human intervention when the LLM fails to generate valid actions.
The human can provide actions via command line input.
"""

import os
import json
import sys
from typing import Dict, Optional, Tuple
from pathlib import Path

# Import the base agent
from textcraft_agent import TextCraftAgent, TextCraftAgentConfig


class HumanTextCraftAgent(TextCraftAgent):
    """
    TextCraft agent with human-in-the-loop capability.
    
    When the LLM fails to generate a valid action, the agent prompts
    the human to provide an action via command line input.
    """
    
    def __init__(self, config: Optional[TextCraftAgentConfig] = None, human_mode: str = "fallback"):
        """
        Initialize the Human-in-the-Loop TextCraft Agent.
        
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
        
    def get_human_action(self, observation: str, llm_action: Optional[Tuple[str, str]] = None) -> Tuple[str, str]:
        """
        Get action from human via command line input.
        
        Args:
            observation: Current observation
            llm_action: LLM's proposed (full_response, action) tuple (if any)
            
        Returns:
            Tuple of (full_response, action_command)
        """
        print("\n" + "="*70)
        print("HUMAN INPUT REQUIRED")
        print("="*70)
        print(f"\nCurrent Observation:\n{observation}\n")
        
        if llm_action:
            print(f"LLM's Proposed Response:\n{llm_action[0]}\n")
            print(f"LLM's Extracted Action: {llm_action[1]}\n")
        
        print("Available Action Types:")
        print("  1. get <amount> <item>")
        print("     Example: get 1 oak log")
        print("  2. craft <output> using <ingredients>")
        print("     Example: craft 4 oak planks using 1 oak log")
        print("  3. inventory")
        print("\nSpecial Commands:")
        print("  - 'accept' (accept LLM's action if shown)")
        print("  - 'skip' (use fallback action: inventory)")
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
                    return ("Thought:\nUser quit.\n\nAction:\nQUIT", "QUIT")
                
                if user_input.lower() == 'skip':
                    print("Using fallback action: inventory")
                    return ("Thought:\nHuman chose to skip.\n\nAction:\ninventory", "inventory")
                
                if user_input.lower() == 'accept' and llm_action:
                    print("Accepting LLM's action.")
                    return llm_action
                
                if user_input.lower() == 'accept' and not llm_action:
                    print("No LLM action to accept. Please enter an action.")
                    continue
                
                # Format the action
                formatted_response = f"Thought:\nHuman intervention.\n\nAction:\n{user_input}"
                
                confirm = input(f"\nConfirm action '{user_input}'? (y/n): ").strip().lower()
                if confirm == 'y' or confirm == 'yes':
                    self.human_actions_count += 1
                    return (formatted_response, user_input)
                else:
                    print("Action cancelled. Please try again.")
                    
            except (EOFError, KeyboardInterrupt):
                print("\n\nInterrupted by user. Using fallback action.")
                return ("Thought:\nInterrupted.\n\nAction:\ninventory", "inventory")
    
    def generate_action(self, observation: str, include_goal: bool = False, llm_action_failed: bool = False) -> Tuple[str, str]:
        """
        Generate an action with human-in-the-loop capability.
        
        Args:
            observation: Current observation from the environment
            include_goal: Whether the observation includes the goal
            llm_action_failed: Whether the previous LLM action got reward 0
            
        Returns:
            Tuple of (full_response, action_command)
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
            full_response, action = super().generate_action(observation, include_goal, max_retries=3)
            self.llm_actions_count += 1
            
            if self.human_mode == "hybrid":
                # Show LLM action and allow override
                print(f"\n[LLM Action Generated]")
                print(f"Response: {full_response}")
                print(f"Action: {action}")
                override = input("Override? (y/n, default=n): ").strip().lower()
                
                if override == 'y' or override == 'yes':
                    return self.get_human_action(observation, (full_response, action))
            
            return full_response, action
            
        except Exception as e:
            # API call failed, ask human
            print(f"\n[LLM API Error: {e}]")
            print("Requesting human input...")
            return self.get_human_action(observation)


def main():
    """Interactive demo of the human-in-the-loop agent."""
    print("="*70)
    print("Human-in-the-Loop TextCraft Agent - Interactive Demo")
    print("="*70)
    
    # Check for agentenv
    try:
        from agentenv.envs.textcraft import TextCraftEnvClient
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
    config = TextCraftAgentConfig(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        max_rounds=50
    )
    
    agent = HumanTextCraftAgent(config, human_mode=human_mode)
    
    # Connect to environment
    env_server = os.getenv("TEXTCRAFT_SERVER", "http://localhost:36001")
    print(f"\nConnecting to TextCraft server at {env_server}...")
    
    try:
        env_client = TextCraftEnvClient(
            env_server_base=env_server,
            data_len=200,
            timeout=300
        )
        
        env_idx = env_client.create()
        print(f"Environment created with ID: {env_idx}")
        
        # Get task index
        data_idx = input("\nEnter task index (default=0): ").strip()
        data_idx = int(data_idx) if data_idx else 0
        
        # Reset environment and run
        print("\nStarting episode...")
        agent.reset_conversation()
        reset_response = env_client.reset(env_idx, data_idx)
        
        observation = reset_response["observation"]
        done = reset_response.get("done", False)
        reward = reset_response.get("reward", 0)
        
        print(f"\nInitial Observation:\n{observation}\n")
        
        step_count = 0
        episode_history = []
        previous_reward = 0
        
        # Run episode loop
        while not done and step_count < config.max_rounds:
            is_first_step = (step_count == 0)
            llm_action_failed = (step_count > 0 and previous_reward == 0)
            
            try:
                full_response, action = agent.generate_action(observation, include_goal=is_first_step, llm_action_failed=llm_action_failed)
                
                if action == "QUIT":
                    print("Episode terminated by user.")
                    break
                
                print(f"\n--- Step {step_count + 1} ---")
                print(f"Executing: {action}")
                
                # Take step
                step_output = env_client.step(env_idx, action)
                
                observation = step_output.state
                previous_reward = step_output.reward
                reward = step_output.reward
                done = step_output.done
                
                print(f"Observation: {observation}")
                print(f"Reward: {reward}, Done: {done}")
                
                episode_history.append({
                    "step": step_count,
                    "action": action,
                    "observation": observation,
                    "reward": reward,
                    "done": done
                })
                
                step_count += 1
                
            except Exception as e:
                print(f"Error during step: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Show results
        success = reward >= 1.0
        
        print("\n" + "="*70)
        print("EPISODE COMPLETE")
        print("="*70)
        print(f"Success: {success}")
        print(f"Reward: {reward}")
        print(f"Steps: {step_count}")
        print(f"Human Actions: {agent.human_actions_count}")
        print(f"LLM Actions: {agent.llm_actions_count}")
        print("="*70)
        
        # Save conversation
        save = input("\nSave conversation? (y/n): ").strip().lower()
        if save == 'y' or save == 'yes':
            filename = f"human_textcraft_episode_{data_idx}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "task_idx": data_idx,
                    "success": success,
                    "reward": reward,
                    "steps": step_count,
                    "human_actions": agent.human_actions_count,
                    "llm_actions": agent.llm_actions_count,
                    "conversation": agent.conversation_history,
                    "history": episode_history
                }, f, indent=2, ensure_ascii=False)
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
