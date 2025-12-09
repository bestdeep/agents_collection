"""
Interactive demo for the BabyAI Agent.

This script allows you to interact with the agent manually,
providing observations and seeing the generated actions.
"""

import os
import sys
from babyai_agent import BabyAIAgent, BabyAIAgentConfig


def print_header():
    """Print welcome header."""
    print("\n" + "=" * 60)
    print("BabyAI Agent Interactive Demo")
    print("=" * 60)
    print("\nThis demo lets you test the agent's action generation.")
    print("You can provide observations and see how the agent responds.")
    print("\nType 'quit' or 'exit' to end the demo.")
    print("Type 'reset' to reset the conversation history.")
    print("Type 'history' to see the conversation history.")
    print("=" * 60 + "\n")


def print_action(action: str):
    """Print the generated action in a formatted way."""
    print("\n" + "-" * 60)
    print("Agent Response:")
    print("-" * 60)
    print(action)
    print("-" * 60 + "\n")


def print_history(agent: BabyAIAgent):
    """Print the conversation history."""
    print("\n" + "=" * 60)
    print("Conversation History")
    print("=" * 60)
    for i, msg in enumerate(agent.conversation_history):
        role = msg["role"].upper()
        content = msg["content"]
        
        # Truncate long content
        if len(content) > 200 and role == "SYSTEM":
            content = content[:200] + "... (truncated)"
        
        print(f"\n[{i+1}] {role}:")
        print(content)
    print("\n" + "=" * 60 + "\n")


def get_api_key():
    """Get API key from user or environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("OpenAI API key not found in environment variables.")
        api_key = input("Please enter your API key (or press Enter to use demo mode): ").strip()
        
        if not api_key:
            print("\nRunning in demo mode (no actual API calls will be made).")
            return "demo-mode"
    
    return api_key


def demo_mode():
    """Run in demo mode without API calls."""
    print("\n" + "=" * 60)
    print("Demo Mode")
    print("=" * 60)
    print("\nRunning without API calls. The agent will generate")
    print("template responses based on the observation.")
    print("\nTo use real API calls, set OPENAI_API_KEY or provide")
    print("your API key when prompted.")
    print("=" * 60 + "\n")
    
    while True:
        observation = input("Enter observation (or 'quit' to exit): ").strip()
        
        if observation.lower() in ['quit', 'exit']:
            break
        
        if not observation:
            print("Please enter an observation.")
            continue
        
        # Generate template response
        print("\n" + "-" * 60)
        print("Agent Response (Demo):")
        print("-" * 60)
        print("Thought:")
        print("Based on the observation, I need to analyze the environment")
        print("and determine the best action to take toward the goal.")
        print("\nAction:")
        print("move forward")
        print("-" * 60 + "\n")


def interactive_mode(agent: BabyAIAgent):
    """Run in interactive mode with real API calls."""
    print("\nAgent initialized successfully!")
    print(f"Model: {agent.config.model}")
    print(f"Temperature: {agent.config.temperature}")
    print(f"Max tokens: {agent.config.max_tokens}\n")
    
    # Example observations for quick testing
    examples = [
        "Goal: Go to the red ball.\n\nIn front of you in this room, you can see several objects: There is a red ball 1 right in front of you 3 steps away.",
        "Goal: Pick up the blue key.\n\nYou can see: There is a blue key 1 right in front of you 2 steps away. There is a yellow door 1 that is 5 steps in front of you.",
        "Goal: Open the door.\n\nYou can see: There is a grey closed door 1 right in front of you 1 step away."
    ]
    
    print("Example observations you can try:")
    for i, ex in enumerate(examples, 1):
        print(f"\n{i}. {ex[:80]}...")
    print("\nOr type your own observation.\n")
    
    while True:
        observation = input("\nEnter observation (or command): ").strip()
        
        if observation.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break
        
        if observation.lower() == 'reset':
            agent.reset_conversation()
            print("\n✓ Conversation history reset.")
            continue
        
        if observation.lower() == 'history':
            print_history(agent)
            continue
        
        if observation.lower() in ['1', '2', '3']:
            try:
                observation = examples[int(observation) - 1]
                print(f"\nUsing example {observation}:")
                print(observation)
            except (ValueError, IndexError):
                print("Invalid example number.")
                continue
        
        if not observation:
            print("Please enter an observation or command.")
            continue
        
        # Extract goal if it's the first turn or contains "Goal:"
        goal = None
        if len(agent.conversation_history) == 2 and "Goal:" in observation:
            # First user message with goal
            goal = observation.split("Goal:")[1].split("\n")[0].strip()
        
        # Generate action
        try:
            print("\nGenerating action...")
            action = agent.generate_action(observation, goal=goal)
            print_action(action)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")
            print("The agent may not be configured correctly.")
            print("Please check your API key and connection.")


def main():
    """Main demo function."""
    print_header()
    
    # Get API key
    api_key = get_api_key()
    
    if api_key == "demo-mode":
        demo_mode()
        return 0
    
    # Get model choice
    print("\nSelect model:")
    print("1. gpt-3.5-turbo (faster, cheaper)")
    print("2. gpt-4 (better quality)")
    print("3. gpt-4-turbo-preview (fast + high quality)")
    print("4. Custom")
    
    choice = input("\nEnter choice (1-4, default=1): ").strip()
    
    model_map = {
        "1": "gpt-3.5-turbo",
        "2": "gpt-4",
        "3": "gpt-4-turbo-preview",
        "": "gpt-3.5-turbo"
    }
    
    if choice == "4":
        model = input("Enter model name: ").strip()
    else:
        model = model_map.get(choice, "gpt-3.5-turbo")
    
    print(f"\nUsing model: {model}")
    
    # Optional: Get custom temperature
    temp_input = input("Enter temperature (0.0-1.0, default=0.7, press Enter to skip): ").strip()
    temperature = float(temp_input) if temp_input else 0.7
    
    # Create agent
    try:
        config = BabyAIAgentConfig(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=512
        )
        agent = BabyAIAgent(config)
        
        interactive_mode(agent)
        
    except Exception as e:
        print(f"\n✗ Error initializing agent: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Goodbye!")
        exit(0)
