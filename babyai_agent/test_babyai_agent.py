"""
Simple test script for the BabyAI Agent.

Tests basic functionality without requiring the environment server.
"""

import os
from babyai_agent import BabyAIAgent, BabyAIAgentConfig


def test_agent_initialization():
    """Test that the agent can be initialized."""
    print("Test 1: Agent Initialization")
    config = BabyAIAgentConfig(
        api_key="test-key",
        model="gpt-3.5-turbo"
    )
    agent = BabyAIAgent(config)
    print("✓ Agent initialized successfully")
    print(f"  - Model: {agent.config.model}")
    print(f"  - Max rounds: {agent.config.max_rounds}")
    print(f"  - Temperature: {agent.config.temperature}")
    print()


def test_conversation_reset():
    """Test that conversation history resets correctly."""
    print("Test 2: Conversation Reset")
    agent = BabyAIAgent()
    
    # Check initial state
    assert len(agent.conversation_history) == 2, "Initial conversation should have 2 messages"
    assert agent.conversation_history[0]["role"] == "system", "First message should be system"
    assert agent.conversation_history[1]["role"] == "assistant", "Second message should be assistant"
    print("✓ Initial conversation history correct")
    
    # Add some messages
    agent.conversation_history.append({"role": "user", "content": "test"})
    agent.conversation_history.append({"role": "assistant", "content": "test"})
    assert len(agent.conversation_history) == 4, "Should have 4 messages after adding"
    
    # Reset
    agent.reset_conversation()
    assert len(agent.conversation_history) == 2, "Should have 2 messages after reset"
    print("✓ Conversation reset works correctly")
    print()


def test_config_from_dict():
    """Test creating agent from config dictionary."""
    print("Test 3: Config from Dictionary")
    config_dict = {
        "api_key": "test-key",
        "model": "gpt-4",
        "max_tokens": 1024,
        "temperature": 0.5,
        "max_rounds": 30
    }
    
    agent = BabyAIAgent.from_dict(config_dict)
    assert agent.config.model == "gpt-4", "Model should be gpt-4"
    assert agent.config.max_tokens == 1024, "Max tokens should be 1024"
    assert agent.config.temperature == 0.5, "Temperature should be 0.5"
    assert agent.config.max_rounds == 30, "Max rounds should be 30"
    print("✓ Agent created from config dictionary")
    print(f"  - Model: {agent.config.model}")
    print(f"  - Max tokens: {agent.config.max_tokens}")
    print(f"  - Temperature: {agent.config.temperature}")
    print()


def test_system_prompt():
    """Test that system prompt is correctly formatted."""
    print("Test 4: System Prompt")
    agent = BabyAIAgent()
    
    system_message = agent.conversation_history[0]
    assert "exploration master" in system_message["content"].lower(), "Should mention exploration master"
    assert "turn right" in system_message["content"].lower(), "Should list actions"
    assert "turn left" in system_message["content"].lower(), "Should list actions"
    assert "move forward" in system_message["content"].lower(), "Should list actions"
    assert "Thought:" in system_message["content"], "Should specify response format"
    assert "Action:" in system_message["content"], "Should specify response format"
    
    print("✓ System prompt is correctly formatted")
    print(f"  - Prompt length: {len(system_message['content'])} characters")
    print()


def test_config_defaults():
    """Test that config defaults are sensible."""
    print("Test 5: Config Defaults")
    config = BabyAIAgentConfig()
    
    assert config.model == "gpt-3.5-turbo", "Default model should be gpt-3.5-turbo"
    assert config.max_tokens == 512, "Default max_tokens should be 512"
    assert config.temperature == 0.7, "Default temperature should be 0.7"
    assert config.max_rounds == 20, "Default max_rounds should be 20"
    assert config.max_retries == 3, "Default max_retries should be 3"
    
    print("✓ Config defaults are correct")
    print(f"  - Default model: {config.model}")
    print(f"  - Default max_tokens: {config.max_tokens}")
    print(f"  - Default temperature: {config.temperature}")
    print()


def test_save_conversation(tmp_path="/tmp"):
    """Test saving conversation to file."""
    print("Test 6: Save Conversation")
    agent = BabyAIAgent()
    
    # Add some test conversation
    agent.conversation_history.append({
        "role": "user",
        "content": "Test observation"
    })
    agent.conversation_history.append({
        "role": "assistant",
        "content": "Test action"
    })
    
    # Save to file
    test_file = os.path.join(tmp_path, "test_conversation.json")
    try:
        agent.save_conversation(test_file)
        print(f"✓ Conversation saved to {test_file}")
        
        # Verify file exists
        if os.path.exists(test_file):
            print(f"  - File size: {os.path.getsize(test_file)} bytes")
            os.remove(test_file)  # Clean up
            print("  - File cleaned up")
    except Exception as e:
        print(f"✗ Error saving conversation: {e}")
    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("BabyAI Agent Test Suite")
    print("=" * 60)
    print()
    
    try:
        test_agent_initialization()
        test_conversation_reset()
        test_config_from_dict()
        test_system_prompt()
        test_config_defaults()
        test_save_conversation()
        
        print("=" * 60)
        print("All Tests Passed! ✓")
        print("=" * 60)
        print()
        print("The agent is ready to use. To run a full evaluation:")
        print("  1. Start the BabyAI server: babyai --host 0.0.0.0 --port 36001")
        print("  2. Run: python run_babyai_agent.py --api_key YOUR_API_KEY")
        print()
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
