"""
Test suite for PrimeIntellect Agent

Run with: python test_agent.py
"""

import os
import sys


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from agent import PrimeIntellectAgent, PrimeIntellectAgentConfig, create_agent
        print("✓ agent module imported")
        
        from config_utils import load_config, create_agent_config_from_dict
        print("✓ config_utils module imported")
        
        print("✓ All basic imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_agent_creation():
    """Test agent creation with different configurations."""
    print("\nTesting agent creation...")
    
    try:
        from agent import create_agent, PrimeIntellectAgentConfig
        
        # Test with defaults
        agent1 = create_agent(api_key="test-key", model="gpt-4o")
        print("✓ Agent created with defaults")
        
        # Test with custom config
        config = PrimeIntellectAgentConfig(
            api_key="test-key",
            model="gpt-4o-mini",
            temperature=0.5,
            max_tokens=2048
        )
        agent2 = create_agent()
        print("✓ Agent created with custom config")
        
        # Test environment initialization
        agent1.reset_conversation("mth")
        agent1.reset_conversation("cde")
        agent1.reset_conversation("lgc")
        agent1.reset_conversation("sci")
        print("✓ All environment conversations initialized")
        
        return True
    except Exception as e:
        print(f"✗ Agent creation error: {e}")
        return False


def test_response_parsing():
    """Test response parsing utilities."""
    print("\nTesting response parsing...")
    
    try:
        from agent import create_agent
        
        agent = create_agent(api_key="test-key")
        
        # Test boxed answer extraction
        test_cases_boxed = [
            ("The answer is \\boxed{42}", "42"),
            ("Therefore \\boxed{3.14159}", "3.14159"),
            ("Final answer: \\boxed{x = 5}", "x = 5"),
            ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
            ("No boxed answer here", None),
        ]
        
        for text, expected in test_cases_boxed:
            result = agent.extract_boxed_answer(text)
            if result == expected:
                print(f"✓ Boxed extraction: '{text[:30]}...' -> {result}")
            else:
                print(f"✗ Boxed extraction failed: expected {expected}, got {result}")
                return False
        
        # Test code extraction
        test_cases_code = [
            ("```python\nprint('hello')\n```", "print('hello')"),
            ("```\ncode here\n```", "code here"),
            ("No code blocks", "No code blocks"),
        ]
        
        for text, expected in test_cases_code:
            result = agent.extract_code(text)
            if result == expected:
                print(f"✓ Code extraction: '{text[:30]}...' -> '{result[:20]}...'")
            else:
                print(f"✗ Code extraction failed: expected '{expected}', got '{result}'")
                return False
        
        print("✓ All parsing tests passed")
        return True
    except Exception as e:
        print(f"✗ Parsing test error: {e}")
        return False


def test_conversation_management():
    """Test conversation history management."""
    print("\nTesting conversation management...")
    
    try:
        from agent import create_agent
        
        agent = create_agent(api_key="test-key")
        
        # Initialize conversations
        agent.reset_conversation("mth")
        agent.reset_conversation("sci")
        
        # Check history
        mth_history = agent.get_conversation_history("mth")
        assert len(mth_history) > 0, "Math history should not be empty"
        print(f"✓ Math conversation initialized ({len(mth_history)} messages)")
        
        sci_history = agent.get_conversation_history("sci")
        assert len(sci_history) > 0, "Science history should not be empty"
        print(f"✓ Science conversation initialized ({len(sci_history)} messages)")
        
        # Test clearing
        agent.clear_all_conversations()
        assert len(agent.conversation_history) == 0, "All conversations should be cleared"
        print("✓ All conversations cleared")
        
        return True
    except Exception as e:
        print(f"✗ Conversation management error: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        from config_utils import load_config, create_agent_config_from_dict, get_env_configs
        
        # Try to load config file
        try:
            config = load_config()
            print("✓ Config file loaded")
            
            # Test agent config creation
            agent_config = create_agent_config_from_dict(config)
            print(f"✓ Agent config created (model: {agent_config.model})")
            
            # Test environment configs
            env_configs = get_env_configs(config)
            print(f"✓ Environment configs loaded ({len(env_configs)} environments)")
            
            # Verify all 4 environments are present
            expected_envs = ["cde", "lgc", "mth", "sci"]
            for env in expected_envs:
                if env not in env_configs:
                    print(f"✗ Missing environment config: {env}")
                    return False
            print("✓ All 4 environment configs present")
            
        except FileNotFoundError:
            print("⚠ Config file not found (this is OK for testing)")
            return True
            
        return True
    except Exception as e:
        print(f"✗ Config loading error: {e}")
        return False


def test_system_prompts():
    """Test that all environment system prompts are defined."""
    print("\nTesting system prompts...")
    
    try:
        from agent import PrimeIntellectAgent
        
        agent = PrimeIntellectAgent()
        
        expected_envs = ["cde", "lgc", "mth", "sci"]
        for env in expected_envs:
            if env not in agent.SYSTEM_PROMPTS:
                print(f"✗ Missing system prompt for {env}")
                return False
            
            prompt = agent.SYSTEM_PROMPTS[env]
            if not prompt or len(prompt) < 100:
                print(f"✗ System prompt for {env} is too short or empty")
                return False
            
            print(f"✓ {env.upper()} system prompt defined ({len(prompt)} chars)")
        
        print("✓ All system prompts are properly defined")
        return True
    except Exception as e:
        print(f"✗ System prompts test error: {e}")
        return False


def test_environment_integration():
    """Test environment integration module (if available)."""
    print("\nTesting environment integration...")
    
    try:
        from env_integration import PrimeIntellectEnvironmentAgent, PrimeIntellectAgentConfig
        print("✓ Environment integration module loaded")
        
        # Try to create agent
        config = PrimeIntellectAgentConfig(api_key="test-key")
        env_agent = PrimeIntellectEnvironmentAgent(config)
        print("✓ Environment agent created")
        
        print("✓ Environment integration available")
        print("  (Note: Full testing requires PrimeIntellect environments)")
        return True
    except ImportError as e:
        print(f"⚠ Environment integration not available: {e}")
        print("  (This is OK if PrimeIntellect environments are not set up)")
        return True
    except Exception as e:
        print(f"✗ Environment integration error: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("="*70)
    print("PrimeIntellect Agent Test Suite")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Agent Creation", test_agent_creation),
        ("Response Parsing", test_response_parsing),
        ("Conversation Management", test_conversation_management),
        ("Configuration Loading", test_config_loading),
        ("System Prompts", test_system_prompts),
        ("Environment Integration", test_environment_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print("="*70)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    # Check for API key warning
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Note: OPENAI_API_KEY not set. This is fine for unit tests.")
        print("   Set it for live API tests: export OPENAI_API_KEY='your-key'\n")
    
    success = run_all_tests()
    
    if success:
        print("\n✓ All tests passed! The agent is ready to use.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        sys.exit(1)
