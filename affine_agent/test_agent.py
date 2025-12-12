#!/usr/bin/env python3
"""
Test script for Affine Agent

Quick tests to verify agent functionality.
"""

import asyncio
from agent import create_agent, AffineAgentConfig


async def test_agent_basic():
    """Test basic agent functionality without environment integration."""
    print("Testing basic agent functionality...")
    print("="*60)
    
    # Create agent
    agent = create_agent(
        model="gpt-4o",
        temperature=0.7,
        verbose=False
    )
    
    # Test ABD prompt
    print("\n1. Testing ABD (Algorithm By Deduction)")
    print("-"*60)
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
    
    try:
        abd_response = agent.solve(abd_prompt, env="abd")
        extracted_input = agent.extract_input(abd_response)
        
        print("✓ ABD response generated")
        print(f"Extracted input: {extracted_input}")
        
        if extracted_input:
            print("✓ Input extraction successful")
        else:
            print("✗ Input extraction failed")
    except Exception as e:
        print(f"✗ ABD test failed: {e}")
    
    # Test DED prompt
    print("\n2. Testing DED (Direct Execution Debug)")
    print("-"*60)
    ded_prompt = """Write a Python program that:
- Reads two integers from stdin (one per line)
- Outputs their sum to stdout

Example:
Input: 5, 3
Output: 8"""
    
    try:
        ded_response = agent.solve(ded_prompt, env="ded")
        extracted_code = agent.extract_code(ded_response)
        
        print("✓ DED response generated")
        if extracted_code:
            print(f"Extracted code preview: {extracted_code[:100]}...")
            print("✓ Code extraction successful")
        else:
            print("✗ Code extraction failed")
    except Exception as e:
        print(f"✗ DED test failed: {e}")
    
    print("\n" + "="*60)
    print("Basic agent tests completed")


async def test_environment_integration():
    """Test environment integration (requires Affine environment)."""
    print("\nTesting environment integration...")
    print("="*60)
    
    try:
        from env_integration import AffineEnvironmentAgent
        
        # Create config
        config = AffineAgentConfig(
            model="gpt-4o",
            temperature=0.7,
            verbose=True
        )
        
        env_agent = AffineEnvironmentAgent(config)
        
        # Test ABD environment
        print("\n3. Testing ABD environment integration")
        print("-"*60)
        try:
            abd_result = await env_agent.solve_and_evaluate(
                env="abd",
                task_id=0
            )
            print(f"✓ ABD evaluation completed")
            print(f"  Score: {abd_result['score']}")
            print(f"  Extracted answer: {abd_result['extracted_answer']}")
        except Exception as e:
            print(f"✗ ABD environment test failed: {e}")
        
        # Test DED environment
        print("\n4. Testing DED environment integration")
        print("-"*60)
        try:
            ded_result = await env_agent.solve_and_evaluate(
                env="ded",
                task_id=0
            )
            print(f"✓ DED evaluation completed")
            print(f"  Score: {ded_result['score']}")
            if ded_result['extracted_answer']:
                print(f"  Code preview: {ded_result['extracted_answer'][:80]}...")
        except Exception as e:
            print(f"✗ DED environment test failed: {e}")
        
        print("\n" + "="*60)
        print("Environment integration tests completed")
        
    except ImportError as e:
        print(f"⚠ Environment integration tests skipped: {e}")
        print("  Make sure Affine environment is accessible")


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Affine Agent Test Suite")
    print("="*60)
    
    # Test basic functionality
    await test_agent_basic()
    
    # Test environment integration
    await test_environment_integration()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
