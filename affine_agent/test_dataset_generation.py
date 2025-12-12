#!/usr/bin/env python3
"""
Quick test script for dataset generation functionality.
Tests a single task to verify the pipeline works.
"""

import asyncio
import os
from agent import AffineAgentConfig
from dataset_generator import DatasetGenerator


async def test_single_task():
    """Test generating a dataset entry for a single task."""
    print("="*70)
    print("Testing Dataset Generation")
    print("="*70)
    
    # Create agent config
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return False
    
    agent_config = AffineAgentConfig(
        api_key=api_key,
        model="gpt-4o",
        temperature=0.7,
        verbose=True
    )
    
    # Test with DED environment
    generator = DatasetGenerator(
        env="ded",
        agent_config=agent_config,
        output_dir="test_datasets"
    )
    
    print("\n📝 Testing DED task generation...")
    print("Task ID: 20000")
    print("-"*70)
    
    try:
        # Generate single entry
        result = await generator.process_single_task(20000)
        
        if result:
            print("\n✅ Successfully generated dataset entry!")
            print(f"Score: {result.get('score', 'N/A')}")
            print(f"Task ID: {result.get('task_id', 'N/A')}")
            
            if result.get('score') == 1.0:
                print("✅ Perfect score! This entry will be saved to dataset.")
            else:
                print(f"⚠️  Score {result.get('score')} < 1.0, entry will not be saved.")
            
            return True
        else:
            print("❌ Failed to generate entry")
            return False
            
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    success = await test_single_task()
    
    print("\n" + "="*70)
    if success:
        print("✅ Test completed successfully!")
        print("\nNext steps:")
        print("1. Run full generation:")
        print("   python cli.py generate-dataset --env ded --start-id 20000 --end-id 20010")
        print("\n2. Or use the convenience script:")
        print("   bash generate_datasets.sh")
    else:
        print("❌ Test failed. Please check the errors above.")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
