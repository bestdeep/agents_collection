#!/usr/bin/env python3
"""
Quick test script to verify code_task.py implementation.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, '/home/120/affinetes/environments/agents_collection/affine_agent')
sys.path.insert(0, '/home/120/affinetes/environments/affine')

from code_task import CodeTask


async def main():
    """Test the CodeTask implementation"""
    
    # Initialize task generator
    print("Initializing CodeTask...")
    task = CodeTask()
    
    # Generate a sample challenge
    task_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 23297
    print("\nGenerating test challenge...")
    challenge = await task.generate(task_id=task_ID)  # Use first ded-v2 task from the local cache
    
    print(f"Challenge prompt (first 100 chars):")
    print(challenge.prompt[:100])
    
    # Read solution code from solution.py
    solution_path = Path(__file__).parent / "solution.py"
    solution_code = None
    try:
        with open(solution_path, 'r') as f:
            solution_code = f.read().strip()
            if not solution_code:
                solution_code = None
    except FileNotFoundError:
        solution_code = None
    
    # If solution.py is empty or missing, use gold_standard_solution from challenge
    if solution_code is None:
        print("\nNo solution found in solution.py, using gold_standard_solution from challenge...")
        sample = challenge.extra.get('sample', {})
        solution_code = sample.get('gold_standard_solution', '')
        if not solution_code:
            print("ERROR: No gold_standard_solution found in challenge either!")
            return 1
        
        solution_code = solution_code.replace("```python", "").replace("```", "").strip()
        with open(solution_path, 'w') as f:
            f.write(solution_code)
    
    # Mock a simple response (this would come from LLM in production)
    mock_response = f'''
```python
{solution_code}
```
'''
    
    print(f"\n{task_ID} Evaluating response...")
    try:
        score, result = await task.evaluate(mock_response, challenge)
        print(f"\n✓ {task_ID} Evaluation completed successfully!")
        print(f"  Score: {score}")
        print(f"  Result: {result}\n\n")
        
        # Verify the format matches expected output
        assert isinstance(score, float), "Score must be float"
        assert isinstance(result, str), "Result must be string"
        assert "/" in result, "Result must be in 'passed/total' format"
        
        if score == 0.0:
            print(f"Challenge prompt:\n......................................................\n{challenge.prompt}\n....................................................\n")
        else:
            # remove content of solution.py
            try:
                solution_path.write_text("")
                print("Cleared solution.py (score == 1.0).")
            except Exception as e:
                print(f"WARNING: Failed to clear solution.py: {e}")
            
        print(f"Example Response:\n......................................................\n{json.dumps(mock_response)}\n.....................................................")
        
        # print("\nChallenge object structure:")
        # print(f"  Type: {type(challenge)}")
        # if hasattr(challenge, '__dict__'):
        #     print(f"  Keys/Attributes: {list(challenge.__dict__.keys())}")
        #     for key, value in challenge.__dict__.items():
        #         if key == 'prompt':
        #             print(f"    {key}: <{len(value)} chars>")
        #         elif key == 'extra':
        #             print(f"    {key}: {list(value.keys()) if isinstance(value, dict) else type(value)}")
        #             if isinstance(value, dict) and 'sample' in value:
        #                 print(f"      sample keys: {list(value['sample'].keys())}")
        #         else:
        #             print(f"    {key}: {value}")
        # else:
        #     print(f"  Challenge object: {challenge}")
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
