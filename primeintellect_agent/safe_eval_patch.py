"""
Patch for code_task to limit concurrent test execution and prevent memory issues.
"""
import asyncio
from typing import List, Any

# Semaphore to limit concurrent test execution
_TEST_SEMAPHORE = asyncio.Semaphore(1)  # Max 1 test at a time to prevent issues


async def limited_gather(*tasks):
    """
    Execute tasks with concurrency limit to prevent memory exhaustion.
    """
    results = []
    
    async def run_with_semaphore(task):
        async with _TEST_SEMAPHORE:
            return await task
    
    # Wrap each task with semaphore
    limited_tasks = [run_with_semaphore(task) for task in tasks]
    
    # Execute all tasks (but semaphore limits concurrency)
    return await asyncio.gather(*limited_tasks, return_exceptions=True)


def apply_patch():
    """
    Apply patches to code_task module to limit concurrency.
    """
    try:
        import sys
        if '/home/120/affinetes/environments/primeintellect/cde' not in sys.path:
            sys.path.insert(0, '/home/120/affinetes/environments/primeintellect/cde')
        
        import code_task
        
        # Store original evaluate method
        original_evaluate = code_task.CodeTask.evaluate
        
        async def patched_evaluate(self, response: str, challenge, timeout: int = 10):
            """Patched evaluate with limited concurrency."""
            # Import necessary items from the module
            from code_task import extract_code_from_markdown, compare_stdout_results, logger
            import json
            
            # Get tests string
            tests_str = challenge.extra.get("tests", "")
            if not tests_str:
                logger.warning("No tests provided")
                return 0.0, "0/0"
            
            # Extract code
            code = extract_code_from_markdown(response)
            if not code:
                logger.warning("No code found in response")
                return 0.0, "0/0"
            
            # Parse tests
            try:
                tests = json.loads(tests_str)
            except Exception as e:
                logger.error(f"Failed to parse tests: {e}")
                return 0.0, "0/0"
            
            inputs = tests.get("inputs", [])
            outputs = tests.get("outputs", [])
            fn_name = tests.get("fn_name", "")
            
            if not inputs or not outputs:
                logger.warning("No test inputs/outputs found")
                return 0.0, "0/0"
            
            if len(inputs) != len(outputs):
                logger.error(f"Mismatch: {len(inputs)} inputs vs {len(outputs)} outputs")
                return 0.0, f"0/{len(inputs)}"
            
            use_function_mode = bool(fn_name and fn_name.strip())
            total = len(inputs)
            tasks = []
            
            for i in range(total):
                try:
                    test_input = json.loads(inputs[i])
                    expected_output = json.loads(outputs[i])
                    
                    if use_function_mode:
                        task = self._run_function_test(
                            code=code,
                            fn_name=fn_name,
                            test_input=test_input,
                            expected_output=outputs[i],
                            timeout=timeout,
                            test_index=i
                        )
                    else:
                        if isinstance(test_input, str):
                            stdin_input = test_input
                        else:
                            stdin_input = str(test_input)
                        
                        task = self._run_stdin_test(
                            code=code,
                            stdin_input=stdin_input,
                            expected_output=expected_output,
                            timeout=timeout,
                            test_index=i
                        )
                    
                    tasks.append(task)
                except Exception as e:
                    logger.debug(f"Test {i}: Failed to prepare - {e}")
                    tasks.append(asyncio.create_task(asyncio.sleep(0)))
            
            # Use limited_gather instead of asyncio.gather
            results = await limited_gather(*tasks)
            
            # Count passed tests
            passed = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.debug(f"Test {i}: EXCEPTION - {result}")
                elif result is True:
                    passed += 1
                    logger.debug(f"Test {i}: PASSED")
                else:
                    logger.debug(f"Test {i}: FAILED")
            
            pass_rate = passed / total if total > 0 else 0.0
            passed_all = (pass_rate == 1.0)
            score = 1.0 if passed_all else 0.0
            test_result = f"{passed}/{total}"
            
            logger.info(f"Evaluation complete: {test_result} tests passed, pass_rate={pass_rate:.2%}, score={score}")
            
            return score, test_result
        
        # Replace the evaluate method
        code_task.CodeTask.evaluate = patched_evaluate
        
        print("[PATCH] Applied concurrency limit to CodeTask.evaluate (SEQUENTIAL - max 1 concurrent test)")
        return True
        
    except Exception as e:
        print(f"[PATCH] Warning: Could not apply patch: {e}")
        return False
