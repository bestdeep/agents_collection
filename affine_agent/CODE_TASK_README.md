# Code Task Implementation for DED-v2

## Overview

This document describes the implementation of `affinetes/environments/agents_collection/affine_agent/code_task.py`, which enables ded‑v2 task evaluation with both **score** and **success/total** metrics.

## Problem Statement

The user requested:
> "make the code of `code_task.py` act like as `code_task_.py` … so I can get the `success/total` and `Score` for `task_id` at `ded-v2`."

### Requirements
1. Return **both** score (float) and pass counts ("passed/total" string)
2. Work with ded‑v2 task IDs
3. Compatible with existing affine_agent integration layer

## Solution Architecture

### Key Components

#### 1. **CodeTask Class** ([code_task.py](code_task.py))
- Wraps the existing DED evaluation logic from `affinetes/environments/affine/ded.py`
- Modified to return tuple `(score, "passed/total")` instead of just `float`
- Reuses all existing test execution infrastructure

#### 2. **Integration Points**
- **Dataset Access**: Uses `HFDataset` (same as DED) with configurable dataset name
- **Code Extraction**: Leverages `ProgramExecutor._strip_fences()` to extract Python code from markdown
- **Test Execution**: Reuses `ProgramExecutor.execute()` for safe, isolated execution
- **Testcase Parsing**: Supports both `stdin_stdout` and `function_call` test types

#### 3. **Return Format**
```python
async def evaluate(response: str, challenge: Challenge) -> Tuple[float, str]:
    # ...
    return (score, f"{passed}/{total}")
```

Where:
- `score`: `1.0` if all tests pass, `0.0` otherwise
- `"passed/total"`: e.g., `"3/5"` for 3 out of 5 tests passing

### Design Decisions

#### Why Wrap DED Logic Instead of Modifying It?
- **Separation of Concerns**: DED returns `float` for compatibility with existing APIs
- **No Breaking Changes**: Existing DED consumers continue to work unchanged
- **Clean Extension**: `CodeTask` adds new functionality without modifying stable code

#### Why Keep Test Execution in Main Repo?
- The testcase payloads are **not** in local prompt banks like `/home/120/ded_task_20000_23301.json`
- They are fetched from the Affine API or loaded via HuggingFace datasets
- The existing `HFDataset` class handles this transparently

## Usage Example

### Basic Usage
```python
import asyncio
from code_task import CodeTask

async def evaluate_ded_v2_task():
    # Initialize evaluator
    task = CodeTask()
    
    # Generate challenge from ded-v2 dataset
    challenge = await task.generate(task_id=20039)
    
    # Get LLM response (from your agent)
    response = """
    ```python
    # Your Python solution here
    ```
    """
    
    # Evaluate and get both score and pass counts
    score, result = await task.evaluate(response, challenge)
    
    print(f"Score: {score}")      # 0.0 or 1.0
    print(f"Result: {result}")    # "3/5" etc.
    
asyncio.run(evaluate_ded_v2_task())
```

### Integration with Existing Agent Wrapper
The `env_integration.py` wrapper already supports tuple returns:
```python
# From env_integration.py line ~158:
async def eval_with_safety():
    try:
        return await asyncio.wait_for(
            task_gen.evaluate(response, challenge, **eval_kwargs),
            timeout=180
        )
    except asyncio.TimeoutError:
        return (0.0, "0/0 (timeout)")  # ← Already tuple-aware!

score = await eval_with_safety()
```

So `CodeTask` can be dropped in as a replacement for DED without any changes to the wrapper.

## Testing

### Test Coverage
- ✅ Initialization and dataset loading
- ✅ Challenge generation from task_id
- ✅ Code extraction from markdown fences
- ✅ Test case parsing (JSON and Python literal formats)
- ✅ Execution with timeout protection
- ✅ Return format validation (`(float, str)` tuple)

### Running Tests
```bash
cd /home/120
source /home/120/affinetes/.venv/bin/activate
python test_code_task.py
```

## Data Flow

```
┌─────────────────┐
│  User Request   │
│   (task_id)     │
└────────┬────────┘
         │
         v
┌─────────────────┐
│   CodeTask      │
│  .generate()    │
└────────┬────────┘
         │
         v
┌─────────────────┐
│   HFDataset     │
│  (fetch sample) │
└────────┬────────┘
         │
         v
┌─────────────────┐
│   Challenge     │
│ (prompt+extra)  │
└────────┬────────┘
         │
         v
┌─────────────────┐
│   LLM Agent     │
│  (generates     │
│   response)     │
└────────┬────────┘
         │
         v
┌─────────────────┐
│   CodeTask      │
│  .evaluate()    │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Extract Code    │
│ (strip fences)  │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Parse Tests     │
│ (verification_  │
│  info/test_     │
│  cases)         │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Execute Each    │
│ Test Case       │
│ (ProgramExecutor│
│  with timeout)  │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Count Passed    │
│ vs Total        │
└────────┬────────┘
         │
         v
┌─────────────────┐
│   Return        │
│ (score, "P/T")  │
└─────────────────┘
```

## Implementation Details

### Code Extraction
```python
program = self._executor._strip_fences(response)
```
Removes markdown fences like:
```
\`\`\`python
<code>
\`\`\`
```
→ extracts just `<code>`

### Test Case Types

#### 1. **stdin_stdout**
```python
{
    "type": "stdin_stdout",
    "input": "3\n1 2 3\n",
    "output": "6"
}
```
- Feeds `input` to program's stdin
- Compares stdout to `output`

#### 2. **function_call**
```python
{
    "type": "function_call",
    "fn_name": "solve",
    "input": [1, 2, 3],
    "output": [6]
}
```
- Wraps code with function call harness
- Captures return value and compares

### Timeout Handling
```python
out, err = await asyncio.wait_for(
    loop.run_in_executor(None, self._executor.execute, program, input),
    timeout=35  # 30s executor timeout + 5s buffer
)
```

### Score Calculation
- **Strict Mode**: All tests must pass for score = 1.0
- **Early Exit**: Stops on first failure to save time
- **Pass Count**: Tracked for "passed/total" string

## Troubleshooting

### Common Issues

#### 1. **Import Errors**
```
ModuleNotFoundError: No module named 'numpy'
```
**Solution**: Activate the virtual environment:
```bash
source /home/120/affinetes/.venv/bin/activate
```

#### 2. **No Testcases Found**
```
Result: 0/0
```
**Causes**:
- Sample missing `verification_info` or `test_cases` field
- JSON parsing failure
- Empty testcase list

**Debug**:
```python
sample = challenge.extra.get("sample", {})
print(sample.keys())  # Check what fields exist
```

#### 3. **Timeouts**
```
Result: 0/5 (timeout)
```
**Causes**:
- Infinite loop in submitted code
- Very slow algorithm
- System overload

**Solutions**:
- Review submitted code for correctness
- Increase timeout in `ProgramExecutor` (default 30s)

## Performance Considerations

### Current Limits
- **Dataset Size**: ded-v2 typically ~20k-100k samples
- **Execution Timeout**: 30s per test case (configurable)
- **Memory**: Isolated subprocess per execution (no shared state)

### Optimization Opportunities
1. **Parallel Test Execution**: Currently sequential; could batch independent tests
2. **Test Caching**: Reuse execution results for identical code+input pairs
3. **Early Exit Strategy**: Already implemented (stops on first failure)

## Future Enhancements

### Potential Improvements
1. **Partial Credit**: Return fractional scores based on pass percentage
   ```python
   score = passed / total  # Instead of binary 0.0/1.0
   ```

2. **Detailed Error Reporting**: Include which specific test failed
   ```python
   return (score, f"{passed}/{total}", {"failed_tests": [...]})
   ```

3. **Custom Dataset Support**: Allow providing local testcase files
   ```python
   CodeTask(local_tests="/path/to/tests.json")
   ```

4. **Metrics Collection**: Track execution time, memory usage per test

## Related Files

| File | Purpose |
|------|---------|
| `/home/120/affinetes/environments/agents_collection/affine_agent/code_task.py` | Main implementation |
| `/home/120/affinetes/environments/affine/ded.py` | Reference DED evaluator |
| `/home/120/affinetes/environments/primeintellect/cde/code_task_.py` | Reference format |
| `/home/120/affinetes/environments/agents_collection/affine_agent/env_integration.py` | Agent wrapper |
| `/home/120/test_code_task.py` | Test script |

## Summary

The `code_task.py` implementation successfully provides:
- ✅ **Score**: Binary pass/fail (1.0 or 0.0)
- ✅ **Pass Counts**: Human-readable "passed/total" string
- ✅ **Compatibility**: Drop-in replacement for DED evaluator
- ✅ **Robustness**: Timeout protection, error handling, multiple test types
- ✅ **Integration**: Works seamlessly with existing affine_agent wrapper

This enables the user to retrieve both evaluation metrics for any ded‑v2 task ID as requested.
