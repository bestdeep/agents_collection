# Affine LLM Agent

A unified LLM-based agent for Affine environments in the AffineFoundation/affinetes framework.

## Overview

This agent provides intelligent solutions for two Affine task types:

- **ABD (Algorithm By Deduction)**: Reverse engineering - given a Python program and its output, deduce the input that produces that output
- **DED (Direct Execution Debug)**: Code generation - write Python programs that solve given requirements and pass test cases

## Features

- 🎯 **Unified Interface**: Single agent handles both ABD and DED environments
- 🧠 **Intelligent Prompting**: Environment-specific system prompts optimized for each task type
- 🔄 **Conversation Management**: Maintains separate conversation histories per environment
- 📊 **Batch Processing**: Evaluate multiple challenges efficiently
- 🎨 **Response Parsing**: Extract code blocks and input data automatically
- 🔧 **Configurable**: Flexible configuration for API settings and generation parameters
- 🚀 **Environment Integration**: Direct integration with Affine task generators and evaluators
- 💾 **Result Saving**: Save conversation history, scores, and extracted answers to JSON files

## Installation

```bash
# Navigate to the agent directory
cd /home/120/affinetes/environments/agents_collection/affine_agent

# Install required dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from agent import create_agent

# Create an agent
agent = create_agent(
    api_key="your-api-key",
    model="gpt-4o",
    verbose=True
)

# Solve an ABD task (reverse engineering)
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

abd_solution = agent.solve(abd_prompt, env="abd")
print("Input:", agent.extract_input(abd_solution))

# Solve a DED task (code generation)
ded_prompt = """Write a Python program that reads two integers from stdin and outputs their sum."""

ded_solution = agent.solve(ded_prompt, env="ded")
print("Code:", agent.extract_code(ded_solution))
```

### Environment Integration

```python
import asyncio
from env_integration import AffineEnvironmentAgent, AffineAgentConfig

async def main():
    # Create agent with configuration
    config = AffineAgentConfig(
        model="gpt-4o",
        temperature=0.7,
        verbose=True
    )
    
    env_agent = AffineEnvironmentAgent(config)
    
    # Solve and evaluate an ABD task
    abd_result = await env_agent.solve_and_evaluate(
        env="abd",
        task_id=0
    )
    
    print(f"Score: {abd_result['score']}")
    print(f"Input: {abd_result['extracted_answer']}")
    
    # Solve and evaluate a DED task
    ded_result = await env_agent.solve_and_evaluate(
        env="ded",
        task_id=0
    )
    
    print(f"Score: {ded_result['score']}")
    print(f"Code: {ded_result['extracted_answer']}")

asyncio.run(main())
```

### Running Benchmarks

```python
import asyncio
from env_integration import evaluate_agent

async def main():
    # Evaluate on 10 ABD tasks
    abd_results = await evaluate_agent(
        env="abd",
        num_tasks=10,
        verbose=True
    )
    
    print(f"ABD Average Score: {abd_results['avg_score']:.3f}")
    
    # Evaluate on 10 DED tasks
    ded_results = await evaluate_agent(
        env="ded",
        num_tasks=10,
        verbose=True,
        save_results=True,
        output_dir="ded_results"
    )
    
    print(f"DED Average Score: {ded_results['avg_score']:.3f}")

asyncio.run(main())
```

## Configuration

The agent can be configured via `config.json` or programmatically:

```json
{
  "agent": {
    "api_key": "${OPENAI_API_KEY}",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 4096,
    "timeout": 120.0,
    "max_retries": 3,
    "retry_delay": 2.0,
    "verbose": true
  },
  "environments": {
    "abd": {
      "dataset_name": "satpalsr/rl-python",
      "split": "train"
    },
    "ded": {
      "dataset_name": "satpalsr/rl-python",
      "split": "train"
    }
  }
}
```

### Configuration Parameters

#### Agent Configuration

- **api_key**: API key for LLM service (defaults to `OPENAI_API_KEY` env var)
- **base_url**: Base URL for API endpoint
- **model**: Model name to use (e.g., "gpt-4o", "deepseek-ai/DeepSeek-V3")
- **temperature**: Sampling temperature (0.0-1.0)
- **max_tokens**: Maximum tokens in response
- **timeout**: API request timeout in seconds
- **max_retries**: Number of retry attempts on failure
- **retry_delay**: Delay between retries in seconds
- **verbose**: Enable verbose logging

## Task Types

### ABD (Algorithm By Deduction)

**Goal**: Given a Python program and its expected output, determine what input would produce that output.

**Example**:
```
Input Program:
x = int(input())
y = int(input())
print(f"Sum: {x + y}")

Expected Output:
Sum: 15

Task: Determine the input values for x and y
```

**Agent Response Format**:
The agent provides its answer within `<INPUT></INPUT>` tags:
```
<INPUT>
7
8
</INPUT>
```

**Evaluation**: The input is extracted and fed to the program to verify it produces the expected output.

### DED (Direct Execution Debug)

**Goal**: Write a Python program that solves a given problem and passes all test cases.

**Example**:
```
Write a Python program that:
- Reads two integers from stdin
- Outputs their sum to stdout
```

**Agent Response Format**:
The agent provides code in a Python markdown block:
````python
x = int(input())
y = int(input())
print(x + y)
````

**Evaluation**: The code is executed with test cases to verify correctness.

## API Reference

### AffineAgent

The main agent class for solving Affine tasks.

#### Methods

- `solve(prompt: str, env: str) -> str`: Solve a task with the given prompt
- `solve_challenge(challenge: Dict) -> str`: Solve a challenge object
- `extract_input(response: str) -> Optional[str]`: Extract input from ABD response
- `extract_code(response: str) -> Optional[str]`: Extract code from DED response
- `reset_conversation(env: str)`: Reset conversation history for environment
- `get_conversation_history(env: str) -> List[Dict]`: Get conversation history

### AffineEnvironmentAgent

Wrapper for environment integration.

#### Methods

- `initialize_environment(env: str)`: Initialize task generator for environment
- `generate_task(env: str, task_id: Optional[int]) -> Challenge`: Generate a task
- `solve_and_evaluate(env: str, task_id: Optional[int], **kwargs) -> Dict`: Solve and evaluate a task
- `save_result(result: Dict, output_dir: str)`: Save evaluation result

## Output Format

The `solve_and_evaluate` method returns a dictionary with:

```python
{
    "env": "abd" | "ded",
    "task_id": int,
    "challenge": Challenge,
    "response": str,  # Raw agent response
    "score": float | tuple,  # Evaluation score
    "conversation_history": List[Dict],
    "extracted_answer": str  # Extracted input or code
}
```

## Result Saving

Results can be automatically saved to JSON files:

```python
result = await env_agent.solve_and_evaluate(
    env="ded",
    task_id=5,
    save_results=True,
    output_dir="results"
)
```

Result files are named: `{env}_task{id}_{timestamp}.json`

### Analyzing Results

```python
from result_saver import load_results, analyze_results

# Load all DED results
results = load_results("results", env="ded")

# Analyze performance
analysis = analyze_results(results)
print(f"Average Score: {analysis['avg_score']:.3f}")
print(f"Total Tasks: {analysis['total_tasks']}")
```

## System Prompts

The agent uses specialized system prompts for each environment:

### ABD System Prompt
Focuses on:
- Reverse engineering and program analysis
- Understanding input formats
- Working backwards from output to input
- Extracting input within `<INPUT></INPUT>` tags

### DED System Prompt
Focuses on:
- Writing clean, executable Python code
- Reading from STDIN, writing to STDOUT
- Handling test cases correctly
- Providing code in markdown blocks

## Tips for Best Performance

1. **Temperature**: Use lower temperature (0.3-0.5) for more deterministic code generation
2. **Model Selection**: More capable models (GPT-4, DeepSeek-V3) perform better on complex tasks
3. **Verbose Mode**: Enable verbose mode during development to see API calls and debugging info
4. **Batch Evaluation**: Use `evaluate_agent()` for efficient batch processing
5. **Result Saving**: Always save results for later analysis and debugging

## Troubleshooting

### API Errors
- Verify your API key is set correctly
- Check base_url matches your API provider
- Ensure model name is correct for your provider

### Evaluation Timeouts
- Increase timeout in configuration
- Some tasks may have infinite loops or performance issues
- Check task difficulty and complexity

### Import Errors
- Ensure environment paths are correct in `env_integration.py`
- Verify Affine environment modules are accessible
- Check Python path includes `/home/120/affinetes/environments/affine`

## Examples

See `agent.py` and `env_integration.py` for complete example usage in their `__main__` sections.

## License

Part of the AffineFoundation/affinetes project.

## Contributing

This agent is part of the agents_collection in the affinetes framework. For contributions and issues, refer to the main repository.
