# PrimeIntellect LLM Agent

A unified LLM-based agent for all PrimeIntellect INTELLECT-3-RL environments.

## Overview

This agent provides intelligent solutions for four different task types:

- **CDE (Code)**: Programming challenges with Python code generation and execution-based evaluation
- **LGC (Logic)**: Logic puzzles and reasoning tasks with verifier-based evaluation
- **MTH (Math)**: Mathematical problem solving with rule-based and judge-based verification
- **SCI (Science)**: Science questions with grading and verification

## Features

- 🎯 **Unified Interface**: Single agent handles all four environment types
- 🧠 **Intelligent Prompting**: Environment-specific system prompts optimized for each task type
- 🔄 **Conversation Management**: Maintains separate conversation histories per environment
- 📊 **Batch Processing**: Solve multiple challenges efficiently
- 🎨 **Response Parsing**: Extract code blocks and boxed answers automatically
- 🔧 **Configurable**: Flexible configuration for API settings, generation parameters, and more
- 🚀 **Environment Integration**: Direct integration with PrimeIntellect task generators and evaluators

## Installation

```bash
# Install required dependencies
pip install requests openai datasets httpx pydantic

# For full environment integration, ensure you have the PrimeIntellect environments set up
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

# Solve a math problem
math_solution = agent.solve(
    "What is the sum of the first 10 prime numbers?",
    env="mth"
)
print("Solution:", math_solution)

# Solve a coding problem
code_solution = agent.solve(
    "Write a function to find the nth Fibonacci number",
    env="cde"
)
print("Code:", agent.extract_code(code_solution))
```

### Environment Integration

```python
import asyncio
from env_integration import PrimeIntellectEnvironmentAgent, PrimeIntellectAgentConfig

async def main():
    # Create agent with configuration
    config = PrimeIntellectAgentConfig(
        model="gpt-4o",
        temperature=0.7,
        verbose=True
    )
    
    env_agent = PrimeIntellectEnvironmentAgent(config)
    
    # Solve and evaluate a task
    result = await env_agent.solve_and_evaluate(
        env="mth",
        task_id=0
    )
    
    print(f"Score: {result['score']}")
    print(f"Response: {result['response']}")

asyncio.run(main())
```

### Running Benchmarks

```python
import asyncio
from env_integration import evaluate_agent

async def main():
    # Evaluate on 10 math tasks
    results = await evaluate_agent(
        env="mth",
        num_tasks=10,
        model="gpt-4o",
        verbose=True
    )
    
    print(f"Average Score: {results['average_score']:.2%}")
    print(f"Success Rate: {results['success_rate']:.2%}")

asyncio.run(main())
```

## Configuration

### Agent Configuration

```python
from agent import PrimeIntellectAgentConfig

config = PrimeIntellectAgentConfig(
    # API settings
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    model="gpt-4o",
    
    # Generation parameters
    max_tokens=4096,
    temperature=0.7,
    top_p=1.0,
    timeout=120.0,
    
    # Agent parameters
    max_retries=3,
    retry_delay=2.0,
    verbose=True
)
```

### Environment-Specific Settings

You can customize task generators for each environment:

```python
from env_integration import PrimeIntellectEnvironmentAgent

env_configs = {
    "mth": {
        "dataset_subset": "math",
        "min_avg_reward": 0.0,
        "max_avg_reward": 1.0,
        "judge_model": "gpt-4o",  # Optional: for judge-based evaluation
    },
    "sci": {
        "dataset_subset": "science",
        "min_avg_reward": 0.0,
        "max_avg_reward": 1.0,
    },
    "cde": {
        "dataset_subset": "code",
    },
    "lgc": {
        "dataset_subset": "logic",
        "tasks_to_skip": ["arc_agi", "arc_agi_2", "buggy_tables"],
    }
}

env_agent = PrimeIntellectEnvironmentAgent(
    agent_config=config,
    env_configs=env_configs
)
```

## API Reference

### PrimeIntellectAgent

Main agent class for generating solutions.

#### Methods

- `solve(prompt, env, extra)`: Solve a single task
- `solve_challenge(challenge)`: Solve a challenge dictionary
- `batch_solve(challenges, reset_between)`: Solve multiple challenges
- `extract_code(response)`: Extract Python code from response
- `extract_boxed_answer(response)`: Extract boxed answer from response
- `reset_conversation(env)`: Reset conversation history for an environment
- `clear_all_conversations()`: Clear all conversation histories

### PrimeIntellectEnvironmentAgent

Agent wrapper with environment integration.

#### Methods

- `initialize_environment(env)`: Initialize task generator
- `generate_task(env, task_id)`: Generate a task
- `solve_and_evaluate(env, task_id, **eval_kwargs)`: Generate, solve, and evaluate a task
- `run_benchmark(env, num_tasks, start_task_id, **eval_kwargs)`: Run benchmark on multiple tasks

## Environment Details

### CDE (Code)

- **Task Type**: Programming challenges
- **Input**: Problem description
- **Output**: Python code in markdown code block
- **Evaluation**: Code execution with test cases

### LGC (Logic)

- **Task Type**: Logic puzzles and reasoning
- **Input**: Logic problem description
- **Output**: Reasoned answer
- **Evaluation**: Task-specific verifiers

### MTH (Math)

- **Task Type**: Mathematical problems
- **Input**: Math problem statement
- **Output**: Solution with answer in `\boxed{}`
- **Evaluation**: Rule-based verification with optional LLM judge

### SCI (Science)

- **Task Type**: Science questions
- **Input**: Science problem description
- **Output**: Solution with answer in `\boxed{}`
- **Evaluation**: Verification with optional LLM judge

## Examples

See the example usage at the bottom of `agent.py` and `env_integration.py` for complete examples.

### Math Example

```python
agent = create_agent(model="gpt-4o")
solution = agent.solve(
    "Solve for x: 2x + 5 = 13",
    env="mth"
)
answer = agent.extract_boxed_answer(solution)
print(f"Answer: {answer}")  # Should be "4"
```

### Code Example

```python
solution = agent.solve(
    "Write a function to reverse a string",
    env="cde"
)
code = agent.extract_code(solution)
print(f"Code:\n{code}")
```

## Environment Variables

- `OPENAI_API_KEY`: Default API key for OpenAI-compatible endpoints

## License

This agent is part of the Affinetes project.

## Contributing

Contributions are welcome! Please ensure your code follows the existing style and includes appropriate tests.

## Support

For issues or questions, please refer to the Affinetes documentation or open an issue in the repository.
