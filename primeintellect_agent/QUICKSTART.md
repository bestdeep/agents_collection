# PrimeIntellect Agent - Quick Start Guide

## 🚀 Quick Start (60 seconds)

### 1. Install Dependencies

```bash
cd /home/120/affinetes/environments/agents_collection/primeintellect_agent
pip install -r requirements.txt
```

### 2. Set Your API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Run Your First Task

#### Option A: Command Line
```bash
# Solve a math problem
python cli.py solve --env mth --prompt "What is the sum of the first 10 prime numbers?"

# Solve a code problem
python cli.py solve --env cde --prompt "Write a function to reverse a string"

# Interactive mode
python cli.py interactive --env mth
```

#### Option B: Python Script
```python
from agent import create_agent

# Create agent
agent = create_agent(model="gpt-4o")

# Solve a problem
solution = agent.solve("Calculate the area of a circle with radius 5", env="mth")
print(solution)

# Extract the answer
answer = agent.extract_boxed_answer(solution)
print(f"Answer: {answer}")
```

## 📚 Detailed Usage

### Command Line Interface

#### Solve a Single Task
```bash
python cli.py solve --env mth --prompt "Your question here"
```

Environments:
- `cde` - Code/Programming
- `lgc` - Logic puzzles
- `mth` - Mathematics
- `sci` - Science questions

#### Interactive Mode
```bash
python cli.py interactive --env mth
```
Type your questions and get instant answers. Type `exit` to quit.

#### Evaluate Dataset Tasks
```bash
# Evaluate a specific task from the dataset
python cli.py evaluate --env mth --task-id 0
```

#### Run Benchmarks
```bash
# Evaluate on 10 tasks
python cli.py benchmark --env mth --num-tasks 10
```

### Python API

#### Basic Usage
```python
from agent import create_agent

agent = create_agent(
    model="gpt-4o",
    temperature=0.7,
    verbose=True
)

# Math
math_solution = agent.solve("What is 15 + 27?", env="mth")
print(agent.extract_boxed_answer(math_solution))

# Code
code_solution = agent.solve("Write a fibonacci function", env="cde")
print(agent.extract_code(code_solution))

# Logic
logic_solution = agent.solve("If A > B and B > C, is A > C?", env="lgc")
print(logic_solution)

# Science
sci_solution = agent.solve("What is the speed of light?", env="sci")
print(agent.extract_boxed_answer(sci_solution))
```

#### Environment Integration
```python
import asyncio
from env_integration import PrimeIntellectEnvironmentAgent, PrimeIntellectAgentConfig

async def main():
    config = PrimeIntellectAgentConfig(
        model="gpt-4o",
        temperature=0.7
    )
    
    env_agent = PrimeIntellectEnvironmentAgent(config)
    
    # Solve and evaluate a task
    result = await env_agent.solve_and_evaluate("mth", task_id=0)
    print(f"Score: {result['score']}")
    
    # Run benchmark
    benchmark = await env_agent.run_benchmark("mth", num_tasks=5)
    print(f"Average: {benchmark['average_score']:.2%}")

asyncio.run(main())
```

#### Batch Processing
```python
challenges = [
    {"env": "mth", "prompt": "What is 2+2?"},
    {"env": "sci", "prompt": "What is H2O?"},
    {"env": "cde", "prompt": "Write a hello world function"},
]

solutions = agent.batch_solve(challenges)
for challenge, solution in zip(challenges, solutions):
    print(f"{challenge['env']}: {solution[:100]}...")
```

### Configuration

#### Using Config File
```python
from config_utils import load_config, create_agent_config_from_dict
from agent import PrimeIntellectAgent

# Load from config.json
config = load_config()
agent_config = create_agent_config_from_dict(config)

agent = PrimeIntellectAgent(agent_config)
```

#### Programmatic Configuration
```python
from agent import PrimeIntellectAgentConfig, PrimeIntellectAgent

config = PrimeIntellectAgentConfig(
    api_key="your-key",
    base_url="https://api.openai.com/v1",
    model="gpt-4o",
    max_tokens=4096,
    temperature=0.7,
    timeout=120.0,
    verbose=True
)

agent = PrimeIntellectAgent(config)
```

## 🎯 Environment-Specific Tips

### CDE (Code)
- Agent returns code in markdown code blocks
- Use `agent.extract_code()` to get clean code
- Best with models that have strong coding capabilities (gpt-4o, claude-3.5-sonnet)

### LGC (Logic)  
- Requires careful reasoning
- Agent shows step-by-step logic
- Consider higher temperature for creative reasoning

### MTH (Math)
- Expects answers in `\boxed{}` format
- Use `agent.extract_boxed_answer()` to get final answer
- Works with rule-based verification + optional judge

### SCI (Science)
- Similar to math, uses `\boxed{}` format
- Covers physics, chemistry, biology, earth science
- Benefits from models with strong scientific knowledge

## 🔧 Advanced Features

### Custom System Prompts
```python
agent.SYSTEM_PROMPTS["mth"] = """Your custom math system prompt..."""
agent.reset_conversation("mth")
```

### Conversation History
```python
# Get history for an environment
history = agent.get_conversation_history("mth")

# Reset specific environment
agent.reset_conversation("mth")

# Clear all histories
agent.clear_all_conversations()
```

### Error Handling
```python
from agent import PrimeIntellectAgentConfig

config = PrimeIntellectAgentConfig(
    max_retries=5,
    retry_delay=2.0,
    timeout=180.0
)
```

## 📊 Examples

See `examples.py` for complete working examples:
```bash
python examples.py
```

## 🐛 Troubleshooting

### "No API key provided"
```bash
export OPENAI_API_KEY="your-key"
# or
python cli.py solve --api-key "your-key" --env mth --prompt "..."
```

### "Module not found"
```bash
pip install -r requirements.txt
```

### "Environment integration not available"
Make sure you're in the correct directory with access to PrimeIntellect environments:
```bash
cd /home/120/affinetes/environments/agents_collection/primeintellect_agent
```

### Rate Limiting
```python
config = PrimeIntellectAgentConfig(
    max_retries=10,
    retry_delay=5.0  # Wait longer between retries
)
```

## 📖 More Information

- See `README.md` for complete documentation
- See `examples.py` for runnable examples  
- See `config.json` for configuration options
- Use `python cli.py --help` for CLI help

## 🎓 Common Use Cases

### Testing Different Models
```bash
python cli.py solve --env mth --model gpt-4o --prompt "..."
python cli.py solve --env mth --model gpt-3.5-turbo --prompt "..."
```

### Comparing Temperatures
```python
for temp in [0.3, 0.7, 1.0]:
    agent = create_agent(temperature=temp)
    solution = agent.solve("...", env="mth")
    print(f"Temp {temp}: {solution}")
```

### Running Evaluations
```bash
# Quick evaluation
python cli.py evaluate --env mth --task-id 0

# Full benchmark
python cli.py benchmark --env mth --num-tasks 100
```

Happy coding! 🚀
