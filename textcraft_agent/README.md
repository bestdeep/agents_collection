# TextCraft LLM Agent

An LLM-based agent for the TextCraft environment in AgentGym. This agent uses Large Language Models (via OpenAI-compatible APIs) to craft items in a Minecraft-like crafting environment.

## Overview

The TextCraft agent is designed to solve crafting tasks by:
- Understanding crafting recipes provided in the environment
- Planning a sequence of actions to craft target items
- Managing inventory and tracking available materials
- Making decisions about what to get and what to craft

## Features

- **LLM-powered decision making**: Uses GPT or compatible models to generate actions
- **Conversation history**: Maintains context across multiple steps
- **Flexible configuration**: Easy to customize model, parameters, and evaluation settings
- **Comprehensive evaluation**: Run batch evaluations and save detailed results
- **Error handling**: Robust retry logic for API calls

## Installation

1. **Install the AgentGym core package**:
```bash
cd ../../agentenv
pip install -e .
```

2. **Install the TextCraft environment**:
```bash
cd ../agentenv-textcraft
pip install -e .
```

3. **Install required dependencies for the agent**:
```bash
pip install requests
```

## Setup

1. Set your OpenAI API key:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
```

2. (Optional) Configure a custom API endpoint:
```bash
# Windows PowerShell
$env:OPENAI_BASE_URL="https://your-api-endpoint.com/v1"

# Linux/Mac
export OPENAI_BASE_URL="https://your-api-endpoint.com/v1"
```

3. Copy and customize the configuration file:
```bash
cp config_example.json config.json
# Edit config.json with your settings
```

## Usage

### Starting the TextCraft Environment

First, start the TextCraft environment server:

```bash
textcraft --host 0.0.0.0 --port 36001
```

### Running the Agent

#### Quick Test
Test the agent with a single example:
```bash
python textcraft_agent.py
```

#### Full Evaluation
Run evaluation on multiple tasks:
```bash
# Use default config
python run_textcraft_agent.py --config config.json --verbose

# Run specific number of tasks
python run_textcraft_agent.py --config config.json --num-tasks 5 --verbose

# Start from a specific task index
python run_textcraft_agent.py --config config.json --start-idx 10 --num-tasks 10
```

### Command Line Arguments

- `--config`: Path to configuration JSON file (default: `config_example.json`)
- `--verbose`: Print detailed progress during evaluation
- `--num-tasks`: Override the number of tasks to run
- `--start-idx`: Override the starting task index

## Configuration

The configuration file (`config.json`) has three main sections:

### Agent Configuration
```json
{
  "agent": {
    "api_key": "your-api-key-here",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-3.5-turbo",
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 1.0,
    "timeout": 60.0,
    "max_rounds": 50,
    "max_retries": 3,
    "retry_delay": 1.0
  }
}
```

- `api_key`: Your OpenAI API key (can use environment variable)
- `base_url`: API endpoint URL
- `model`: Model name (e.g., "gpt-3.5-turbo", "gpt-4")
- `max_tokens`: Maximum tokens per generation
- `temperature`: Sampling temperature (0.0-1.0)
- `max_rounds`: Maximum steps per episode
- `max_retries`: Number of API call retries
- `retry_delay`: Delay between retries (seconds)

### Environment Configuration
```json
{
  "environment": {
    "env_server": "http://localhost:36001",
    "data_len": 200,
    "timeout": 300
  }
}
```

- `env_server`: URL of the TextCraft environment server
- `data_len`: Total number of available tasks
- `timeout`: Request timeout (seconds)

### Evaluation Configuration
```json
{
  "evaluation": {
    "num_tasks": 10,
    "start_idx": 0,
    "output_dir": "./textcraft_results",
    "save_conversations": true
  }
}
```

- `num_tasks`: Number of tasks to evaluate
- `start_idx`: Starting task index
- `output_dir`: Directory to save results
- `save_conversations`: Whether to save conversation histories

## Action Space

The agent can perform three types of actions:

1. **Get items**: `get <amount> <item>`
   - Example: `get 1 oak log`
   - Used for obtaining raw/basic materials

2. **Craft items**: `craft <output> using <inputs>`
   - Example: `craft 4 oak planks using 1 oak log`
   - Example: `craft 1 wooden pickaxe using 3 oak planks, 2 sticks`
   - Used for crafting items according to recipes

3. **Check inventory**: `inventory`
   - Shows current items in inventory

## Output

Results are saved in the specified output directory with timestamps:

```
textcraft_results/
└── 20231209_143022/
    ├── evaluation_summary.json
    ├── task_0_conversation.json
    ├── task_1_conversation.json
    └── ...
```

- `evaluation_summary.json`: Overall results and statistics
- `task_X_conversation.json`: Full conversation history for each task

## Example

```python
from textcraft_agent import TextCraftAgent, TextCraftAgentConfig

# Create agent
config = TextCraftAgentConfig(
    api_key="your-api-key",
    model="gpt-3.5-turbo",
    max_rounds=50
)
agent = TextCraftAgent(config)

# Run on a single task
env_url = "http://localhost:36001"
env_id = 0  # Environment instance ID
data_idx = 0  # Task index

result = agent.run_episode(env_url, env_id, data_idx, verbose=True)

print(f"Success: {result['success']}")
print(f"Steps: {result['steps']}")
print(f"Reward: {result['reward']}")
```

## Troubleshooting

### Environment Server Not Running
```
Error: Could not connect to environment server
```
**Solution**: Start the TextCraft server first:
```bash
textcraft --host 0.0.0.0 --port 36001
```

### API Key Issues
```
Error: Authentication failed
```
**Solution**: Set your API key properly:
```bash
$env:OPENAI_API_KEY="your-actual-api-key"
```

### Rate Limiting
The agent automatically handles rate limiting with exponential backoff. If you see rate limit messages, the agent will retry automatically.

## Advanced Usage

### Custom System Prompt
You can modify the `SYSTEM_PROMPT` in `textcraft_agent.py` to change how the agent reasons about tasks.

### Custom Action Parsing
Override the `parse_action` method to customize how actions are extracted from LLM responses.

### Integration with Other LLMs
The agent works with any OpenAI-compatible API. Just set the `base_url` and ensure your API follows the OpenAI format.

## License

See the main AgentGym repository for license information.
