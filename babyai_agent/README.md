# BabyAI LLM Agent

A complete LLM-based agent implementation for the BabyAI environment in AgentGym.

## Overview

This agent uses an OpenAI-compatible API (e.g., GPT-3.5, GPT-4, or other compatible models) to navigate and complete tasks in the BabyAI environment. The agent maintains conversation history and generates thoughtful actions based on observations.

## Files

- `babyai_agent.py` - Main agent implementation
- `run_babyai_agent.py` - Evaluation script
- `README.md` - This file

## Features

- 🤖 OpenAI-compatible API integration
- 💬 Conversation history management
- 🔄 Automatic retry logic with exponential backoff
- 📊 Episode tracking and results saving
- 🎯 Configurable generation parameters
- 📝 Optional conversation history saving

## Installation

### 1. Install AgentGym dependencies

```bash
# Install the main agentenv package
cd ../agentenv
pip install -e .

# Install BabyAI environment
cd ../agentenv-babyai
pip install -e .
```

### 2. Install additional dependencies

```bash
pip install requests tqdm
```

### 3. Set up environment variables

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Optional: Set custom API base URL (for Azure, local models, etc.)
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

## Usage

### Starting the BabyAI Server

First, start the BabyAI environment server:

```bash
babyai --host 0.0.0.0 --port 36001
```

### Running the Agent

#### Basic Usage

```bash
python run_babyai_agent.py --api_key YOUR_API_KEY
```

#### Advanced Usage

```bash
python run_babyai_agent.py \
    --env_server http://localhost:36001 \
    --api_key YOUR_API_KEY \
    --model gpt-4 \
    --max_rounds 30 \
    --num_tasks 50 \
    --output_dir ./results \
    --save_conversations
```

### Command Line Arguments

#### Environment Settings
- `--env_server` - BabyAI server URL (default: `http://localhost:36001`)
- `--data_len` - Number of tasks in dataset (default: `200`)
- `--timeout` - Request timeout in seconds (default: `300`)

#### API Settings
- `--api_key` - OpenAI API key (or set `OPENAI_API_KEY` env var)
- `--base_url` - API base URL (default: `https://api.openai.com/v1`)
- `--model` - Model to use (default: `gpt-3.5-turbo`)

#### Agent Settings
- `--max_tokens` - Maximum tokens for generation (default: `512`)
- `--temperature` - Temperature for generation (default: `0.7`)
- `--max_rounds` - Maximum rounds per episode (default: `20`)

#### Evaluation Settings
- `--num_tasks` - Number of tasks to evaluate (default: `10`)
- `--start_idx` - Starting task index (default: `0`)
- `--output_dir` - Directory to save results (default: `./babyai_results`)
- `--save_conversations` - Save full conversation histories

## Using the Agent Programmatically

```python
from babyai_agent import BabyAIAgent, BabyAIAgentConfig
from agentenv.envs import BabyAIEnvClient

# Configure the agent
config = BabyAIAgentConfig(
    api_key="your-api-key",
    model="gpt-3.5-turbo",
    max_rounds=20,
    temperature=0.7
)

# Initialize agent
agent = BabyAIAgent(config)

# Initialize environment client
env_client = BabyAIEnvClient(
    env_server_base="http://localhost:36001",
    data_len=200,
    timeout=300
)

# Create environment instance
env_idx = env_client.create()

# Run an episode
result = agent.run_episode(env_client, env_idx, data_idx=0)

print(f"Success: {result['success']}")
print(f"Reward: {result['reward']}")
print(f"Steps: {result['steps']}")

# Clean up
env_client.close(env_idx)
```

### Single Action Generation

```python
from babyai_agent import BabyAIAgent, BabyAIAgentConfig

# Initialize agent
agent = BabyAIAgent(BabyAIAgentConfig(api_key="your-api-key"))

# Generate action from observation
observation = "In front of you in this room, you can see several objects: There is a red ball 1 right in front of you 3 steps away."
goal = "Go to the red ball"

action = agent.generate_action(observation, goal=goal)
print(action)
```

## Agent Actions

The agent can use the following actions:

- `turn right` - Turn 90 degrees to the right
- `turn left` - Turn 90 degrees to the left
- `move forward` - Move one step forward
- `go to <obj> <id>` - Navigate to a specific object (e.g., "go to red ball 1")
- `pick up <obj> <id>` - Pick up an object
- `go through <door> <id>` - Go through an open door
- `toggle and go through <door> <id>` - Open and go through a closed/locked door
- `toggle` - Toggle a door right in front of you

## Output Format

The agent generates responses in the following format:

```
Thought:
I need to navigate to the red ball that is 3 steps ahead.

Action:
go to red ball 1
```

## Results

Results are saved in JSON format:

```json
{
  "config": {
    "model": "gpt-3.5-turbo",
    "max_rounds": 20,
    "num_tasks": 10
  },
  "results": {
    "total_tasks": 10,
    "total_success": 7,
    "success_rate": 0.7,
    "average_reward": 0.85,
    "tasks": [
      {
        "task_idx": 0,
        "success": true,
        "reward": 1.0,
        "steps": 5,
        "done": true
      }
    ]
  }
}
```

## Troubleshooting

### Environment Server Not Running

If you get connection errors, make sure the BabyAI server is running:

```bash
babyai --host 0.0.0.0 --port 36001
```

### API Key Issues

Make sure your API key is set correctly:

```bash
export OPENAI_API_KEY="your-api-key"
```

Or pass it directly:

```bash
python run_babyai_agent.py --api_key your-api-key
```

### Rate Limiting

The agent includes automatic retry logic with exponential backoff for rate limiting. You can adjust the retry parameters in `BabyAIAgentConfig`:

```python
config = BabyAIAgentConfig(
    max_retries=5,
    retry_delay=2.0
)
```

### Using Alternative Models

You can use any OpenAI-compatible API:

```python
# Azure OpenAI
config = BabyAIAgentConfig(
    api_key="your-azure-key",
    base_url="https://your-resource.openai.azure.com/openai/deployments/your-deployment",
    model="gpt-35-turbo"
)

# Local model (e.g., LM Studio)
config = BabyAIAgentConfig(
    api_key="not-needed",
    base_url="http://localhost:1234/v1",
    model="local-model"
)
```

## Examples

### Evaluate on 50 tasks with GPT-4

```bash
python run_babyai_agent.py \
    --model gpt-4 \
    --num_tasks 50 \
    --output_dir ./gpt4_results \
    --save_conversations
```

### Use a local model

```bash
python run_babyai_agent.py \
    --base_url http://localhost:1234/v1 \
    --api_key local \
    --model local-model \
    --num_tasks 10
```

### Evaluate specific task range

```bash
python run_babyai_agent.py \
    --start_idx 100 \
    --num_tasks 20 \
    --output_dir ./results_100_120
```

## Performance Tips

1. **Temperature**: Lower temperature (0.3-0.5) for more deterministic behavior
2. **Max Rounds**: Increase for more complex tasks
3. **Model**: GPT-4 generally performs better than GPT-3.5 but is more expensive
4. **Batch Evaluation**: Run multiple parallel evaluations with different task ranges

## Citation

If you use this agent in your research, please cite the AgentGym paper:

```bibtex
@article{agentgym,
  title={AgentGym: Evolving Large Language Model-based Agents across Diverse Environments},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This project follows the AgentGym license. See the main repository LICENSE file for details.
