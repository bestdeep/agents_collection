# AgentGym LLM Agents

This directory contains LLM-based agent implementations for various environments in AgentGym. These agents use Large Language Models (via OpenAI-compatible APIs) to interact with and solve tasks in different environments.

## 📋 Overview

AgentGym provides a unified platform for training and evaluating LLM-based agents across diverse environments. This agents directory contains reference implementations that demonstrate how to:

- Build LLM-powered agents for specific environments
- Handle environment observations and generate actions
- Maintain conversation history and context
- Run evaluations and collect results
- Integrate with OpenAI-compatible APIs

## 🤖 Available Agents

### 1. BabyAI Agent

**Environment**: BabyAI navigation tasks  
**Directory**: `babyai_agent/`  
**Description**: An agent that navigates grid-based rooms to accomplish goals like "go to the red ball" or "open the blue door".

**Key Features**:
- Grid world navigation
- Object interaction (doors, keys, balls)
- Goal-oriented behavior
- Visual understanding of room layouts

**Quick Start**:
```bash
cd babyai_agent
python run_babyai_agent.py --config config.json --verbose
```

See [babyai_agent/README.md](babyai_agent/README.md) for detailed documentation.

---

### 2. TextCraft Agent

**Environment**: TextCraft item crafting  
**Directory**: `textcraft_agent/`  
**Description**: An agent that crafts items in a Minecraft-like environment by understanding recipes and managing inventory.

**Key Features**:
- Recipe understanding and planning
- Multi-step crafting sequences
- Inventory management
- Resource gathering

**Quick Start**:
```bash
cd textcraft_agent
python run_textcraft_agent.py --config config.json --verbose
```

See [textcraft_agent/README.md](textcraft_agent/README.md) for detailed documentation.

---

### 3. PrimeIntellect Agent

**Environment**: PrimeIntellect INTELLECT-3-RL (CDE, LGC, MTH, SCI)  
**Directory**: `primeintellect_agent/`  
**Description**: A unified LLM-based agent for solving challenges across four PrimeIntellect environments: code generation, logic puzzles, mathematics, and science questions.

**Key Features**:
- Unified interface for 4 environment types (CDE, LGC, MTH, SCI)
- Environment-specific intelligent prompting
- Conversation history management
- Response parsing (code blocks and boxed answers)
- Direct integration with PrimeIntellect evaluators
- Result saving with conversation history and scores
- CLI and Python API support

**Quick Start**:
```bash
cd primeintellect_agent

# Single evaluation
python cli.py --api-key YOUR_KEY --model gpt-4o evaluate --env mth --task-id 0

# Benchmark (10 tasks)
python cli.py --api-key YOUR_KEY --model gpt-4o benchmark --env mth --num-tasks 10

# Save results
python cli.py --save --output-dir results --api-key YOUR_KEY --model gpt-4o benchmark --env cde --num-tasks 5
```

See [primeintellect_agent/README.md](primeintellect_agent/README.md) for detailed documentation.

---

## 🚀 Getting Started

### Prerequisites

1. **Install AgentGym environments**:
```bash
# Install the main agentenv package
cd ../agentenv
pip install -e .

# Install specific environment (example: BabyAI)
cd ../agentenv-babyai
pip install -e .
```

2. **Install common dependencies**:
```bash
pip install requests openai
```

3. **Set up API credentials**:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional
```

### Running an Agent

Each agent follows a similar pattern:

1. **Start the environment server**:
```bash
# Example for BabyAI
babyai --host 0.0.0.0 --port 36001

# Example for TextCraft
textcraft --host 0.0.0.0 --port 36001
```

2. **Configure the agent**:
   - Copy `config_example.json` to `config.json`
   - Edit with your API key and preferences

3. **Run the agent**:
```bash
python run_<agent_name>_agent.py --config config.json --verbose
```

## 📁 Agent Structure

Each agent directory typically contains:

```
agent_name/
├── README.md                    # Detailed documentation
├── config_example.json          # Example configuration
├── <agent_name>_agent.py       # Main agent implementation
└── run_<agent_name>_agent.py   # Evaluation script
```

### Core Components

All agents implement similar core functionality:

#### 1. Agent Configuration
```python
@dataclass
class AgentConfig:
    api_key: str
    base_url: str
    model: str
    max_tokens: int
    temperature: float
    max_rounds: int
```

#### 2. Agent Class
```python
class Agent:
    def __init__(self, config: AgentConfig)
    def generate_action(self, observation: str) -> str
    def run_episode(self, env_url: str, env_id: int, data_idx: int) -> Dict
    def save_conversation(self, filepath: str)
```

#### 3. Evaluation Script
- Batch processing of multiple tasks
- Results tracking and saving
- Success rate calculation
- Conversation history logging

## 🎯 Configuration

All agents use JSON configuration files with three main sections:

### Agent Configuration
```json
{
  "agent": {
    "api_key": "your-api-key-here",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-3.5-turbo",
    "max_tokens": 512,
    "temperature": 0.7,
    "max_rounds": 50
  }
}
```

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

### Evaluation Configuration
```json
{
  "evaluation": {
    "num_tasks": 10,
    "start_idx": 0,
    "output_dir": "./results",
    "save_conversations": true
  }
}
```

## 📊 Results and Output

Evaluation results are saved in timestamped directories:

```
results/
└── 20231209_143022/
    ├── evaluation_summary.json      # Overall statistics
    ├── task_0_conversation.json     # Task 0 conversation history
    ├── task_1_conversation.json     # Task 1 conversation history
    └── ...
```

**Summary includes**:
- Total tasks evaluated
- Success count and rate
- Individual task results
- Configuration used

## 🛠️ Developing New Agents

To create an agent for a new environment:

1. **Create agent directory**:
```bash
mkdir agents/new_agent
cd agents/new_agent
```

2. **Implement the agent class**:
   - Study existing agents (BabyAI, TextCraft) as references
   - Define environment-specific system prompt
   - Implement action parsing for the environment
   - Handle environment-specific observations

3. **Create configuration template**:
   - Copy `config_example.json` from existing agent
   - Adjust parameters for your environment

4. **Implement evaluation script**:
   - Copy `run_*_agent.py` from existing agent
   - Adapt for your environment's API

5. **Write documentation**:
   - Create README.md with setup and usage instructions
   - Document environment-specific actions and observations

### Key Considerations

- **System Prompt**: Design a clear system prompt that explains the environment, available actions, and response format
- **Action Parsing**: Implement robust parsing to extract actions from LLM responses
- **Error Handling**: Handle API failures, rate limits, and environment errors gracefully
- **Conversation Management**: Maintain context while avoiding token limit issues

## 🔧 Supported LLM Providers

All agents support any OpenAI-compatible API:

### OpenAI
```bash
$env:OPENAI_API_KEY="sk-..."
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
```

### Azure OpenAI
```bash
$env:OPENAI_API_KEY="your-azure-key"
$env:OPENAI_BASE_URL="https://your-resource.openai.azure.com/openai/deployments/your-deployment"
```

### Local Models (via OpenAI-compatible servers)
```bash
$env:OPENAI_API_KEY="not-needed"
$env:OPENAI_BASE_URL="http://localhost:8000/v1"
```

Popular local serving options:
- [vLLM](https://github.com/vllm-project/vllm)
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui)
- [LocalAI](https://github.com/mudler/LocalAI)
- [Ollama](https://ollama.ai/) (with OpenAI compatibility)

## 📈 Performance Tips

1. **Model Selection**:
   - GPT-4 generally performs better but is more expensive
   - GPT-3.5-turbo offers good balance of cost and performance
   - Fine-tuned models can be more efficient for specific tasks

2. **Temperature Settings**:
   - Lower (0.1-0.3): More deterministic, better for structured tasks
   - Medium (0.5-0.7): Balanced exploration and exploitation
   - Higher (0.8-1.0): More creative, better for open-ended tasks

3. **Max Rounds**:
   - Adjust based on task complexity
   - Too low: Agent may not complete tasks
   - Too high: Inefficient, may loop unnecessarily

4. **Retry Logic**:
   - Exponential backoff handles rate limits automatically
   - Adjust `max_retries` and `retry_delay` based on your API limits

## 🐛 Troubleshooting

### Environment Server Not Running
```
Error: Could not connect to environment server
```
**Solution**: Start the environment server first (see environment-specific README).

### API Authentication Errors
```
Error: 401 Unauthorized
```
**Solution**: Check your API key is set correctly:
```bash
echo $env:OPENAI_API_KEY  # Should print your key
```

### Rate Limiting
```
Error: 429 Too Many Requests
```
**Solution**: The agent automatically retries with exponential backoff. If persistent:
- Reduce concurrent evaluations
- Upgrade API plan
- Increase `retry_delay` in config

### Token Limit Exceeded
```
Error: Maximum context length exceeded
```
**Solution**:
- Reduce `max_tokens` in config
- Implement conversation history truncation
- Use a model with larger context window

## 📚 Additional Resources

- **AgentGym Paper**: [arXiv:2406.04151](https://arxiv.org/abs/2406.04151)
- **Project Page**: [agentgym.github.io](https://agentgym.github.io/)
- **Main Repository**: [github.com/WooooDyy/AgentGym](https://github.com/WooooDyy/AgentGym)
- **Trajectory Dataset**: [AgentGym/AgentTraj-L](https://huggingface.co/datasets/AgentGym/AgentTraj-L)
- **Evaluation Benchmark**: [AgentGym/AgentEval](https://huggingface.co/datasets/AgentGym/AgentEval)

## 🤝 Contributing

We welcome contributions of new agents! To contribute:

1. Fork the repository
2. Create an agent following the structure above
3. Test thoroughly on the target environment
4. Submit a pull request with:
   - Complete agent implementation
   - Configuration examples
   - Comprehensive README
   - Example results

## 📝 Citation

If you use these agents in your research, please cite:

```bibtex
@article{xi2024agentgym,
  title={AgentGym: Evolving Large Language Model-based Agents across Diverse Environments},
  author={Xi, Zhiheng and others},
  journal={arXiv preprint arXiv:2406.04151},
  year={2024}
}
```

## 📄 License

See the main AgentGym repository for license information.
