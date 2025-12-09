# BabyAI Agent Quick Start Guide

Get started with the BabyAI LLM Agent in 5 minutes!

## Prerequisites

- Python 3.8+
- OpenAI API key (or compatible API)
- AgentGym repository cloned

## Step 1: Install Dependencies

```bash
# Navigate to the agent directory
cd babyai_agent

# Install agentenv
cd ../../agentenv
pip install -e .

# Install BabyAI environment
cd ../../agentenv-babyai
pip install -e .

# Install additional requirements
cd babyai_agent
pip install requests tqdm
```

## Step 2: Set API Key

### Option A: Environment Variable (Recommended)

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
```

### Option B: Pass as Argument

You can also pass the API key directly when running the script (see Step 4).

## Step 3: Start BabyAI Server

In a separate terminal:

```bash
# Activate the environment
conda activate agentenv-babyai

# Start the server
babyai --host 0.0.0.0 --port 36001
```

Keep this terminal running!

## Step 4: Run the Agent

In the agent directory:

```bash
# Basic run (uses environment variable for API key)
python run_babyai_agent.py

# Or with explicit API key
python run_babyai_agent.py --api_key your-api-key-here

# Run with more tasks
python run_babyai_agent.py --num_tasks 20 --save_conversations
```

## Step 5: View Results

Results are saved in `./babyai_results/`:

```bash
# View summary
cat babyai_results/summary.json

# Or in PowerShell
Get-Content babyai_results/summary.json
```

## Example Output

```
============================================================
BabyAI Agent Evaluation
============================================================
Environment server: http://localhost:36001
Model: gpt-3.5-turbo
Max rounds: 20
Tasks to evaluate: 10
Output directory: ./babyai_results
============================================================
Environment created with ID: env_0
Evaluating tasks: 100%|████████████| 10/10 [02:15<00:00, 13.5s/it]

Progress: 5/10
  Success rate: 60.00%
  Average reward: 0.750

Progress: 10/10
  Success rate: 70.00%
  Average reward: 0.850

Environment closed successfully

============================================================
Evaluation Complete!
============================================================
Total tasks: 10
Successful tasks: 7
Success rate: 70.00%
Average reward: 0.850

Results saved to: babyai_results\summary.json
============================================================
```

## Troubleshooting

### "Failed to create environment"

**Solution:** Make sure the BabyAI server is running:
```bash
babyai --host 0.0.0.0 --port 36001
```

### "API key not provided"

**Solution:** Set the API key:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-key"

# Or pass it directly
python run_babyai_agent.py --api_key your-key
```

### "agentenv package not found"

**Solution:** Install the required packages:
```bash
cd ../../agentenv && pip install -e .
cd ../../agentenv-babyai && pip install -e .
```

## Next Steps

### Use Different Models

```bash
# GPT-4
python run_babyai_agent.py --model gpt-4

# GPT-4 Turbo
python run_babyai_agent.py --model gpt-4-turbo-preview

# Local model (requires local API server)
python run_babyai_agent.py --base_url http://localhost:1234/v1 --model local-model
```

### Customize Agent Behavior

Edit `babyai_agent.py` to:
- Modify the system prompt
- Change generation parameters
- Add custom logic

### Integrate with Your Code

```python
from babyai_agent import BabyAIAgent, BabyAIAgentConfig

# Create agent
agent = BabyAIAgent(BabyAIAgentConfig(
    api_key="your-key",
    model="gpt-3.5-turbo"
))

# Use in your code
action = agent.generate_action("Your observation here")
```

## Tips for Better Performance

1. **Use GPT-4** - Better at following instructions and planning
2. **Lower temperature** - More consistent behavior (0.3-0.5)
3. **Increase max_rounds** - For complex tasks (30-50)
4. **Save conversations** - Debug and improve prompts

## Common Use Cases

### Benchmark Different Models

```bash
# GPT-3.5
python run_babyai_agent.py --model gpt-3.5-turbo --output_dir results_gpt35

# GPT-4
python run_babyai_agent.py --model gpt-4 --output_dir results_gpt4
```

### Evaluate Specific Task Range

```bash
# Tasks 0-49
python run_babyai_agent.py --start_idx 0 --num_tasks 50

# Tasks 50-99
python run_babyai_agent.py --start_idx 50 --num_tasks 50
```

### Debug Failed Tasks

```bash
# Save conversations to see what went wrong
python run_babyai_agent.py --num_tasks 10 --save_conversations

# Then check: babyai_results/conversations/task_X.json
```

## Need Help?

- Check the full README.md for detailed documentation
- Run tests: `python test_babyai_agent.py`
- Check the main AgentGym documentation

Happy agent building! 🤖
