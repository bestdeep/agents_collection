# PrimeIntellect Agent - Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PrimeIntellect Agent System                      │
│                  (Unified LLM Agent for INTELLECT-3-RL)            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│  │   CLI Tool   │    │  Python API  │    │  Interactive │         │
│  │   (cli.py)   │    │ (agent.py)   │    │     Mode     │         │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘         │
│         │                    │                    │                  │
│         └────────────────────┴────────────────────┘                  │
│                              │                                       │
└──────────────────────────────┼───────────────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────────────┐
│                              │     CORE AGENT                         │
├──────────────────────────────┼───────────────────────────────────────┤
│                              ▼                                        │
│         ┌──────────────────────────────────────┐                    │
│         │   PrimeIntellectAgent (agent.py)     │                    │
│         │  ┌────────────────────────────────┐  │                    │
│         │  │  Environment-Specific Prompts  │  │                    │
│         │  │  • CDE (Code)                  │  │                    │
│         │  │  • LGC (Logic)                 │  │                    │
│         │  │  • MTH (Math)                  │  │                    │
│         │  │  • SCI (Science)               │  │                    │
│         │  └────────────────────────────────┘  │                    │
│         │  ┌────────────────────────────────┐  │                    │
│         │  │  Core Functionality            │  │                    │
│         │  │  • solve()                     │  │                    │
│         │  │  • batch_solve()               │  │                    │
│         │  │  • extract_code()              │  │                    │
│         │  │  • extract_boxed_answer()      │  │                    │
│         │  │  • conversation management     │  │                    │
│         │  └────────────────────────────────┘  │                    │
│         └──────────────────────────────────────┘                    │
│                              │                                        │
└──────────────────────────────┼───────────────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────────────┐
│                              │  ENVIRONMENT INTEGRATION               │
├──────────────────────────────┼───────────────────────────────────────┤
│                              ▼                                        │
│    ┌──────────────────────────────────────────────────────┐         │
│    │  PrimeIntellectEnvironmentAgent (env_integration.py) │         │
│    │  ┌────────────────────────────────────────────────┐  │         │
│    │  │  • initialize_environment()                    │  │         │
│    │  │  • generate_task()                             │  │         │
│    │  │  • solve_and_evaluate()                        │  │         │
│    │  │  • run_benchmark()                             │  │         │
│    │  └────────────────────────────────────────────────┘  │         │
│    └──────────────────────────────────────────────────────┘         │
│                              │                                        │
│              ┌───────────────┼───────────────┐                       │
│              │               │               │                       │
│              ▼               ▼               ▼                       │
│    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐              │
│    │ Task Gens    │ │  Evaluators  │ │   Datasets   │              │
│    │ • CodeTask   │ │ • Code exec  │ │ INTELLECT-3  │              │
│    │ • LogicTask  │ │ • Verifiers  │ │      -RL     │              │
│    │ • MathTask   │ │ • Math check │ │   • code     │              │
│    │ • ScienceTask│ │ • Judge LLM  │ │   • logic    │              │
│    └──────────────┘ └──────────────┘ │   • math     │              │
│                                       │   • science  │              │
│                                       └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       CONFIGURATION LAYER                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐      ┌──────────────────┐                    │
│  │  config.json     │◄─────┤ config_utils.py  │                    │
│  │  • agent config  │      │ • load_config()  │                    │
│  │  • env configs   │      │ • save_config()  │                    │
│  │  • env variables │      │ • defaults       │                    │
│  └──────────────────┘      └──────────────────┘                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         LLM API LAYER                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│         ┌────────────────────────────────────────────┐              │
│         │  OpenAI-Compatible API (requests)          │              │
│         │  • Retry logic with exponential backoff    │              │
│         │  • Rate limit handling                     │              │
│         │  • Timeout management                      │              │
│         │  • Error recovery                          │              │
│         └────────────────┬───────────────────────────┘              │
│                          │                                           │
│              ┌───────────┴───────────┐                              │
│              ▼                       ▼                              │
│    ┌──────────────────┐    ┌──────────────────┐                    │
│    │  OpenAI API      │    │  Custom APIs     │                    │
│    │  • GPT-4o        │    │  • Claude        │                    │
│    │  • GPT-4         │    │  • Local models  │                    │
│    │  • GPT-3.5       │    │  • vLLM          │                    │
│    └──────────────────┘    └──────────────────┘                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘


════════════════════════════════════════════════════════════════════════
                          DATA FLOW EXAMPLE
════════════════════════════════════════════════════════════════════════

1. USER REQUEST
   │
   └─► "Solve: What is 2 + 2?"
       │
       ▼
2. CLI/API ENTRY
   │
   └─► cli.py or agent.solve()
       │
       ▼
3. AGENT PROCESSING
   │
   ├─► Select environment (MTH)
   ├─► Load system prompt
   ├─► Format user message
   └─► Add to conversation history
       │
       ▼
4. LLM API CALL
   │
   └─► POST to /v1/chat/completions
       │
       ├─► [Retry if needed]
       ├─► [Handle rate limits]
       └─► Get response
           │
           ▼
5. RESPONSE PROCESSING
   │
   ├─► Parse response
   ├─► Extract boxed answer: "4"
   └─► Update conversation history
       │
       ▼
6. EVALUATION (Optional)
   │
   └─► env_integration.py
       │
       ├─► Generate task from dataset
       ├─► Get agent solution
       ├─► Run evaluator
       └─► Return score
           │
           ▼
7. RESULT
   │
   └─► Return to user: "4"


════════════════════════════════════════════════════════════════════════
                          FILE DEPENDENCIES
════════════════════════════════════════════════════════════════════════

agent.py
 ├─► requests (HTTP client)
 ├─► dataclasses (config)
 └─► re (regex parsing)

env_integration.py
 ├─► agent.py (PrimeIntellectAgent)
 ├─► asyncio (async operations)
 ├─► code_task.py (CDE environment)
 ├─► logic_task.py (LGC environment)
 ├─► math_task.py (MTH environment)
 └─► sci_task.py (SCI environment)

cli.py
 ├─► agent.py (core agent)
 ├─► config_utils.py (configuration)
 ├─► env_integration.py (optional)
 └─► argparse (CLI parsing)

config_utils.py
 ├─► agent.py (PrimeIntellectAgentConfig)
 └─► json (config parsing)

examples.py
 ├─► agent.py (basic examples)
 └─► env_integration.py (advanced examples)

test_agent.py
 ├─► agent.py (testing core)
 ├─► config_utils.py (testing config)
 └─► env_integration.py (testing integration)


════════════════════════════════════════════════════════════════════════
                      ENVIRONMENT SPECIFICS
════════════════════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────────────┐
│ CDE (Code)                                                          │
├────────────────────────────────────────────────────────────────────┤
│ Input:  Programming challenge description                          │
│ Output: Python code in ```python blocks                           │
│ Eval:   Code execution with test cases                            │
│ Prompt: Expert Python programmer                                   │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ LGC (Logic)                                                         │
├────────────────────────────────────────────────────────────────────┤
│ Input:  Logic puzzle or reasoning task                            │
│ Output: Reasoned natural language answer                          │
│ Eval:   Task-specific verifiers                                   │
│ Prompt: Expert in logic and reasoning                             │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ MTH (Math)                                                          │
├────────────────────────────────────────────────────────────────────┤
│ Input:  Mathematical problem                                       │
│ Output: Solution with \boxed{answer}                              │
│ Eval:   Rule-based verification + optional judge                  │
│ Prompt: Expert mathematician                                       │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ SCI (Science)                                                       │
├────────────────────────────────────────────────────────────────────┤
│ Input:  Science question (physics/chem/bio/earth)                 │
│ Output: Solution with \boxed{answer}                              │
│ Eval:   Verification + optional judge                             │
│ Prompt: Expert in science                                          │
└────────────────────────────────────────────────────────────────────┘


════════════════════════════════════════════════════════════════════════
                          QUICK COMMANDS
════════════════════════════════════════════════════════════════════════

# Installation
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"

# Testing
python test_agent.py

# CLI Usage
python cli.py solve --env mth --prompt "What is 2+2?"
python cli.py interactive --env cde
python cli.py evaluate --env sci --task-id 0
python cli.py benchmark --env lgc --num-tasks 10

# Python Usage
from agent import create_agent
agent = create_agent(model="gpt-4o")
solution = agent.solve("Your question", env="mth")

# Environment Integration
import asyncio
from env_integration import evaluate_agent
results = asyncio.run(evaluate_agent("mth", num_tasks=10))


════════════════════════════════════════════════════════════════════════
