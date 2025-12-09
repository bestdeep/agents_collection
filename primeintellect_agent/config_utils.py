"""
Configuration utilities for PrimeIntellect Agent
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from agent import PrimeIntellectAgentConfig


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config file. If None, looks for config.json in current directory.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.json"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Replace environment variables in config
    config = _replace_env_vars(config)
    
    return config


def _replace_env_vars(config: Any) -> Any:
    """
    Recursively replace environment variable references in config.
    
    Format: ${VAR_NAME}
    """
    if isinstance(config, dict):
        return {k: _replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_replace_env_vars(item) for item in config]
    elif isinstance(config, str):
        if config.startswith("${") and config.endswith("}"):
            var_name = config[2:-1]
            return os.getenv(var_name, config)
        return config
    else:
        return config


def create_agent_config_from_dict(config_dict: Dict[str, Any]) -> PrimeIntellectAgentConfig:
    """
    Create PrimeIntellectAgentConfig from configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary (usually from load_config)
        
    Returns:
        PrimeIntellectAgentConfig instance
    """
    agent_config = config_dict.get("agent", {})
    
    return PrimeIntellectAgentConfig(
        api_key=agent_config.get("api_key", os.getenv("OPENAI_API_KEY", "")),
        base_url=agent_config.get("base_url", "https://api.openai.com/v1"),
        model=agent_config.get("model", "gpt-4o"),
        max_tokens=agent_config.get("max_tokens", 4096),
        temperature=agent_config.get("temperature", 0.7),
        top_p=agent_config.get("top_p", 1.0),
        timeout=agent_config.get("timeout", 120.0),
        max_retries=agent_config.get("max_retries", 3),
        retry_delay=agent_config.get("retry_delay", 2.0),
        use_thinking=agent_config.get("use_thinking", True),
        verbose=agent_config.get("verbose", False)
    )


def get_env_configs(config_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract environment configurations from config dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Dictionary of environment configs
    """
    return config_dict.get("environments", {})


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration dictionary.
    
    Returns:
        Default configuration
    """
    return {
        "agent": {
            "api_key": "${OPENAI_API_KEY}",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o",
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 1.0,
            "timeout": 120.0,
            "max_retries": 3,
            "retry_delay": 2.0,
            "use_thinking": True,
            "verbose": False
        },
        "environments": {
            "cde": {
                "dataset_name": "PrimeIntellect/INTELLECT-3-RL",
                "dataset_subset": "code",
                "dataset_split": "train",
                "dataset_shuffle": False,
                "difficulty_key": "avg@8_qwen3_4b_instruct_2507",
                "min_avg_reward": 0.0,
                "max_avg_reward": 1.0
            },
            "lgc": {
                "dataset_name": "PrimeIntellect/INTELLECT-3-RL",
                "dataset_subset": "logic",
                "dataset_split": "train",
                "dataset_shuffle": False,
                "difficulty_key": "avg@16_qwen3_4b_instruct_2507",
                "min_avg_reward": 0.0,
                "max_avg_reward": 1.0,
                "tasks_to_skip": ["arc_agi", "arc_agi_2", "buggy_tables"]
            },
            "mth": {
                "dataset_name": "PrimeIntellect/INTELLECT-3-RL",
                "dataset_subset": "math",
                "dataset_split": "train",
                "dataset_shuffle": False,
                "difficulty_key": "avg@8_qwen3_4b_thinking_2507",
                "min_avg_reward": 0.0,
                "max_avg_reward": 1.0,
                "judge_model": None,
                "judge_base_url": None,
                "judge_api_key": None,
                "judge_sampling_args": {}
            },
            "sci": {
                "dataset_name": "PrimeIntellect/INTELLECT-3-RL",
                "dataset_subset": "science",
                "dataset_split": "train",
                "dataset_shuffle": False,
                "difficulty_key": "avg@8_qwen3_4b_instruct_2507",
                "min_avg_reward": 0.0,
                "max_avg_reward": 1.0,
                "judge_model": None,
                "judge_base_url": None,
                "judge_api_key": None,
                "judge_sampling_args": {}
            }
        }
    }


# Example usage
if __name__ == "__main__":
    # Load config from file
    try:
        config = load_config()
        print("Configuration loaded successfully!")
        print(f"Model: {config['agent']['model']}")
        print(f"Environments: {list(config['environments'].keys())}")
        
        # Create agent config
        agent_config = create_agent_config_from_dict(config)
        print(f"\nAgent Config:")
        print(f"  Model: {agent_config.model}")
        print(f"  Temperature: {agent_config.temperature}")
        print(f"  Max Tokens: {agent_config.max_tokens}")
        
        # Get environment configs
        env_configs = get_env_configs(config)
        print(f"\nEnvironment Configs:")
        for env_name, env_config in env_configs.items():
            print(f"  {env_name.upper()}: {env_config.get('dataset_subset')}")
            
    except FileNotFoundError:
        print("Config file not found. Creating default config...")
        default_config = create_default_config()
        save_config(default_config, "config.json")
        print("Default config created at config.json")
