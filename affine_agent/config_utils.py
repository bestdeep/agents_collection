"""
Configuration utilities for Affine Agent
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from agent import AffineAgentConfig


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


def create_agent_config_from_dict(config_dict: Dict[str, Any]) -> AffineAgentConfig:
    """
    Create AffineAgentConfig from configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary (usually from load_config)
        
    Returns:
        AffineAgentConfig instance
    """
    agent_config = config_dict.get("agent", {})
    
    return AffineAgentConfig(
        api_key=agent_config.get("api_key", os.getenv("OPENAI_API_KEY", "")),
        base_url=agent_config.get("base_url", "https://api.openai.com/v1"),
        model=agent_config.get("model", "gpt-4o"),
        max_tokens=agent_config.get("max_tokens", 4096),
        temperature=agent_config.get("temperature", 0.7),
        top_p=agent_config.get("top_p", 1.0),
        timeout=agent_config.get("timeout", 120.0),
        max_retries=agent_config.get("max_retries", 3),
        retry_delay=agent_config.get("retry_delay", 2.0),
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
            "verbose": False
        },
        "environments": {
            "sat": {
                "n": 15,
                "k": 10
            },
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


if __name__ == "__main__":
    # Example usage
    config = create_default_config()
    save_config(config, "config_example.json")
    print("Created example config file: config_example.json")
