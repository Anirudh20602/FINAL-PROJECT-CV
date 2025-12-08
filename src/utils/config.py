"""
Configuration management utilities
"""

import os
import yaml
from easydict import EasyDict
from typing import Dict, Any


def load_config(config_path: str) -> EasyDict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration as EasyDict
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return EasyDict(config)


def save_config(config: Dict[Any, Any], save_path: str) -> None:
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(dict(config), f, default_flow_style=False)


def merge_configs(base_config: EasyDict, override_config: EasyDict) -> EasyDict:
    """
    Merge two configurations (override takes precedence)
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = EasyDict(base_config.copy())
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(EasyDict(merged[key]), EasyDict(value))
        else:
            merged[key] = value
    
    return merged


def get_config(config_path: str, overrides: Dict[str, Any] = None) -> EasyDict:
    """
    Load configuration with optional overrides
    
    Args:
        config_path: Path to configuration file
        overrides: Optional dictionary of overrides
        
    Returns:
        Configuration as EasyDict
    """
    config = load_config(config_path)
    
    if overrides:
        config = merge_configs(config, EasyDict(overrides))
    
    return config


def print_config(config: EasyDict, indent: int = 0) -> None:
    """
    Pretty print configuration
    
    Args:
        config: Configuration to print
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(EasyDict(value), indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")
