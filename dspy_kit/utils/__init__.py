"""Shared utilities for dspy-kit modules.

This module provides common utilities used across evaluation, synthetic data generation,
and red teaming modules.
"""

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Version info
__version__ = "0.1.0"


# Common utility functions
def setup_logging(level: str = "INFO", format_string: Optional[str] = None) -> logging.Logger:
    """Setup logging for dspy-kit modules."""
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(level=getattr(logging, level.upper()), format=format_string)
    return logging.getLogger("dspy_kit")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix.lower() == ".json":
        with open(config_path) as f:
            return json.load(f)
    elif config_path.suffix.lower() in [".yaml", ".yml"]:
        try:
            import yaml

            with open(config_path) as f:
                return yaml.safe_load(f)
        except ImportError as err:
            raise ImportError("PyYAML is required to load YAML config files") from err
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to JSON or YAML file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path.suffix.lower() == ".json":
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    elif config_path.suffix.lower() in [".yaml", ".yml"]:
        try:
            import yaml

            with open(config_path, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False, indent=2)
        except ImportError as err:
            raise ImportError("PyYAML is required to save YAML config files") from err
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def validate_field(obj: Any, field: str, expected_type: type = str) -> bool:
    """Validate that an object has a field of expected type."""
    if hasattr(obj, field):
        value = getattr(obj, field)
        return isinstance(value, expected_type)
    elif isinstance(obj, dict):
        value = obj.get(field)
        return isinstance(value, expected_type)
    return False


def extract_field(obj: Any, field: str, default: Any = "") -> Any:
    """Extract field from object, handling various formats."""
    if hasattr(obj, field):
        value = getattr(obj, field)
    elif isinstance(obj, dict):
        value = obj.get(field, default)
    else:
        # Try to convert the entire object to string
        value = str(obj) if obj is not None else default

    return value if value is not None else default


def normalize_text(
    text: str, lowercase: bool = True, strip_whitespace: bool = True, remove_extra_spaces: bool = True
) -> str:
    """Normalize text for consistent processing."""
    if not isinstance(text, str):
        text = str(text)

    if strip_whitespace:
        text = text.strip()

    if remove_extra_spaces:
        import re

        text = re.sub(r"\s+", " ", text)

    if lowercase:
        text = text.lower()

    return text


def ensure_list(item: Union[Any, List[Any]]) -> List[Any]:
    """Ensure item is a list."""
    if isinstance(item, list):
        return item
    else:
        return [item]


def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split items into batches of specified size."""
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def check_optional_dependency(package_name: str, feature_name: str = "") -> bool:
    """Check if optional dependency is available."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        feature_msg = f" for {feature_name}" if feature_name else ""
        warnings.warn(
            f"Optional dependency '{package_name}' not found{feature_msg}. Install with: pip install {package_name}",
            ImportWarning,
            stacklevel=2,
        )
        return False


def get_env_var(var_name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Get environment variable with optional default and validation."""
    value = os.environ.get(var_name, default)

    if required and value is None:
        raise ValueError(f"Required environment variable '{var_name}' not set")

    return value


class DSPyKitError(Exception):
    """Base exception for dspy-kit."""

    pass


class ConfigurationError(DSPyKitError):
    """Error in configuration."""

    pass


class ValidationError(DSPyKitError):
    """Error in validation."""

    pass


class DependencyError(DSPyKitError):
    """Error with dependencies."""

    pass


# Export all public utilities
__all__ = [
    # Logging
    "setup_logging",
    # Configuration
    "load_config",
    "save_config",
    # Data processing
    "validate_field",
    "extract_field",
    "normalize_text",
    "ensure_list",
    "batch_items",
    "safe_divide",
    # Dependencies and environment
    "check_optional_dependency",
    "get_env_var",
    # Exceptions
    "DSPyKitError",
    "ConfigurationError",
    "ValidationError",
    "DependencyError",
    # Version
    "__version__",
]
