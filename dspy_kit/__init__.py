"""
DSPy Kit - Comprehensive Toolkit for DSPy Programs

A batteries-included toolkit for DSPy programs including:
- Comprehensive evaluation framework with graders and metrics
- Synthetic data generation (coming soon)
- Red teaming and adversarial testing (coming soon)

Built following OpenAI and Anthropic best practices for LLM evaluation and safety.

Features:
- String-based graders (exact match, fuzzy matching, regex, etc.)
- Model-based graders (LLM-as-a-judge with async support)  
- Python code graders (custom evaluation logic)
- Classification graders (precision, recall, F1, accuracy)
- Domain-specific graders (customer support, QA, etc.)
- Composite graders (weighted combinations)
- Edge case handling and configuration-driven evaluation
- Full DSPy integration and optimization support
"""

__version__ = "0.1.0"
__author__ = "Goooyi"
__email__ = "gaaoyi@gmail.com"

# Import all evaluation graders (main functionality)
from .evaluation import *

# Import utilities
from .utils import *

# Future modules available but not auto-imported to avoid warnings
# Users can import them explicitly: from dspy_kit import synthetic, red_team

# Main exports - re-export everything from evaluation module
from .evaluation.graders import __all__ as _grader_exports

__all__ = [
    # Re-export all graders and evaluation tools
    *_grader_exports,
    
    # Utility functions
    "setup_logging",
    "load_config", 
    "save_config",
    "extract_field",
    "normalize_text",
    "check_optional_dependency",
    
    # Version and metadata
    "get_version",
    "get_info",
]

# Convenience aliases for common patterns (maintain backward compatibility)
ExactMatch = ExactMatchGrader
FuzzyMatch = FuzzyMatchGrader  
Contains = ContainsGrader
LLMJudge = ScoreModelGrader
BinaryChoice = BinaryClassificationGrader
LikertScale = LikertScaleGrader
Precision = PrecisionGrader
Recall = RecallGrader
F1Score = F1Grader
Accuracy = AccuracyGrader
IntentClassifier = IntentClassificationGrader

# Add aliases to exports
__all__.extend([
    "ExactMatch",
    "FuzzyMatch", 
    "Contains",
    "LLMJudge",
    "BinaryChoice",
    "LikertScale",
    "Precision",
    "Recall", 
    "F1Score",
    "Accuracy",
    "IntentClassifier",
])

# Package metadata
__description__ = "Comprehensive toolkit for DSPy programs: evaluation, synthetic data, and red teaming"
__url__ = "https://github.com/Goooyi/dspy-kit" 
__license__ = "MIT"
__keywords__ = ["dspy", "evaluation", "synthetic-data", "red-team", "llm", "metrics", "ai", "nlp"]

# Version info
VERSION_INFO = tuple(map(int, __version__.split(".")))


def get_version():
    """Get the current version."""
    return __version__


def get_info():
    """Get package information."""
    return {
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": __description__,
        "url": __url__,
        "license": __license__,
        "modules": {
            "evaluation": "✅ Available - Comprehensive grader framework",
            "synthetic": "🚧 Coming soon - Synthetic data generation", 
            "red_team": "🚧 Coming soon - Red teaming and adversarial testing",
            "utils": "✅ Available - Shared utilities"
        }
    }


# Welcome message for interactive use
def _print_welcome():
    """Print welcome message with key info."""
    try:
        import sys
        if hasattr(sys, "ps1"):  # Interactive mode
            print(f"🛠️  DSPy Kit v{__version__} - Comprehensive DSPy Toolkit")
            print("📖 Documentation: https://github.com/Goooyi/dspy-kit#readme")
            print("🎯 Evaluation: Try `create_exact_match()` or `ScoreModelGrader()`")
            print("📊 Classification: Try `F1Grader()` or `create_intent_classifier_grader()`")
            print("🔮 Synthetic data & red teaming: Coming soon!")
    except Exception:
        pass  # Silently fail if there are any issues