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

# Explicit imports from evaluation module
# Import domain-specific graders (only customer support is implemented)
from .evaluation.domains.customer_support import (
    CustomerSatisfactionGrader,
    CustomerSupportCompositeGrader,
    EscalationDetectionGrader,
    IntentAccuracyGrader,
    create_advanced_support_grader,
    create_basic_support_grader,
    create_intent_classifier_grader,
)
from .evaluation.graders import (
    UTILITY_FUNCTIONS,
    BaseDSPyGrader,
    BaseGrader,
    CompositeDSPyGrader,
    CompositeGrader,
    ConfigurableGrader,
    ContainsGrader,
    CustomMetricGrader,
    DSPyContextUtilizationGrader,
    DSPyFactualAccuracyGrader,
    DSPyLikertScaleGrader,
    DSPyRelevanceGrader,
    DSPySafetyGrader,
    DSPyToneEvaluationGrader,
    EdgeCaseAwareGrader,
    ExactMatchGrader,
    FuzzyMatchGrader,
    HelpfulnessGrader,
    JSONValidationGrader,
    ListComparisonGrader,
    MultiFieldGrader,
    NumericAccuracyGrader,
    PythonGrader,
    RegexGrader,
    RegexMatchGrader,
    SemanticSimilarityGrader,
    SQLExecutionGrader,
    StartsWithGrader,
    StringCheckGrader,
    TextSimilarityGrader,
    create_advanced_customer_support_grader,
    create_comprehensive_qa_grader,
    create_contains_check,
    create_dspy_customer_support_grader,
    create_dspy_qa_grader,
    create_exact_match,
    create_fuzzy_match,
    create_lambda_grader,
    create_python_grader_from_file,
    dspy_metric,
)

# Explicit imports from utils module
from .utils import (
    ConfigurationError,
    DependencyError,
    DSPyKitError,
    ValidationError,
    batch_items,
    check_optional_dependency,
    ensure_list,
    extract_field,
    get_env_var,
    load_config,
    normalize_text,
    safe_divide,
    save_config,
    setup_logging,
    validate_field,
)

# Convenience aliases for common patterns (maintain backward compatibility)
ExactMatch = ExactMatchGrader
FuzzyMatch = FuzzyMatchGrader
Contains = ContainsGrader

# Define explicit __all__ list
__all__ = [
    # Base classes
    "BaseGrader",
    "CompositeGrader",
    "ConfigurableGrader",
    "EdgeCaseAwareGrader",
    "dspy_metric",
    # String graders
    "ContainsGrader",
    "ExactMatchGrader",
    "MultiFieldGrader",
    "RegexGrader",
    "StartsWithGrader",
    "StringCheckGrader",
    "TextSimilarityGrader",
    "create_contains_check",
    "create_exact_match",
    "create_fuzzy_match",
    # DSPy-optimizable graders
    "BaseDSPyGrader",
    "CompositeDSPyGrader",
    "DSPyContextUtilizationGrader",
    "DSPyFactualAccuracyGrader",
    "DSPyLikertScaleGrader",
    "DSPyRelevanceGrader",
    "DSPySafetyGrader",
    "DSPyToneEvaluationGrader",
    "HelpfulnessGrader",
    "SemanticSimilarityGrader",
    "create_advanced_customer_support_grader",
    "create_comprehensive_qa_grader",
    "create_dspy_customer_support_grader",
    "create_dspy_qa_grader",
    # Python graders
    "CustomMetricGrader",
    "FuzzyMatchGrader",
    "JSONValidationGrader",
    "ListComparisonGrader",
    "NumericAccuracyGrader",
    "PythonGrader",
    "RegexMatchGrader",
    "SQLExecutionGrader",
    "UTILITY_FUNCTIONS",
    "create_lambda_grader",
    "create_python_grader_from_file",
    # Domain-specific graders (customer support)
    "CustomerSatisfactionGrader",
    "CustomerSupportCompositeGrader",
    "EscalationDetectionGrader",
    "IntentAccuracyGrader",
    "create_advanced_support_grader",
    "create_basic_support_grader",
    "create_intent_classifier_grader",
    # Utility functions
    "batch_items",
    "check_optional_dependency",
    "ensure_list",
    "extract_field",
    "get_env_var",
    "load_config",
    "normalize_text",
    "safe_divide",
    "save_config",
    "setup_logging",
    "validate_field",
    # Exceptions
    "ConfigurationError",
    "DSPyKitError",
    "DependencyError",
    "ValidationError",
    # Convenience aliases
    "Contains",
    "ExactMatch",
    "FuzzyMatch",
    # Version and metadata
    "get_info",
    "get_version",
]

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
            "utils": "✅ Available - Shared utilities",
        },
    }


# Welcome message for interactive use
def _print_welcome():
    """Print welcome message with key info."""
    try:
        import sys

        if hasattr(sys, "ps1"):  # Interactive mode
            print(f"🛠️  DSPy Kit v{__version__} - Comprehensive DSPy Toolkit")
            print("📖 Documentation: https://github.com/Goooyi/dspy-kit#readme")
            print("🎯 Evaluation: Try `create_exact_match()` or `SemanticSimilarityGrader()`")
            print("📊 Classification: Try `FuzzyMatchGrader()` or `create_dspy_customer_support_grader()`")
            print("🔮 Synthetic data & red teaming: Coming soon!")
    except Exception:
        pass  # Silently fail if there are any issues
