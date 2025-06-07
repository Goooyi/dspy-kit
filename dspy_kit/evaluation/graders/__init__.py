"""Evaluation graders for DSPy programs following OpenAI and Anthropic best practices."""

# Base classes and utilities
from .base import (
    BaseGrader,
    CompositeGrader,
    ConfigurableGrader,
    EdgeCaseAwareGrader,
    dspy_metric,
)

# Model-based graders (LLM-as-a-judge)
from .model_graders import (
    ModelGrader,
    BinaryClassificationGrader,
    ContextUtilizationGrader,
    FactualAccuracyGrader,
    LabelModelGrader,
    LikertScaleGrader,
    RelevanceGrader,
    SafetyGrader,
    ScoreModelGrader,
    ToneEvaluationGrader,
    create_customer_support_grader,
    create_qa_grader,
)

# Python code graders
from .python_graders import (
    UTILITY_FUNCTIONS,
    CustomMetricGrader,
    FuzzyMatchGrader,
    JSONValidationGrader,
    ListComparisonGrader,
    NumericAccuracyGrader,
    PythonGrader,
    RegexMatchGrader,
    SQLExecutionGrader,
    create_lambda_grader,
    create_python_grader_from_file,
)

# String-based graders
from .string_graders import (
    ContainsGrader,
    ExactMatchGrader,
    MultiFieldGrader,
    RegexGrader,
    StartsWithGrader,
    StringCheckGrader,
    TextSimilarityGrader,
    create_contains_check,
    create_exact_match,
    create_fuzzy_match,
)

__all__ = [
    # Base classes
    "BaseGrader",
    "CompositeGrader",
    "EdgeCaseAwareGrader",
    "ConfigurableGrader",
    "dspy_metric",
    # String graders
    "StringCheckGrader",
    "TextSimilarityGrader",
    "ExactMatchGrader",
    "ContainsGrader",
    "StartsWithGrader",
    "RegexGrader",
    "MultiFieldGrader",
    "create_exact_match",
    "create_fuzzy_match",
    "create_contains_check",
    # Model graders
    "ModelGrader",
    "ScoreModelGrader",
    "LabelModelGrader",
    "LikertScaleGrader",
    "BinaryClassificationGrader",
    "FactualAccuracyGrader",
    "ToneEvaluationGrader",
    "ContextUtilizationGrader",
    "SafetyGrader",
    "RelevanceGrader",
    "create_customer_support_grader",
    "create_qa_grader",
    # Python graders
    "PythonGrader",
    "FuzzyMatchGrader",
    "RegexMatchGrader",
    "JSONValidationGrader",
    "NumericAccuracyGrader",
    "ListComparisonGrader",
    "SQLExecutionGrader",
    "CustomMetricGrader",
    "create_python_grader_from_file",
    "create_lambda_grader",
    "UTILITY_FUNCTIONS",
]

# Convenience aliases for common patterns
ExactMatch = ExactMatchGrader
FuzzyMatch = FuzzyMatchGrader
Contains = ContainsGrader
LLMJudge = ScoreModelGrader
BinaryChoice = BinaryClassificationGrader
