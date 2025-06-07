"""Evaluation graders for DSPy programs following OpenAI and Anthropic best practices."""

# Base classes and utilities
from .base import (
    BaseGrader,
    CompositeGrader,
    ConfigurableGrader,
    EdgeCaseAwareGrader,
    dspy_metric,
)

# Legacy model-based graders (deprecated - use DSPy-optimizable versions instead)
# from .model_graders import (...)  # Removed - migrated to dspy_model_graders
# DSPy-optimizable graders with flexible field extraction
from .dspy_model_graders import (
    BaseDSPyGrader,
    CompositeDSPyGrader,
    HelpfulnessGrader,
    SemanticSimilarityGrader,
    create_advanced_customer_support_grader,
    create_comprehensive_qa_grader,
)
from .dspy_model_graders import (
    ContextUtilizationGrader as DSPyContextUtilizationGrader,
)
from .dspy_model_graders import (
    FactualAccuracyGrader as DSPyFactualAccuracyGrader,
)
from .dspy_model_graders import (
    LikertScaleGrader as DSPyLikertScaleGrader,
)
from .dspy_model_graders import (
    RelevanceGrader as DSPyRelevanceGrader,
)
from .dspy_model_graders import (
    SafetyGrader as DSPySafetyGrader,
)
from .dspy_model_graders import (
    ToneEvaluationGrader as DSPyToneEvaluationGrader,
)
from .dspy_model_graders import (
    create_customer_support_grader as create_dspy_customer_support_grader,
)
from .dspy_model_graders import (
    create_qa_grader as create_dspy_qa_grader,
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
    # Legacy model graders (deprecated - use DSPy versions)
    # Use DSPyFactualAccuracyGrader, DSPyRelevanceGrader, etc. instead
    # DSPy-optimizable graders
    "BaseDSPyGrader",
    "SemanticSimilarityGrader",
    "DSPyFactualAccuracyGrader",
    "DSPyRelevanceGrader",
    "HelpfulnessGrader",
    "DSPySafetyGrader",
    "DSPyToneEvaluationGrader",
    "DSPyContextUtilizationGrader",
    "DSPyLikertScaleGrader",
    "CompositeDSPyGrader",
    "create_dspy_qa_grader",
    "create_dspy_customer_support_grader",
    "create_advanced_customer_support_grader",
    "create_comprehensive_qa_grader",
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

# Legacy aliases (deprecated - use DSPy versions)
# LLMJudge = ScoreModelGrader  # Use DSPy graders instead
# BinaryChoice = BinaryClassificationGrader  # Use DSPy graders instead

# Recommended aliases for new DSPy-optimizable graders
FlexibleSemanticSimilarity = SemanticSimilarityGrader
FlexibleFactualAccuracy = DSPyFactualAccuracyGrader
FlexibleRelevance = DSPyRelevanceGrader
FlexibleHelpfulness = HelpfulnessGrader
FlexibleSafety = DSPySafetyGrader
FlexibleTone = DSPyToneEvaluationGrader
FlexibleContextUtilization = DSPyContextUtilizationGrader
FlexibleLikertScale = DSPyLikertScaleGrader
FlexibleComposite = CompositeDSPyGrader
