"""Customer support domain-specific evaluation graders."""

# Import commonly used core graders for convenience
# from dspy_kit.evaluation.graders.dspy_model_graders import (
#     ContextUtilizationGrader as DSPyContextUtilizationGrader,
# )
from dspy_kit.evaluation.graders.dspy_model_graders import (
    FactualAccuracyGrader as DSPyFactualAccuracyGrader,
)
from dspy_kit.evaluation.graders.dspy_model_graders import (
    RelevanceGrader as DSPyRelevanceGrader,
)
from dspy_kit.evaluation.graders.dspy_model_graders import (
    SafetyGrader as DSPySafetyGrader,
)
from dspy_kit.evaluation.graders.dspy_model_graders import (
    ToneEvaluationGrader as DSPyToneEvaluationGrader,
)

# from dspy_kit.evaluation.graders.dspy_model_graders import (
#     create_customer_support_grader as create_dspy_customer_support_grader,
# )
from dspy_kit.evaluation.graders.string_graders import (
    ExactMatchGrader,
)
from dspy_kit.evaluation.graders.string_graders import (
    TextSimilarityGrader as FuzzyMatchGrader,
)

from .graders import (
    # ComplianceGrader,  # Not implemented yet
    CustomerSatisfactionGrader,
    # Composite graders
    CustomerSupportCompositeGrader,
    # CustomerSupportRouterGrader,  # Not implemented yet
    # EmpathyEvaluationGrader,  # Not implemented yet
    EscalationDetectionGrader,
    # FirstContactResolutionGrader,  # Not implemented yet
    # Core customer support graders
    IntentAccuracyGrader,
    # KnowledgeBaseAccuracyGrader,  # Not implemented yet
    # ProblemResolutionGrader,  # Not implemented yet
    # ResponseCompletenessGrader,  # Not implemented yet
    # UrgencyAssessmentGrader,  # Not implemented yet
    create_advanced_support_grader,
    create_basic_support_grader,
    # Convenience functions
    create_intent_classifier_grader,
    # create_routing_agent_grader,  # Not implemented yet
)

__all__ = [
    # Core customer support graders (implemented)
    "IntentAccuracyGrader",
    "EscalationDetectionGrader",
    "CustomerSatisfactionGrader",
    # "EmpathyEvaluationGrader",  # Not implemented yet
    # "ProblemResolutionGrader",  # Not implemented yet
    # "FirstContactResolutionGrader",  # Not implemented yet
    # "ComplianceGrader",  # Not implemented yet
    # "ResponseCompletenessGrader",  # Not implemented yet
    # "UrgencyAssessmentGrader",  # Not implemented yet
    # "KnowledgeBaseAccuracyGrader",  # Not implemented yet
    # Composite graders (implemented)
    "CustomerSupportCompositeGrader",
    # "CustomerSupportRouterGrader",  # Not implemented yet
    # Convenience functions (implemented)
    "create_intent_classifier_grader",
    "create_basic_support_grader",
    "create_advanced_support_grader",
    # "create_routing_agent_grader",  # Not implemented yet
    # Re-exported core graders
    "DSPyToneEvaluationGrader",
    "DSPyFactualAccuracyGrader",
    "DSPySafetyGrader",
    "DSPyRelevanceGrader",
    # "DSPyContextUtilizationGrader",  # Not implemented yet
    "ExactMatchGrader",
    "FuzzyMatchGrader",
    # "create_dspy_customer_support_grader",  # Not implemented yet
]

# Convenience aliases for common patterns (only for implemented graders)
IntentClassifier = IntentAccuracyGrader
EscalationDetector = EscalationDetectionGrader
SatisfactionPredictor = CustomerSatisfactionGrader
# EmpathyEvaluator = EmpathyEvaluationGrader  # Not implemented yet
# ProblemResolver = ProblemResolutionGrader  # Not implemented yet
ComprehensiveSupport = CustomerSupportCompositeGrader
# SupportRouter = CustomerSupportRouterGrader  # Not implemented yet
