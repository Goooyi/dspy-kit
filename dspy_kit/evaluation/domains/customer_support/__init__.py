"""Customer support domain-specific evaluation graders."""

# Import commonly used core graders for convenience
from dspy_kit.evaluation.graders.dspy_model_graders import (
    ContextUtilizationGrader as DSPyContextUtilizationGrader,
    FactualAccuracyGrader as DSPyFactualAccuracyGrader,
    RelevanceGrader as DSPyRelevanceGrader,
    SafetyGrader as DSPySafetyGrader,
    ToneEvaluationGrader as DSPyToneEvaluationGrader,
    create_customer_support_grader as create_dspy_customer_support_grader,
)
from dspy_kit.evaluation.graders.string_graders import (
    ExactMatchGrader,
)
from dspy_kit.evaluation.graders.string_graders import (
    TextSimilarityGrader as FuzzyMatchGrader,
)

from .graders import (
    ComplianceGrader,
    CustomerSatisfactionGrader,
    # Composite graders
    CustomerSupportCompositeGrader,
    CustomerSupportRouterGrader,
    EmpathyEvaluationGrader,
    EscalationDetectionGrader,
    FirstContactResolutionGrader,
    # Core customer support graders
    IntentAccuracyGrader,
    KnowledgeBaseAccuracyGrader,
    ProblemResolutionGrader,
    ResponseCompletenessGrader,
    UrgencyAssessmentGrader,
    create_advanced_support_grader,
    create_basic_support_grader,
    # Convenience functions
    create_intent_classifier_grader,
    create_routing_agent_grader,
)

__all__ = [
    # Core customer support graders
    # Core customer support graders
    "IntentAccuracyGrader",
    "EscalationDetectionGrader",
    "CustomerSatisfactionGrader",
    "EmpathyEvaluationGrader",
    "ProblemResolutionGrader",
    "FirstContactResolutionGrader",
    "ComplianceGrader",
    "ResponseCompletenessGrader",
    "UrgencyAssessmentGrader",
    "KnowledgeBaseAccuracyGrader",
    # Composite graders
    "CustomerSupportCompositeGrader",
    "CustomerSupportRouterGrader",
    # Convenience functions
    "create_intent_classifier_grader",
    "create_basic_support_grader",
    "create_advanced_support_grader",
    "create_routing_agent_grader",
    # Re-exported core graders
    "ToneEvaluationGrader",
    "FactualAccuracyGrader",
    "SafetyGrader",
    "RelevanceGrader",
    "ContextUtilizationGrader",
    "ExactMatchGrader",
    "FuzzyMatchGrader",
    "create_customer_support_grader",
]

# Convenience aliases for common patterns
IntentClassifier = IntentAccuracyGrader
EscalationDetector = EscalationDetectionGrader
SatisfactionPredictor = CustomerSatisfactionGrader
EmpathyEvaluator = EmpathyEvaluationGrader
ProblemResolver = ProblemResolutionGrader
ComprehensiveSupport = CustomerSupportCompositeGrader
SupportRouter = CustomerSupportRouterGrader
