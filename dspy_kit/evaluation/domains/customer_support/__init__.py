"""Customer support domain-specific evaluation graders."""

from .graders import (
    # Core customer support graders
    IntentAccuracyGrader,
    EscalationDetectionGrader,
    CustomerSatisfactionGrader,
    EmpathyEvaluationGrader,
    ProblemResolutionGrader,
    FirstContactResolutionGrader,
    ComplianceGrader,
    ResponseCompletenessGrader,
    UrgencyAssessmentGrader,
    KnowledgeBaseAccuracyGrader,
    
    # Composite graders
    CustomerSupportCompositeGrader,
    CustomerSupportRouterGrader,
    
    # Convenience functions
    create_intent_classifier_grader,
    create_basic_support_grader,
    create_advanced_support_grader,
    create_routing_agent_grader,
)

# Import commonly used core graders for convenience
from dspy_evals.core import (
    ToneEvaluationGrader,
    FactualAccuracyGrader,
    SafetyGrader,
    RelevanceGrader,
    ContextUtilizationGrader,
    ExactMatchGrader,
    FuzzyMatchGrader,
    create_customer_support_grader,
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