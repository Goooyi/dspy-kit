"""Customer support domain-specific graders."""

from typing import Any, Optional, Union

from dspy_kit.evaluation.graders.base import CompositeGrader, ConfigurableGrader
from dspy_kit.evaluation.graders.model_graders import (
    BinaryClassificationGrader,
    LabelModelGrader,
    LikertScaleGrader,
    ScoreModelGrader,
)
from dspy_kit.evaluation.graders.string_graders import ExactMatchGrader


class IntentAccuracyGrader(ConfigurableGrader):
    """
    Evaluates intent classification accuracy for customer support routing.
    Critical for multi-agent customer support systems.
    """

    DEFAULT_CONFIG = {
        "pred": "predicted_intent",
        "ideal": "true_intent",
        "valid_intents": [
            "billing", "technical_support", "account_management",
            "product_inquiry", "complaint", "cancellation", "other"
        ],
        "strict_matching": True,
        "case_sensitive": False
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pred_field = getattr(self, 'pred', self.DEFAULT_CONFIG['pred'])
        ideal_field = getattr(self, 'ideal', self.DEFAULT_CONFIG['ideal'])
        case_sensitive = getattr(self, 'case_sensitive', self.DEFAULT_CONFIG['case_sensitive'])

        self.exact_match_grader = ExactMatchGrader(
            pred=pred_field,
            ideal=ideal_field,
            case_sensitive=case_sensitive
        )

    def __call__(self, example: Any, pred: Any, trace: Optional[Any] = None) -> Union[float, bool]:
        # Extract predicted and true intents
        pred_field = getattr(self, 'pred', self.DEFAULT_CONFIG['pred'])
        ideal_field = getattr(self, 'ideal', self.DEFAULT_CONFIG['ideal'])
        predicted_intent = self.extract_field(pred, pred_field)
        true_intent = self.extract_field(example, ideal_field)

        # Validate intents are in valid set
        strict_matching = getattr(self, 'strict_matching', self.DEFAULT_CONFIG['strict_matching'])
        valid_intents = getattr(self, 'valid_intents', self.DEFAULT_CONFIG['valid_intents'])
        if strict_matching:
            if predicted_intent not in valid_intents:
                return 0.0 if trace is None else False
            if true_intent not in valid_intents:
                return 0.0 if trace is None else False

        # Use exact match grader
        return self.exact_match_grader(example, pred, trace)


class EscalationDetectionGrader(BinaryClassificationGrader):
    """
    Detects when a customer query requires human escalation.
    """

    DEFAULT_CONFIG = {
        **BinaryClassificationGrader.DEFAULT_CONFIG,
        "question": "Should this customer query be escalated to a human agent?",
        "prompt_template": """Determine if this customer query requires human escalation.

Escalate if:
- Customer is extremely frustrated or angry
- Complex technical issue beyond AI capabilities
- Legal or compliance concerns
- Account security issues
- Multiple failed resolution attempts

Customer Query: {{item.question}}
AI Response: {{sample.output_text}}
Customer History: {{item.interaction_history}}

Should this be escalated to a human? (yes/no):""",
        "labels": ["yes", "no"],
        "passing_labels": ["yes"]
    }


class CustomerSatisfactionGrader(ScoreModelGrader):
    """
    Predicts customer satisfaction based on the interaction.
    """

    DEFAULT_CONFIG = {
        **ScoreModelGrader.DEFAULT_CONFIG,
        "range": [1, 5],
        "pass_threshold": 4.0,
        "prompt_template": """Rate the likely customer satisfaction with this support interaction (1-5):

1 = Very Dissatisfied (angry, unresolved, poor experience)
2 = Dissatisfied (frustrated, partially resolved)
3 = Neutral (basic resolution, no strong feelings)
4 = Satisfied (good resolution, positive experience)
5 = Very Satisfied (excellent service, exceeded expectations)

Customer Query: {{item.question}}
Support Response: {{sample.output_text}}
Resolution Status: {{item.resolution_status}}

Predicted satisfaction score (1-5):""",
        "system_prompt": "You are a customer experience expert predicting satisfaction scores."
    }


class EmpathyEvaluationGrader(LikertScaleGrader):
    """
    Evaluates the empathy and emotional intelligence in customer support responses.
    """

    DEFAULT_CONFIG = {
        **LikertScaleGrader.DEFAULT_CONFIG,
        "range": [1, 5],
        "pass_threshold": 3.0,
        "criteria": "Empathy, emotional understanding, and human connection",
        "prompt_template": """Rate the empathy shown in this customer support response (1-5):

1 = No empathy, robotic, dismissive of customer feelings
2 = Low empathy, acknowledges issue but lacks emotional connection
3 = Moderate empathy, shows understanding but somewhat generic
4 = Good empathy, demonstrates care and understanding
5 = Excellent empathy, deeply understanding and emotionally supportive

Customer Query: {{item.question}}
Customer Sentiment: {{item.customer_sentiment}}
Support Response: {{sample.output_text}}

Empathy rating (1-5):""",
        "system_prompt": "You are an emotional intelligence expert evaluating empathy in customer service."
    }


class ProblemResolutionGrader(ScoreModelGrader):
    """
    Evaluates how well the response addresses and resolves the customer's problem.
    """

    DEFAULT_CONFIG = {
        **ScoreModelGrader.DEFAULT_CONFIG,
        "range": [1, 5],
        "pass_threshold": 4.0,
        "prompt_template": """Rate how well this response resolves the customer's problem (1-5):

1 = Doesn't address the problem at all
2 = Partially addresses but doesn't provide solution
3 = Addresses the problem with basic solution
4 = Good resolution with clear next steps
5 = Excellent resolution, comprehensive and actionable

Customer Problem: {{item.question}}
Problem Category: {{item.problem_category}}
Support Response: {{sample.output_text}}
Available Solutions: {{item.available_solutions}}

Resolution quality (1-5):""",
        "system_prompt": "You are a customer support quality expert evaluating problem resolution."
    }


class FirstContactResolutionGrader(BinaryClassificationGrader):
    """
    Evaluates if the issue can be resolved in the first contact.
    """

    DEFAULT_CONFIG = {
        **BinaryClassificationGrader.DEFAULT_CONFIG,
        "question": "Can this customer issue be resolved in the first contact?",
        "prompt_template": """Determine if this customer issue can be fully resolved in the first contact.

Consider:
- Complexity of the issue
- Information provided by customer
- Available solutions and tools
- Need for additional verification or escalation

Customer Issue: {{item.question}}
Support Response: {{sample.output_text}}
Customer Account Status: {{item.account_status}}

Can be resolved in first contact? (yes/no):""",
        "labels": ["yes", "no"],
        "passing_labels": ["yes"]
    }


class ComplianceGrader(BinaryClassificationGrader):
    """
    Checks if the response complies with company policies and regulations.
    """

    DEFAULT_CONFIG = {
        **BinaryClassificationGrader.DEFAULT_CONFIG,
        "question": "Does this response comply with company policies?",
        "prompt_template": """Check if this support response complies with policies:

Policy Areas to Check:
- Privacy and data protection
- Terms of service adherence
- Refund and cancellation policies
- Regulatory compliance
- Appropriate language and tone

Customer Query: {{item.question}}
Support Response: {{sample.output_text}}
Company Policies: {{item.company_policies}}

Is compliant? (yes/no):""",
        "labels": ["yes", "no"],
        "passing_labels": ["yes"]
    }


class ResponseCompletenessGrader(ScoreModelGrader):
    """
    Evaluates if the response completely addresses all aspects of the customer query.
    """

    DEFAULT_CONFIG = {
        **ScoreModelGrader.DEFAULT_CONFIG,
        "range": [1, 5],
        "pass_threshold": 4.0,
        "prompt_template": """Rate how completely this response addresses the customer query (1-5):

1 = Addresses none of the questions/concerns
2 = Addresses some but misses major points
3 = Addresses most points but lacks some details
4 = Addresses all main points with good detail
5 = Comprehensive response covering everything thoroughly

Customer Query: {{item.question}}
Key Points to Address: {{item.key_points}}
Support Response: {{sample.output_text}}

Completeness score (1-5):""",
        "system_prompt": "You are a quality assurance expert evaluating response completeness."
    }


class UrgencyAssessmentGrader(LabelModelGrader):
    """
    Assesses the urgency level of customer issues for proper prioritization.
    """

    DEFAULT_CONFIG = {
        **LabelModelGrader.DEFAULT_CONFIG,
        "labels": ["low", "medium", "high", "critical"],
        "passing_labels": ["high", "critical"],
        "prompt_template": """Classify the urgency level of this customer issue:

- Critical: Service down, security breach, data loss
- High: Major functionality broken, billing errors, angry customer
- Medium: Minor bugs, feature requests, general questions
- Low: Information requests, minor cosmetic issues

Customer Issue: {{item.question}}
Customer Type: {{item.customer_tier}}
Business Impact: {{item.business_impact}}

Urgency level (low/medium/high/critical):""",
        "system_prompt": "You are a support triage expert classifying issue urgency."
    }


class KnowledgeBaseAccuracyGrader(ScoreModelGrader):
    """
    Evaluates if the response accurately uses knowledge base information.
    """

    DEFAULT_CONFIG = {
        **ScoreModelGrader.DEFAULT_CONFIG,
        "range": [1, 5],
        "pass_threshold": 4.0,
        "prompt_template": """Rate how accurately this response uses knowledge base information (1-5):

1 = Contradicts knowledge base or provides wrong information
2 = Partially correct but has significant inaccuracies
3 = Mostly correct with minor inaccuracies
4 = Accurate and well-aligned with knowledge base
5 = Perfectly accurate and expertly uses knowledge base

Customer Query: {{item.question}}
Relevant KB Articles: {{item.kb_articles}}
Support Response: {{sample.output_text}}

Accuracy score (1-5):""",
        "system_prompt": "You are a knowledge management expert evaluating information accuracy."
    }


class CustomerSupportCompositeGrader(CompositeGrader):
    """
    Comprehensive customer support grader combining multiple dimensions.
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        include_empathy: bool = True,
        include_escalation: bool = True,
        **kwargs
    ):
        # Default weights optimized for customer support
        default_weights = {
            "problem_resolution": 0.25,
            "response_completeness": 0.20,
            "tone_evaluation": 0.15,
            "factual_accuracy": 0.15,
            "safety": 0.10,
            "compliance": 0.10,
        }

        if include_empathy:
            default_weights["empathy"] = 0.05

        if include_escalation:
            # Adjust weights to accommodate escalation
            for key in default_weights:
                default_weights[key] *= 0.95
            default_weights["escalation_detection"] = 0.05

        # Use provided weights or defaults
        final_weights = weights or default_weights

        # Initialize graders
        graders = {
            "problem_resolution": (ProblemResolutionGrader(), final_weights.get("problem_resolution", 0.25)),
            "response_completeness": (ResponseCompletenessGrader(), final_weights.get("response_completeness", 0.20)),
            "factual_accuracy": (FactualAccuracyGrader(), final_weights.get("factual_accuracy", 0.15)),
            "compliance": (ComplianceGrader(), final_weights.get("compliance", 0.10)),
        }

        if include_empathy:
            graders["empathy"] = (EmpathyEvaluationGrader(), final_weights.get("empathy", 0.05))

        if include_escalation:
            graders["escalation_detection"] = (EscalationDetectionGrader(), final_weights.get("escalation_detection", 0.05))

        super().__init__(graders, **kwargs)


class CustomerSupportRouterGrader(CompositeGrader):
    """
    Specialized grader for customer support routing/intent classification systems.
    """

    def __init__(self, valid_intents: Optional[list[str]] = None, **kwargs):
        graders = {
            "intent_accuracy": (
                IntentAccuracyGrader(valid_intents=valid_intents or []),
                0.6
            ),
            "urgency_assessment": (UrgencyAssessmentGrader(), 0.25),
            "escalation_detection": (EscalationDetectionGrader(), 0.15),
        }

        super().__init__(graders, **kwargs)


# Convenience functions for common customer support scenarios
def create_intent_classifier_grader(valid_intents: list[str]) -> IntentAccuracyGrader:
    """Create an intent classification grader with specific valid intents."""
    return IntentAccuracyGrader(valid_intents=valid_intents)


def create_basic_support_grader() -> CustomerSupportCompositeGrader:
    """Create a basic customer support grader without advanced features."""
    return CustomerSupportCompositeGrader(
        include_empathy=False,
        include_escalation=False
    )


def create_advanced_support_grader() -> CustomerSupportCompositeGrader:
    """Create a comprehensive customer support grader with all features."""
    return CustomerSupportCompositeGrader(
        include_empathy=True,
        include_escalation=True
    )


def create_routing_agent_grader(valid_intents: list[str]) -> CustomerSupportRouterGrader:
    """Create a grader specifically for customer support routing agents."""
    return CustomerSupportRouterGrader(valid_intents=valid_intents)