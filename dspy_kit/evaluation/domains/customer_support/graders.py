"""Customer support domain-specific graders."""

import re
from typing import Any, Optional, Union

import dspy

from dspy_kit.evaluation.graders.base import CompositeGrader, ConfigurableGrader
from dspy_kit.evaluation.graders.dspy_model_graders import (
    BaseDSPyGrader,
)
from dspy_kit.evaluation.graders.dspy_model_graders import (
    FactualAccuracyGrader as DSPyFactualAccuracyGrader,
)
from dspy_kit.evaluation.graders.dspy_model_graders import (
    HelpfulnessGrader as DSPyHelpfulnessGrader,
)
from dspy_kit.evaluation.graders.dspy_model_graders import (
    SafetyGrader as DSPySafetyGrader,
)
from dspy_kit.evaluation.graders.dspy_model_graders import (
    ToneEvaluationGrader as DSPyToneEvaluationGrader,
)
from dspy_kit.evaluation.graders.string_graders import ExactMatchGrader


class IntentAccuracyGrader(ConfigurableGrader):
    """
    Evaluates intent classification accuracy for customer support.

    Checks if the AI correctly identified the customer's intent
    (billing, technical, cancellation, etc.)
    """

    DEFAULT_CONFIG = {
        "valid_intents": ["billing", "technical", "account", "cancellation", "general"],
        "pred": "predicted_intent",
        "ideal": "actual_intent",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exact_match_grader = ExactMatchGrader()

    def __call__(self, example: Any, pred: Any, trace: Optional[Any] = None) -> Union[float, bool]:
        try:
            predicted_intent = self.extract_field(pred, getattr(self, "pred", "predicted_intent"))
            actual_intent = self.extract_field(example, getattr(self, "ideal", "actual_intent"))

            # Normalize intents to lowercase
            predicted_intent = predicted_intent.lower().strip()
            actual_intent = actual_intent.lower().strip()

            # Check if predicted intent is valid
            valid_intents = getattr(self, "valid_intents", self.DEFAULT_CONFIG["valid_intents"])
            if predicted_intent not in [intent.lower() for intent in valid_intents]:
                return 0.0 if trace is None else False

            # Use exact match grader
            return self.exact_match_grader(example, pred, trace)

        except Exception as e:
            print(f"IntentAccuracyGrader error: {e}")
            return 0.0 if trace is None else False


class EscalationDetectionGrader(BaseDSPyGrader):
    """
    Detects when a customer query requires human escalation.
    """

    class EscalationSignature(dspy.Signature):
        """
        Determine if a customer query requires human escalation.
        Consider frustration level, complexity, and resolution attempts.
        """

        customer_query: str = dspy.InputField()
        ai_response: str = dspy.InputField()
        escalation_needed: str = dspy.OutputField(desc="'yes' if escalation needed, 'no' if AI can handle")
        reasoning: str = dspy.OutputField(desc="Brief explanation of decision")

    def __init__(self, pred_field: str = "output", query_field: str = "question", **kwargs):
        super().__init__(pred_field, query_field, 0.5, **kwargs)
        self.query_field = query_field
        self.escalation_evaluator = dspy.ChainOfThought(self.EscalationSignature)

    def __call__(self, example, pred, trace=None):
        try:
            ai_response = self.extract_field(pred, self.pred_field)
            customer_query = self.extract_field(example, self.query_field)

            result = self.escalation_evaluator(customer_query=customer_query, ai_response=ai_response)

            needs_escalation = "yes" in result.escalation_needed.lower()
            score = 1.0 if needs_escalation else 0.0

            return score if trace is None else needs_escalation
        except Exception as e:
            print(f"EscalationDetectionGrader error: {e}")
            return 0.0 if trace is None else False


class CustomerSatisfactionGrader(BaseDSPyGrader):
    """
    Predicts customer satisfaction based on the interaction.
    """

    class SatisfactionSignature(dspy.Signature):
        """
        Predict customer satisfaction based on the support interaction.
        Consider resolution quality, response time, and tone.
        """

        customer_query: str = dspy.InputField()
        agent_response: str = dspy.InputField()
        satisfaction_score: float = dspy.OutputField(
            desc="Satisfaction score from 0.0 to 1.0, where 1.0 is very satisfied"
        )
        analysis: str = dspy.OutputField(desc="Analysis of satisfaction factors")

    def __init__(self, pred_field: str = "output", query_field: str = "question", **kwargs):
        super().__init__(pred_field, query_field, 0.7, **kwargs)
        self.query_field = query_field
        self.satisfaction_evaluator = dspy.ChainOfThought(self.SatisfactionSignature)

    def __call__(self, example, pred, trace=None):
        try:
            agent_response = self.extract_field(pred, self.pred_field)
            customer_query = self.extract_field(example, self.query_field)

            result = self.satisfaction_evaluator(customer_query=customer_query, agent_response=agent_response)

            score = self._parse_satisfaction_score(result.satisfaction_score)

            return score if trace is None else score >= self.threshold
        except Exception as e:
            print(f"CustomerSatisfactionGrader error: {e}")
            return 0.0 if trace is None else False

    def _parse_satisfaction_score(self, score_text: str) -> float:
        """Parse satisfaction score from text output."""
        try:
            numbers = re.findall(r"\d*\.?\d+", str(score_text))
            if numbers:
                score = float(numbers[0])
                if score > 1.0:
                    score = score / 5.0 if score <= 5.0 else 1.0
                return max(0.0, min(1.0, score))
            return 0.0
        except: # noqa "No rating given"
            return 0.0


class CustomerSupportCompositeGrader(CompositeGrader):
    """
    Comprehensive grader for customer support interactions.
    Combines multiple evaluation criteria with appropriate weights.
    """

    def __init__(
        self, include_empathy: bool = True, include_escalation: bool = True, include_satisfaction: bool = True, **kwargs
    ):
        graders = {}

        # Core graders (always included)
        graders["helpfulness"] = (DSPyHelpfulnessGrader(), 0.25)
        graders["accuracy"] = (DSPyFactualAccuracyGrader(), 0.25)
        graders["tone"] = (DSPyToneEvaluationGrader(), 0.2)
        graders["safety"] = (DSPySafetyGrader(), 0.1)

        # Optional graders
        remaining_weight = 0.2
        optional_count = sum([include_escalation, include_satisfaction])

        if optional_count > 0:
            weight_per_optional = remaining_weight / optional_count

            if include_escalation:
                graders["escalation"] = (EscalationDetectionGrader(), weight_per_optional)

            if include_satisfaction:
                graders["satisfaction"] = (CustomerSatisfactionGrader(), weight_per_optional)

        super().__init__(graders, **kwargs)


# Convenience functions
def create_intent_classifier_grader(valid_intents=None):
    """Create an intent classification grader."""
    config = {}
    if valid_intents:
        config["valid_intents"] = valid_intents
    return IntentAccuracyGrader(**config)


def create_basic_support_grader():
    """Create a basic customer support grader."""
    return CustomerSupportCompositeGrader(include_empathy=False, include_escalation=False)


def create_advanced_support_grader():
    """Create an advanced customer support grader with all features."""
    return CustomerSupportCompositeGrader(include_empathy=True, include_escalation=True)
