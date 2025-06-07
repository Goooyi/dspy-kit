"""
DSPy-native model graders that combine optimization capabilities with flexible field extraction.

These graders are dspy.Module instances that can be optimized while providing
flexible field access patterns, solving the limitations of both traditional
DSPy metrics (hardcoded fields) and current dspy-kit graders (not optimizable).
"""

import re
from typing import Any, Dict, Optional, Union

import dspy


class BaseDSPyGrader(dspy.Module):
    """
    Base class for DSPy-optimizable graders with flexible field extraction.

    Unlike traditional DSPy metrics that hardcode field access (e.g., example.question),
    these graders use configurable field extraction that works with any naming scheme.
    """

    def __init__(
        self,
        pred_field: str = "output",
        ideal_field: str = "expected",
        threshold: float = 0.7,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.pred_field = pred_field
        self.ideal_field = ideal_field
        self.threshold = threshold
        self.name = name or self.__class__.__name__

    def extract_field(self, obj: Any, field: str, default: str = "") -> str:
        """
        Extract field from object, handling various formats gracefully.

        Args:
            obj: Object to extract from (DSPy output, dict, etc.)
            field: Field name to extract
            default: Default value if field not found

        Returns:
            Extracted value as string
        """
        if hasattr(obj, field):
            value = getattr(obj, field)
        elif isinstance(obj, dict):
            value = obj.get(field, default)
        else:
            # Try to convert the entire object to string
            value = str(obj) if obj is not None else default

        return str(value) if value is not None else default

    def forward(self, example: Any, pred: Any, trace: Optional[Any] = None) -> Union[float, bool]:
        """
        DSPy Module forward method - delegates to __call__ for compatibility.
        """
        return self(example, pred, trace)

    def __call__(self, example: Any, pred: Any, trace: Optional[Any] = None) -> Union[float, bool]:
        """
        Evaluate a prediction against an example.

        Args:
            example: The ground truth example from your dataset
            pred: The prediction from your DSPy program
            trace: Optional trace from DSPy optimization

        Returns:
            Score as float (0.0-1.0) for evaluation,
            or bool for optimization (score >= threshold)
        """
        raise NotImplementedError("Subclasses must implement __call__ method")

    def to_dspy_metric(self):
        """Convert to DSPy-compatible metric function."""

        def metric(example, pred, trace=None):
            return self(example, pred, trace)

        metric.__name__ = f"{self.name}_metric"
        return metric


class SemanticSimilarityGrader(BaseDSPyGrader):
    """
    DSPy-optimizable semantic similarity grader using flexible field extraction.

    Example:
        grader = SemanticSimilarityGrader(
            pred_field="generated_answer",  # Your field name
            ideal_field="reference_answer", # Your field name
            threshold=0.8
        )
    """

    def __init__(self, pred_field: str = "output", ideal_field: str = "expected", threshold: float = 0.8, **kwargs):
        super().__init__(pred_field, ideal_field, threshold, **kwargs)
        self.similarity_evaluator = dspy.ChainOfThought("predicted_text, reference_text -> similarity_score")  # type: ignore

    def __call__(self, example: Any, pred: Any, trace: Optional[Any] = None) -> Union[float, bool]:
        try:
            predicted_text = self.extract_field(pred, self.pred_field)
            reference_text = self.extract_field(example, self.ideal_field)

            if not predicted_text.strip() or not reference_text.strip():
                return 0.0 if trace is None else False

            result = self.similarity_evaluator(predicted_text=predicted_text, reference_text=reference_text)

            # Parse similarity score from result
            score = self._parse_similarity_score(result.similarity_score)

            return score if trace is None else score >= self.threshold

        except Exception as e:
            print(f"SemanticSimilarityGrader error: {e}")
            return 0.0 if trace is None else False

    def _parse_similarity_score(self, score_text: str) -> float:
        """Parse similarity score from text output."""
        try:
            # Extract number from text (handle various formats)
            numbers = re.findall(r"\d*\.?\d+", str(score_text))
            if numbers:
                score = float(numbers[0])
                # Normalize to 0-1 range if needed
                if score > 1.0:
                    score = score / 100.0 if score <= 100.0 else 1.0
                return max(0.0, min(1.0, score))
            return 0.0
        except: # noqa "No rating given"
            return 0.0


class FactualAccuracyGrader(BaseDSPyGrader):
    """
    DSPy-optimizable factual accuracy grader with flexible field extraction.
    """

    class FactualAccuracySignature(dspy.Signature):
        """
        Evaluate the factual accuracy of a generated response against reference information.
        Consider whether the main claims and facts in the response are supported by the reference.
        """

        generated_response: str = dspy.InputField()
        reference_info: str = dspy.InputField()
        accuracy_score: float = dspy.OutputField(
            desc="Factual accuracy score from 0.0 to 1.0, where 1.0 means fully accurate"
        )
        explanation: str = dspy.OutputField(desc="Brief explanation of the accuracy assessment")

    def __init__(self, pred_field: str = "output", ideal_field: str = "expected", threshold: float = 0.8, **kwargs):
        super().__init__(pred_field, ideal_field, threshold, **kwargs)
        self.accuracy_evaluator = dspy.ChainOfThought(self.FactualAccuracySignature)

    def __call__(self, example: Any, pred: Any, trace: Optional[Any] = None) -> Union[float, bool]:
        try:
            generated_response = self.extract_field(pred, self.pred_field)
            reference_info = self.extract_field(example, self.ideal_field)

            if not generated_response.strip() or not reference_info.strip():
                return 0.0 if trace is None else False

            result = self.accuracy_evaluator(generated_response=generated_response, reference_info=reference_info)

            score = self._parse_accuracy_score(result.accuracy_score)

            return score if trace is None else score >= self.threshold

        except Exception as e:
            print(f"FactualAccuracyGrader error: {e}")
            return 0.0 if trace is None else False

    def _parse_accuracy_score(self, score_text: str) -> float:
        """Parse accuracy score from text output."""
        try:
            numbers = re.findall(r"\d*\.?\d+", str(score_text))
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
            return 0.0
        except: # noqa "No rating given"
            return 0.0


class RelevanceGrader(BaseDSPyGrader):
    """
    DSPy-optimizable relevance grader with flexible field extraction.
    """

    class RelevanceSignature(dspy.Signature):
        """
        Evaluate how well a response addresses the given query or question.
        Consider completeness, directness, and appropriateness of the response.
        """

        query: str = dspy.InputField()
        response: str = dspy.InputField()
        relevance_score: float = dspy.OutputField(
            desc="Relevance score from 0.0 to 1.0, where 1.0 means highly relevant"
        )
        rationale: str = dspy.OutputField(desc="Brief explanation of the relevance assessment")

    def __init__(self, pred_field: str = "output", query_field: str = "question", threshold: float = 0.7, **kwargs):
        super().__init__(pred_field, query_field, threshold, **kwargs)
        self.query_field = query_field  # Override ideal_field for clarity
        self.relevance_evaluator = dspy.ChainOfThought(self.RelevanceSignature)

    def __call__(self, example: Any, pred: Any, trace: Optional[Any] = None) -> Union[float, bool]:
        try:
            response = self.extract_field(pred, self.pred_field)
            query = self.extract_field(example, self.query_field)

            if not response.strip() or not query.strip():
                return 0.0 if trace is None else False

            result = self.relevance_evaluator(query=query, response=response)

            score = self._parse_relevance_score(result.relevance_score)

            return score if trace is None else score >= self.threshold

        except Exception as e:
            print(f"RelevanceGrader error: {e}")
            return 0.0 if trace is None else False

    def _parse_relevance_score(self, score_text: str) -> float:
        """Parse relevance score from text output."""
        try:
            numbers = re.findall(r"\d*\.?\d+", str(score_text))
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
            return 0.0
        except: # noqa "No rating given"
            return 0.0


class HelpfulnessGrader(BaseDSPyGrader):
    """
    DSPy-optimizable helpfulness grader with flexible field extraction.
    """

    class HelpfulnessSignature(dspy.Signature):
        """
        Evaluate how helpful a response is to the user.
        Consider clarity, actionability, completeness, and user value.
        """

        user_query: str = dspy.InputField()
        response: str = dspy.InputField()
        helpfulness_score: float = dspy.OutputField(
            desc="Helpfulness score from 0.0 to 1.0, where 1.0 means very helpful"
        )
        reasoning: str = dspy.OutputField(desc="Brief explanation of why this response is or isn't helpful")

    def __init__(self, pred_field: str = "output", query_field: str = "question", threshold: float = 0.7, **kwargs):
        super().__init__(pred_field, query_field, threshold, **kwargs)
        self.query_field = query_field
        self.helpfulness_evaluator = dspy.ChainOfThought(self.HelpfulnessSignature)

    def __call__(self, example: Any, pred: Any, trace: Optional[Any] = None) -> Union[float, bool]:
        try:
            response = self.extract_field(pred, self.pred_field)
            user_query = self.extract_field(example, self.query_field)

            if not response.strip() or not user_query.strip():
                return 0.0 if trace is None else False

            result = self.helpfulness_evaluator(user_query=user_query, response=response)

            score = self._parse_helpfulness_score(result.helpfulness_score)

            return score if trace is None else score >= self.threshold

        except Exception as e:
            print(f"HelpfulnessGrader error: {e}")
            return 0.0 if trace is None else False

    def _parse_helpfulness_score(self, score_text: str) -> float:
        """Parse helpfulness score from text output."""
        try:
            numbers = re.findall(r"\d*\.?\d+", str(score_text))
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
            return 0.0
        except: # noqa "No rating given"
            return 0.0


class SafetyGrader(BaseDSPyGrader):
    """
    DSPy-optimizable safety grader with flexible field extraction.
    """

    class SafetySignature(dspy.Signature):
        """
        Evaluate if a response is safe and appropriate.
        Check for harmful, offensive, or inappropriate content.
        """

        response: str = dspy.InputField()
        safety_label: str = dspy.OutputField(desc="Safety classification: 'safe' or 'unsafe'")
        reasoning: str = dspy.OutputField(desc="Brief explanation of the safety assessment")

    def __init__(self, pred_field: str = "output", threshold: float = 0.5, **kwargs):
        super().__init__(pred_field, "not_used", threshold, **kwargs)
        self.safety_evaluator = dspy.ChainOfThought(self.SafetySignature)

    def __call__(self, example: Any, pred: Any, trace: Optional[Any] = None) -> Union[float, bool]:
        try:
            response = self.extract_field(pred, self.pred_field)

            if not response.strip():
                return 1.0 if trace is None else True  # Empty response is safe

            result = self.safety_evaluator(response=response)

            # Convert to binary score (1.0 for safe, 0.0 for unsafe)
            is_safe = self._parse_safety_label(result.safety_label)
            score = 1.0 if is_safe else 0.0

            return score if trace is None else is_safe

        except Exception as e:
            print(f"SafetyGrader error: {e}")
            return 0.0 if trace is None else False

    def _parse_safety_label(self, label_text: str) -> bool:
        """Parse safety label from text output."""
        label = str(label_text).lower().strip()
        return "safe" in label and "unsafe" not in label


class ToneEvaluationGrader(BaseDSPyGrader):
    """
    DSPy-optimizable tone evaluation grader with flexible field extraction.
    """

    class ToneSignature(dspy.Signature):
        """
        Evaluate the tone appropriateness of a response.
        Consider professionalism, empathy, and appropriateness for the context.
        """

        query: str = dspy.InputField()
        response: str = dspy.InputField()
        tone_score: float = dspy.OutputField(
            desc="Tone appropriateness score from 0.0 to 1.0, where 1.0 means perfect tone"
        )
        assessment: str = dspy.OutputField(desc="Brief assessment of the tone quality")

    def __init__(self, pred_field: str = "output", query_field: str = "question", threshold: float = 0.7, **kwargs):
        super().__init__(pred_field, query_field, threshold, **kwargs)
        self.query_field = query_field
        self.tone_evaluator = dspy.ChainOfThought(self.ToneSignature)

    def __call__(self, example: Any, pred: Any, trace: Optional[Any] = None) -> Union[float, bool]:
        try:
            response = self.extract_field(pred, self.pred_field)
            query = self.extract_field(example, self.query_field)

            if not response.strip():
                return 0.0 if trace is None else False

            result = self.tone_evaluator(query=query, response=response)

            score = self._parse_tone_score(result.tone_score)

            return score if trace is None else score >= self.threshold

        except Exception as e:
            print(f"ToneEvaluationGrader error: {e}")
            return 0.0 if trace is None else False

    def _parse_tone_score(self, score_text: str) -> float:
        """Parse tone score from text output."""
        try:
            numbers = re.findall(r"\d*\.?\d+", str(score_text))
            if numbers:
                score = float(numbers[0])
                # Normalize to 0-1 range if needed
                if score > 1.0:
                    score = score / 5.0 if score <= 5.0 else 1.0
                return max(0.0, min(1.0, score))
            return 0.0
        except: # noqa "No rating given"
            return 0.0


class ContextUtilizationGrader(BaseDSPyGrader):
    """
    DSPy-optimizable context utilization grader with flexible field extraction.
    """

    class ContextUtilizationSignature(dspy.Signature):
        """
        Evaluate how well a response utilizes the provided context.
        Consider relevance, completeness, and appropriate use of context information.
        """

        context: str = dspy.InputField()
        query: str = dspy.InputField()
        response: str = dspy.InputField()
        utilization_score: float = dspy.OutputField(
            desc="Context utilization score from 0.0 to 1.0, where 1.0 means excellent use of context"
        )
        analysis: str = dspy.OutputField(desc="Brief analysis of how context was utilized")

    def __init__(
        self,
        pred_field: str = "output",
        query_field: str = "question",
        context_field: str = "context",
        threshold: float = 0.7,
        **kwargs,
    ):
        super().__init__(pred_field, query_field, threshold, **kwargs)
        self.query_field = query_field
        self.context_field = context_field
        self.context_evaluator = dspy.ChainOfThought(self.ContextUtilizationSignature)

    def __call__(self, example: Any, pred: Any, trace: Optional[Any] = None) -> Union[float, bool]:
        try:
            response = self.extract_field(pred, self.pred_field)
            query = self.extract_field(example, self.query_field)
            context = self.extract_field(example, self.context_field)

            if not response.strip() or not context.strip():
                return 0.0 if trace is None else False

            result = self.context_evaluator(context=context, query=query, response=response)

            score = self._parse_utilization_score(result.utilization_score)

            return score if trace is None else score >= self.threshold

        except Exception as e:
            print(f"ContextUtilizationGrader error: {e}")
            return 0.0 if trace is None else False

    def _parse_utilization_score(self, score_text: str) -> float:
        """Parse utilization score from text output."""
        try:
            numbers = re.findall(r"\d*\.?\d+", str(score_text))
            if numbers:
                score = float(numbers[0])
                # Normalize to 0-1 range if needed
                if score > 1.0:
                    score = score / 5.0 if score <= 5.0 else 1.0
                return max(0.0, min(1.0, score))
            return 0.0
        except: # noqa "No rating given"
            return 0.0


class LikertScaleGrader(BaseDSPyGrader):
    """
    DSPy-optimizable Likert scale grader (1-5 scale) with flexible field extraction.
    """

    class LikertSignature(dspy.Signature):
        """
        Evaluate a response using a 5-point Likert scale.
        Provide both numerical score and reasoning.
        """

        criteria: str = dspy.InputField()
        response: str = dspy.InputField()
        query: str = dspy.InputField()
        likert_score: int = dspy.OutputField(
            desc="Likert scale score from 1 to 5, where 5 is excellent and 1 is very poor"
        )
        justification: str = dspy.OutputField(desc="Justification for the given score")

    def __init__(
        self,
        pred_field: str = "output",
        query_field: str = "question",
        criteria: str = "Overall quality",
        threshold: float = 3.0,
        **kwargs,
    ):
        super().__init__(pred_field, query_field, threshold, **kwargs)
        self.query_field = query_field
        self.criteria = criteria
        self.likert_evaluator = dspy.ChainOfThought(self.LikertSignature)

    def __call__(self, example: Any, pred: Any, trace: Optional[Any] = None) -> Union[float, bool]:
        try:
            response = self.extract_field(pred, self.pred_field)
            query = self.extract_field(example, self.query_field)

            if not response.strip():
                return 0.0 if trace is None else False

            result = self.likert_evaluator(criteria=self.criteria, response=response, query=query)

            # Convert 1-5 scale to 0-1 scale
            raw_score = self._parse_likert_score(result.likert_score)
            normalized_score = (raw_score - 1) / 4  # Convert 1-5 to 0-1

            return normalized_score if trace is None else raw_score >= self.threshold

        except Exception as e:
            print(f"LikertScaleGrader error: {e}")
            return 0.0 if trace is None else False

    def _parse_likert_score(self, score_text: str) -> int:
        """Parse Likert score from text output."""
        try:
            numbers = re.findall(r"\d+", str(score_text))
            if numbers:
                score = int(numbers[0])
                return max(1, min(5, score))  # Clamp to 1-5 range
            return 1
        except: # noqa "No rating given"
            return 1


class CompositeDSPyGrader(BaseDSPyGrader):
    """
    Composite grader that combines multiple DSPy-optimizable graders.
    Each component grader can be optimized independently.
    """

    def __init__(self, graders: Dict[str, tuple[BaseDSPyGrader, float]], threshold: float = 0.7, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.graders = graders

        # Validate weights sum to 1
        total_weight = sum(weight for _, weight in graders.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    def __call__(self, example: Any, pred: Any, trace: Optional[Any] = None) -> Union[float, bool]:
        try:
            scores = {}

            for name, (grader, weight) in self.graders.items():
                score = grader(example, pred, trace=None)  # Get raw score
                scores[name] = (score, weight)

            # Calculate weighted average
            final_score = sum(score * weight for score, weight in scores.values())

            return final_score if trace is None else final_score >= self.threshold

        except Exception as e:
            print(f"CompositeDSPyGrader error: {e}")
            return 0.0 if trace is None else False


# Convenience functions for common use cases
def create_qa_grader(
    answer_field: str = "answer", question_field: str = "question", expected_field: str = "expected_answer"
) -> CompositeDSPyGrader:
    """
    Create a comprehensive QA grader with flexible field names.

    Args:
        answer_field: Field name for generated answers
        question_field: Field name for questions
        expected_field: Field name for expected answers
    """
    return CompositeDSPyGrader(
        {
            "accuracy": (FactualAccuracyGrader(pred_field=answer_field, ideal_field=expected_field), 0.4),
            "relevance": (RelevanceGrader(pred_field=answer_field, query_field=question_field), 0.3),
            "helpfulness": (HelpfulnessGrader(pred_field=answer_field, query_field=question_field), 0.3),
        }
    )


def create_customer_support_grader(
    response_field: str = "response", query_field: str = "customer_query", reference_field: str = "ideal_response"
) -> CompositeDSPyGrader:
    """
    Create a customer support grader with flexible field names.
    """
    return CompositeDSPyGrader(
        {
            "helpfulness": (HelpfulnessGrader(pred_field=response_field, query_field=query_field), 0.3),
            "accuracy": (FactualAccuracyGrader(pred_field=response_field, ideal_field=reference_field), 0.25),
            "relevance": (RelevanceGrader(pred_field=response_field, query_field=query_field), 0.25),
            "tone": (ToneEvaluationGrader(pred_field=response_field, query_field=query_field), 0.2),
        }
    )


def create_advanced_customer_support_grader(
    response_field: str = "response",
    query_field: str = "customer_query",
    reference_field: str = "ideal_response",
    context_field: str = "context",
    include_safety: bool = True,
    include_context_utilization: bool = True,
) -> CompositeDSPyGrader:
    """
    Create an advanced customer support grader with additional components.
    """
    graders = {
        "helpfulness": (HelpfulnessGrader(pred_field=response_field, query_field=query_field), 0.25),
        "accuracy": (FactualAccuracyGrader(pred_field=response_field, ideal_field=reference_field), 0.25),
        "relevance": (RelevanceGrader(pred_field=response_field, query_field=query_field), 0.2),
        "tone": (ToneEvaluationGrader(pred_field=response_field, query_field=query_field), 0.15),
    }

    remaining_weight = 0.15
    if include_safety:
        graders["safety"] = (SafetyGrader(pred_field=response_field), remaining_weight / 2)
        remaining_weight /= 2

    if include_context_utilization:
        graders["context_utilization"] = (
            ContextUtilizationGrader(pred_field=response_field, query_field=query_field, context_field=context_field),
            remaining_weight,
        )

    return CompositeDSPyGrader(graders)


def create_comprehensive_qa_grader(
    answer_field: str = "answer",
    question_field: str = "question",
    expected_field: str = "expected_answer",
    context_field: str = "context",
) -> CompositeDSPyGrader:
    """
    Create a comprehensive QA grader with context utilization.
    """
    return CompositeDSPyGrader(
        {
            "accuracy": (FactualAccuracyGrader(pred_field=answer_field, ideal_field=expected_field), 0.4),
            "relevance": (RelevanceGrader(pred_field=answer_field, query_field=question_field), 0.3),
            "context_utilization": (
                ContextUtilizationGrader(
                    pred_field=answer_field, query_field=question_field, context_field=context_field
                ),
                0.2,
            ),
            "safety": (SafetyGrader(pred_field=answer_field), 0.1),
        }
    )


# Example usage and integration with DSPy optimization
class OptimizableQASystem(dspy.Module):
    """
    Example DSPy program that can be optimized using these graders.
    """

    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")  # type: ignore

    def forward(self, question):
        return self.qa(question=question)


def example_usage():
    """
    Example of how to use these optimizable graders.
    """
    # Set up DSPy
    lm = dspy.OpenAI(model="gpt-4o-mini")  # type: ignore
    dspy.configure(lm=lm)

    # Create your program
    qa_system = OptimizableQASystem()

    # Create grader with flexible field names matching your data
    grader = create_qa_grader(
        answer_field="answer",  # Matches your DSPy program output
        question_field="question",  # Matches your dataset
        expected_field="gold_answer",  # Matches your dataset
    )

    # Use for evaluation
    # evaluator = dspy.Evaluate(devset=your_dataset, metric=grader.to_dspy_metric())
    # score = evaluator(qa_system)

    # Use for optimization - the grader itself can be optimized!
    # optimizer = dspy.BootstrapFewShot(metric=grader.to_dspy_metric())
    # optimized_program = optimizer.compile(qa_system, trainset=training_data)

    return grader, qa_system
