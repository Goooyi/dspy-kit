"""
Tests for DSPy-optimizable graders with flexible field extraction.
"""

import os
import pytest
import dspy
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from dspy_kit.evaluation.graders.dspy_model_graders import (
    BaseDSPyGrader,
    SemanticSimilarityGrader,
    FactualAccuracyGrader,
    RelevanceGrader,
    HelpfulnessGrader,
    CompositeDSPyGrader,
    create_qa_grader,
    create_customer_support_grader,
)


class TestBaseDSPyGrader:
    """Test the base DSPy grader functionality."""

    def test_is_dspy_module(self):
        """Test that grader is a proper DSPy module."""

        class TestGrader(BaseDSPyGrader):
            def __call__(self, example, pred, trace=None):
                return 0.8

        grader = TestGrader()
        assert isinstance(grader, dspy.Module)
        assert hasattr(grader, "forward")
        assert hasattr(grader, "parameters")

    def test_field_extraction_from_dict(self):
        """Test field extraction from dictionary objects."""

        class TestGrader(BaseDSPyGrader):
            def __call__(self, example, pred, trace=None):
                return 0.8

        grader = TestGrader(pred_field="answer", ideal_field="expected")

        # Test dictionary extraction
        pred_dict = {"answer": "Python is a language", "confidence": 0.9}
        example_dict = {"expected": "Python is programming", "question": "What is Python?"}

        pred_text = grader.extract_field(pred_dict, "answer")
        ideal_text = grader.extract_field(example_dict, "expected")

        assert pred_text == "Python is a language"
        assert ideal_text == "Python is programming"

    def test_field_extraction_from_object(self):
        """Test field extraction from objects with attributes."""

        class TestGrader(BaseDSPyGrader):
            def __call__(self, example, pred, trace=None):
                return 0.8

        grader = TestGrader()

        # Test object attribute extraction
        class MockObj:
            def __init__(self):
                self.output = "test output"
                self.expected = "test expected"

        obj = MockObj()
        output_text = grader.extract_field(obj, "output")
        expected_text = grader.extract_field(obj, "expected")

        assert output_text == "test output"
        assert expected_text == "test expected"

    def test_field_extraction_fallback(self):
        """Test field extraction fallback behavior."""

        class TestGrader(BaseDSPyGrader):
            def __call__(self, example, pred, trace=None):
                return 0.8

        grader = TestGrader()

        # Test missing field with default
        empty_dict = {}
        result = grader.extract_field(empty_dict, "missing_field", "default_value")
        assert result == "default_value"

        # Test string conversion fallback
        string_obj = "just a string"
        result = grader.extract_field(string_obj, "any_field")
        assert result == "just a string"

    def test_to_dspy_metric(self):
        """Test conversion to DSPy metric function."""

        class TestGrader(BaseDSPyGrader):
            def __call__(self, example, pred, trace=None):
                return 0.8

        grader = TestGrader()
        metric = grader.to_dspy_metric()

        assert callable(metric)
        result = metric({}, {})
        assert result == 0.8


class TestSemanticSimilarityGrader:
    """Test semantic similarity grader."""

    def setup_method(self):
        """Setup DSPy configuration for tests."""
        dspy.disable_logging()
        if "OPENAI_MODEL" in os.environ:
            lm = dspy.LM(
                model=os.environ["OPENAI_MODEL"],
                temperature=1.0,
                cache=True,
            )
            dspy.configure(lm=lm)

    @patch("dspy.ChainOfThought")
    def test_initialization(self, mock_chain_of_thought):
        """Test grader initialization."""
        grader = SemanticSimilarityGrader(pred_field="generated_answer", ideal_field="reference_answer", threshold=0.8)

        assert grader.pred_field == "generated_answer"
        assert grader.ideal_field == "reference_answer"
        assert grader.threshold == 0.8
        assert isinstance(grader, dspy.Module)
        mock_chain_of_thought.assert_called_once()

    @patch("dspy.ChainOfThought")
    def test_flexible_field_usage(self, mock_chain_of_thought):
        """Test that grader works with different field names."""
        # Mock the DSPy chain of thought
        mock_evaluator = Mock()
        mock_result = Mock()
        mock_result.similarity_score = "0.85"
        mock_evaluator.return_value = mock_result
        mock_chain_of_thought.return_value = mock_evaluator

        grader = SemanticSimilarityGrader(pred_field="custom_output", ideal_field="custom_reference")
        grader.similarity_evaluator = mock_evaluator

        # Test with custom field names
        example = {"custom_reference": "Python is a programming language"}
        pred = {"custom_output": "Python is a coding language"}

        score = grader(example, pred)

        # Verify the evaluator was called with extracted fields
        mock_evaluator.assert_called_once_with(
            predicted_text="Python is a coding language", reference_text="Python is a programming language"
        )
        assert score == 0.85

    def test_score_parsing(self):
        """Test similarity score parsing from various formats."""
        grader = SemanticSimilarityGrader()

        # Test various score formats
        test_cases = [
            ("0.85", 0.85),
            ("85%", 0.85),
            ("Score: 0.75", 0.75),
            ("The similarity is 90 out of 100", 0.9),
            ("invalid", 0.0),
            ("", 0.0),
        ]

        for input_text, expected in test_cases:
            result = grader._parse_similarity_score(input_text)
            assert abs(result - expected) < 0.01, f"Failed for input '{input_text}'"

    @patch("dspy.ChainOfThought")
    def test_trace_mode(self, mock_chain_of_thought):
        """Test grader behavior in trace mode (optimization)."""
        mock_evaluator = Mock()
        mock_result = Mock()
        mock_result.similarity_score = "0.9"
        mock_evaluator.return_value = mock_result
        mock_chain_of_thought.return_value = mock_evaluator

        grader = SemanticSimilarityGrader(threshold=0.8)
        grader.similarity_evaluator = mock_evaluator

        example = {"expected": "test"}
        pred = {"output": "test"}

        # Test evaluation mode (trace=None)
        score = grader(example, pred, trace=None)
        assert score == 0.9

        # Test optimization mode (trace provided)
        passed = grader(example, pred, trace="some_trace")
        assert passed is True  # 0.9 >= 0.8

        # Test with score below threshold
        mock_result.similarity_score = "0.7"
        passed = grader(example, pred, trace="some_trace")
        assert passed is False  # 0.7 < 0.8


class TestFactualAccuracyGrader:
    """Test factual accuracy grader."""

    def setup_method(self):
        """Setup DSPy configuration for tests."""
        dspy.disable_logging()
        if "OPENAI_MODEL" in os.environ:
            lm = dspy.LM(
                model=os.environ["OPENAI_MODEL"],
                temperature=1.0,
                cache=True,
            )
            dspy.configure(lm=lm)

    @patch("dspy.ChainOfThought")
    def test_initialization(self, mock_chain_of_thought):
        """Test grader initialization."""
        grader = FactualAccuracyGrader(pred_field="generated_response", ideal_field="reference_facts")

        assert grader.pred_field == "generated_response"
        assert grader.ideal_field == "reference_facts"
        assert isinstance(grader, dspy.Module)
        mock_chain_of_thought.assert_called_once()

    @patch("dspy.ChainOfThought")
    def test_accuracy_evaluation(self, mock_chain_of_thought):
        """Test factual accuracy evaluation."""
        mock_evaluator = Mock()
        mock_result = Mock()
        mock_result.accuracy_score = "0.9"
        mock_result.explanation = "Mostly accurate with minor details"
        mock_evaluator.return_value = mock_result
        mock_chain_of_thought.return_value = mock_evaluator

        grader = FactualAccuracyGrader()
        grader.accuracy_evaluator = mock_evaluator

        example = {"expected": "Paris is the capital of France"}
        pred = {"output": "Paris is France's capital city"}

        score = grader(example, pred)

        mock_evaluator.assert_called_once_with(
            generated_response="Paris is France's capital city", reference_info="Paris is the capital of France"
        )
        assert score == 0.9


class TestCompositeDSPyGrader:
    """Test composite DSPy grader."""

    def test_initialization(self):
        """Test composite grader initialization."""
        mock_grader1 = Mock()
        mock_grader2 = Mock()

        graders = {"accuracy": (mock_grader1, 0.6), "relevance": (mock_grader2, 0.4)}

        composite = CompositeDSPyGrader(graders)
        assert composite.graders == graders
        assert isinstance(composite, dspy.Module)

    def test_weight_validation(self):
        """Test weight validation."""
        mock_grader1 = Mock()
        mock_grader2 = Mock()

        # Test invalid weights
        invalid_graders = {
            "accuracy": (mock_grader1, 0.6),
            "relevance": (mock_grader2, 0.5),  # Total = 1.1
        }

        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            CompositeDSPyGrader(invalid_graders)

    def test_composite_scoring(self):
        """Test composite scoring calculation."""
        mock_grader1 = Mock()
        mock_grader1.return_value = 0.8
        mock_grader2 = Mock()
        mock_grader2.return_value = 0.6

        graders = {"accuracy": (mock_grader1, 0.7), "relevance": (mock_grader2, 0.3)}

        composite = CompositeDSPyGrader(graders)

        example = {"test": "data"}
        pred = {"test": "prediction"}

        score = composite(example, pred)

        # Should be weighted average: 0.8 * 0.7 + 0.6 * 0.3 = 0.74
        expected_score = 0.8 * 0.7 + 0.6 * 0.3
        assert abs(score - expected_score) < 0.001


class TestConvenienceFunctions:
    """Test convenience functions for creating graders."""

    def test_create_qa_grader(self):
        """Test QA grader creation."""
        grader = create_qa_grader(answer_field="my_answer", question_field="my_question", expected_field="my_expected")

        assert isinstance(grader, CompositeDSPyGrader)
        assert "accuracy" in grader.graders
        assert "relevance" in grader.graders
        assert "helpfulness" in grader.graders

        # Check field configuration
        accuracy_grader = grader.graders["accuracy"][0]
        assert accuracy_grader.pred_field == "my_answer"
        assert accuracy_grader.ideal_field == "my_expected"

    def test_create_customer_support_grader(self):
        """Test customer support grader creation."""
        grader = create_customer_support_grader(
            response_field="agent_response", query_field="customer_query", reference_field="ideal_response"
        )

        assert isinstance(grader, CompositeDSPyGrader)
        assert "helpfulness" in grader.graders
        assert "accuracy" in grader.graders
        assert "relevance" in grader.graders

        # Check field configuration
        helpfulness_grader = grader.graders["helpfulness"][0]
        assert helpfulness_grader.pred_field == "agent_response"
        assert helpfulness_grader.query_field == "customer_query"


class TestIntegrationWithDSPy:
    """Test integration with DSPy evaluation and optimization."""

    def setup_method(self):
        """Setup DSPy configuration for tests."""
        dspy.disable_logging()
        if "OPENAI_MODEL" in os.environ:
            lm = dspy.LM(
                model=os.environ["OPENAI_MODEL"],
                temperature=1.0,
                cache=True,
            )
            dspy.configure(lm=lm)

    def test_metric_conversion(self):
        """Test conversion to DSPy metric."""
        grader = SemanticSimilarityGrader()
        metric = grader.to_dspy_metric()

        assert callable(metric)
        assert hasattr(metric, "__name__")
        assert "SemanticSimilarityGrader" in metric.__name__

    @patch("dspy.ChainOfThought")
    def test_evaluation_workflow(self, mock_chain_of_thought):
        """Test typical evaluation workflow."""
        # Mock the evaluator
        mock_evaluator = Mock()
        mock_result = Mock()
        mock_result.similarity_score = "0.85"
        mock_evaluator.return_value = mock_result
        mock_chain_of_thought.return_value = mock_evaluator

        # Create grader
        grader = SemanticSimilarityGrader(pred_field="answer", ideal_field="expected_answer")
        grader.similarity_evaluator = mock_evaluator

        # Test data with flexible field names
        example = dspy.Example(question="What is Python?", expected_answer="Python is a programming language")

        prediction = dspy.Prediction(answer="Python is a coding language", confidence=0.9)

        # Evaluate
        score = grader(example, prediction)
        assert score == 0.85

        # Test metric conversion
        metric = grader.to_dspy_metric()
        metric_score = metric(example, prediction)
        assert metric_score == 0.85


class TestErrorHandling:
    """Test error handling and edge cases."""

    @patch("dspy.ChainOfThought")
    def test_empty_fields(self, mock_chain_of_thought):
        """Test handling of empty fields."""
        grader = SemanticSimilarityGrader()

        # Test with empty strings
        example = {"expected": ""}
        pred = {"output": "some text"}

        score = grader(example, pred)
        assert score == 0.0

        # Test with missing fields
        example = {"other_field": "value"}
        pred = {"other_field": "value"}

        score = grader(example, pred)
        assert score == 0.0

    @patch("dspy.ChainOfThought")
    def test_exception_handling(self, mock_chain_of_thought):
        """Test exception handling."""
        mock_evaluator = Mock()
        mock_evaluator.side_effect = Exception("API error")
        mock_chain_of_thought.return_value = mock_evaluator

        grader = SemanticSimilarityGrader()
        grader.similarity_evaluator = mock_evaluator

        example = {"expected": "test"}
        pred = {"output": "test"}

        # Should return 0.0 instead of raising exception
        score = grader(example, pred)
        assert score == 0.0

        # In trace mode, should return False
        passed = grader(example, pred, trace="some_trace")
        assert passed is False


if __name__ == "__main__":
    pytest.main([__file__])
