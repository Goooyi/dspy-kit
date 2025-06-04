"""Tests for classification graders."""

import pytest
import dspy
from dspy_kit import (
    PrecisionGrader,
    RecallGrader,
    F1Grader,
    AccuracyGrader,
    ClassificationMetricsGrader,
    IntentClassificationGrader,
    create_intent_classifier_grader,
    create_classification_grader,
    create_binary_classification_grader,
)


class TestBasicClassificationGraders:
    """Test individual classification graders."""

    def setup_method(self):
        """Setup test data."""
        self.perfect_example = dspy.Example(true_label="billing")
        self.perfect_pred = dspy.Prediction(predicted_label="billing")
        
        self.wrong_example = dspy.Example(true_label="billing") 
        self.wrong_pred = dspy.Prediction(predicted_label="technical")
        
        # Batch test data
        self.batch_examples = [
            dspy.Example(true_label="billing"),
            dspy.Example(true_label="technical"),
            dspy.Example(true_label="billing"),
            dspy.Example(true_label="general"),
        ]
        
        self.batch_predictions = [
            dspy.Prediction(predicted_label="billing"),    # correct
            dspy.Prediction(predicted_label="technical"),  # correct  
            dspy.Prediction(predicted_label="technical"),  # wrong
            dspy.Prediction(predicted_label="general"),    # correct
        ]

    def test_precision_grader_single_correct(self):
        """Test precision grader with single correct prediction."""
        grader = PrecisionGrader()
        score = grader(self.perfect_example, self.perfect_pred)
        assert score == 1.0

    def test_precision_grader_single_wrong(self):
        """Test precision grader with single wrong prediction."""
        grader = PrecisionGrader()
        score = grader(self.wrong_example, self.wrong_pred)
        assert score == 0.0

    def test_precision_grader_batch(self):
        """Test precision grader with batch calculation."""
        grader = PrecisionGrader(average="macro")
        score = grader.batch_calculate(self.batch_examples, self.batch_predictions)
        assert 0.0 <= score <= 1.0

    def test_recall_grader_single_correct(self):
        """Test recall grader with single correct prediction."""
        grader = RecallGrader()
        score = grader(self.perfect_example, self.perfect_pred)
        assert score == 1.0

    def test_recall_grader_single_wrong(self):
        """Test recall grader with single wrong prediction."""
        grader = RecallGrader()
        score = grader(self.wrong_example, self.wrong_pred)
        assert score == 0.0

    def test_f1_grader_single_correct(self):
        """Test F1 grader with single correct prediction."""
        grader = F1Grader()
        score = grader(self.perfect_example, self.perfect_pred)
        assert score == 1.0

    def test_f1_grader_single_wrong(self):
        """Test F1 grader with single wrong prediction."""
        grader = F1Grader()
        score = grader(self.wrong_example, self.wrong_pred)
        assert score == 0.0

    def test_accuracy_grader_single_correct(self):
        """Test accuracy grader with single correct prediction."""
        grader = AccuracyGrader()
        score = grader(self.perfect_example, self.perfect_pred)
        assert score == 1.0

    def test_accuracy_grader_single_wrong(self):
        """Test accuracy grader with single wrong prediction."""
        grader = AccuracyGrader()
        score = grader(self.wrong_example, self.wrong_pred)
        assert score == 0.0

    def test_accuracy_grader_batch(self):
        """Test accuracy grader with batch calculation."""
        grader = AccuracyGrader()
        score = grader.batch_calculate(self.batch_examples, self.batch_predictions)
        # 3 out of 4 correct = 0.75
        assert score == 0.75


class TestAveragingStrategies:
    """Test different averaging strategies."""

    def setup_method(self):
        """Setup test data with class imbalance."""
        # Create imbalanced dataset
        self.examples = [
            dspy.Example(true_label="billing"),     # class 0
            dspy.Example(true_label="billing"),     # class 0
            dspy.Example(true_label="billing"),     # class 0
            dspy.Example(true_label="technical"),   # class 1
        ]
        
        self.predictions = [
            dspy.Prediction(predicted_label="billing"),    # correct
            dspy.Prediction(predicted_label="billing"),    # correct
            dspy.Prediction(predicted_label="technical"),  # wrong
            dspy.Prediction(predicted_label="technical"),  # correct
        ]

    def test_macro_averaging(self):
        """Test macro averaging."""
        grader = F1Grader(average="macro")
        score = grader.batch_calculate(self.examples, self.predictions)
        assert 0.0 <= score <= 1.0

    def test_micro_averaging(self):
        """Test micro averaging."""
        grader = F1Grader(average="micro")
        score = grader.batch_calculate(self.examples, self.predictions)
        assert 0.0 <= score <= 1.0

    def test_weighted_averaging(self):
        """Test weighted averaging."""
        grader = F1Grader(average="weighted")
        score = grader.batch_calculate(self.examples, self.predictions)
        assert 0.0 <= score <= 1.0


class TestCompositeGraders:
    """Test composite classification graders."""

    def setup_method(self):
        """Setup test data."""
        self.example = dspy.Example(expected_intent="billing")
        self.pred = dspy.Prediction(intent="billing", predicted_label="billing")

    def test_classification_metrics_grader(self):
        """Test general classification metrics grader."""
        grader = ClassificationMetricsGrader()
        score = grader(self.example, self.pred)
        assert score == 1.0  # Perfect score for correct classification

    def test_intent_classification_grader(self):
        """Test intent-specific classification grader."""
        grader = IntentClassificationGrader()
        score = grader(self.example, self.pred)
        assert score == 1.0  # Perfect score for correct classification

    def test_custom_weights(self):
        """Test custom weights in composite grader."""
        weights = {
            "accuracy": 0.5,
            "precision": 0.3,
            "recall": 0.1,
            "f1": 0.1
        }
        grader = ClassificationMetricsGrader(weights=weights)
        score = grader(self.example, self.pred)
        assert score == 1.0  # Perfect score for correct classification


class TestFieldExtraction:
    """Test field extraction and normalization."""

    def test_custom_field_names(self):
        """Test custom field names."""
        example = dspy.Example(ground_truth="billing")
        pred = dspy.Prediction(model_output="billing")
        
        grader = PrecisionGrader(
            predicted_field="model_output",
            true_field="ground_truth"
        )
        
        score = grader(example, pred)
        assert score == 1.0

    def test_case_insensitive(self):
        """Test case insensitive matching."""
        example = dspy.Example(true_label="BILLING")
        pred = dspy.Prediction(predicted_label="billing")
        
        grader = PrecisionGrader(case_sensitive=False)
        score = grader(example, pred)
        assert score == 1.0

    def test_label_normalization(self):
        """Test label normalization."""
        example = dspy.Example(true_label="  billing  ")
        pred = dspy.Prediction(predicted_label="billing")
        
        grader = PrecisionGrader(normalize_labels=True)
        score = grader(example, pred)
        assert score == 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_labels(self):
        """Test with empty labels."""
        example = dspy.Example(true_label="")
        pred = dspy.Prediction(predicted_label="")
        
        grader = PrecisionGrader()
        score = grader(example, pred)
        assert score == 1.0  # Empty matches empty

    def test_missing_fields(self):
        """Test with missing fields."""
        example = dspy.Example()
        pred = dspy.Prediction()
        
        grader = PrecisionGrader()
        score = grader(example, pred)
        assert score == 1.0  # Empty matches empty

    def test_optimization_mode(self):
        """Test optimization mode (returns boolean)."""
        example = dspy.Example(true_label="billing")
        pred = dspy.Prediction(predicted_label="billing")
        
        grader = PrecisionGrader(pass_threshold=0.8)
        
        # Evaluation mode
        eval_score = grader(example, pred, trace=None)
        assert eval_score == 1.0
        
        # Optimization mode
        opt_score = grader(example, pred, trace="dummy_trace")
        assert opt_score is True


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_intent_classifier_grader(self):
        """Test intent classifier grader creation."""
        grader = create_intent_classifier_grader()
        assert isinstance(grader, IntentClassificationGrader)

    def test_create_classification_grader(self):
        """Test general classification grader creation."""
        grader = create_classification_grader(
            labels=["class1", "class2"],
            average="macro"
        )
        assert isinstance(grader, ClassificationMetricsGrader)

    def test_create_binary_classification_grader(self):
        """Test binary classification grader creation."""
        grader = create_binary_classification_grader()
        assert isinstance(grader, ClassificationMetricsGrader)


class TestDSPyIntegration:
    """Test integration with DSPy evaluation framework."""

    def test_dspy_metric_conversion(self):
        """Test conversion to DSPy metric."""
        grader = PrecisionGrader()
        metric = grader.to_dspy_metric()
        
        example = dspy.Example(true_label="billing")
        pred = dspy.Prediction(predicted_label="billing")
        
        score = metric(example, pred)
        assert score == 1.0

    def test_async_dspy_metric_conversion(self):
        """Test conversion to async DSPy metric."""
        grader = PrecisionGrader()
        async_metric = grader.to_async_dspy_metric()
        
        assert callable(async_metric)
        assert async_metric.__name__ == f"{grader.name}_async_metric"


class TestBatchEvaluation:
    """Test batch evaluation functionality."""

    def setup_method(self):
        """Setup batch test data."""
        self.examples = [
            dspy.Example(true_label="billing"),
            dspy.Example(true_label="technical"),
            dspy.Example(true_label="general"),
        ]
        
        self.predictions = [
            dspy.Prediction(predicted_label="billing"),
            dspy.Prediction(predicted_label="technical"), 
            dspy.Prediction(predicted_label="general"),
        ]

    def test_precision_batch_calculation(self):
        """Test precision batch calculation."""
        grader = PrecisionGrader()
        score = grader.batch_calculate(self.examples, self.predictions)
        assert score == 1.0  # All correct

    def test_recall_batch_calculation(self):
        """Test recall batch calculation."""
        grader = RecallGrader()
        score = grader.batch_calculate(self.examples, self.predictions)
        assert score == 1.0  # All correct

    def test_f1_batch_calculation(self):
        """Test F1 batch calculation."""
        grader = F1Grader()
        score = grader.batch_calculate(self.examples, self.predictions)
        assert score == 1.0  # All correct

    def test_accuracy_batch_calculation(self):
        """Test accuracy batch calculation.""" 
        grader = AccuracyGrader()
        score = grader.batch_calculate(self.examples, self.predictions)
        assert score == 1.0  # All correct


if __name__ == "__main__":
    pytest.main([__file__])