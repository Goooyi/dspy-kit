"""Classification graders for precision, recall, F1, and accuracy metrics."""

import warnings
from typing import Any, Optional, Union

from .base import CompositeGrader, ConfigurableGrader

# Optional imports with fallbacks
try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    precision_score = None
    recall_score = None
    f1_score = None
    accuracy_score = None
    warnings.warn("sklearn not available, using built-in implementations", stacklevel=2)


class ClassificationGrader(ConfigurableGrader):
    """
    Base class for classification metrics following OpenAI's pattern.

    Handles single-label classification tasks like intent classification
    for customer support agents.
    """

    DEFAULT_CONFIG = {
        "pred": "predicted_label",
        "ideal": "true_label",
        "labels": None,  # List of all possible labels, auto-detected if None
        "average": "macro",  # "macro", "micro", "weighted", "binary", or None
        "zero_division": 0.0,  # Value to return when division by zero
        "pass_threshold": 0.7,
        "normalize_labels": True,
        "case_sensitive": False,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._label_set = None

    def _extract_labels(self, example: Any, pred: Any) -> tuple[str, str]:
        """Extract predicted and true labels from example and prediction."""
        pred_field = getattr(self, "pred", self.DEFAULT_CONFIG["pred"])
        ideal_field = getattr(self, "ideal", self.DEFAULT_CONFIG["ideal"])

        pred_label = self.extract_field(pred, pred_field, "")
        true_label = self.extract_field(example, ideal_field, "")

        # Normalize labels if configured
        normalize_labels = getattr(self, "normalize_labels", self.DEFAULT_CONFIG["normalize_labels"])
        case_sensitive = getattr(self, "case_sensitive", self.DEFAULT_CONFIG["case_sensitive"])

        if normalize_labels:
            pred_label = pred_label.strip()
            true_label = true_label.strip()

        if not case_sensitive:
            pred_label = pred_label.lower()
            true_label = true_label.lower()

        return pred_label, true_label

    def _get_labels(self, true_labels: list[str], pred_labels: list[str]) -> list[str]:
        """Get the set of labels to use for calculation."""
        configured_labels = getattr(self, "labels", self.DEFAULT_CONFIG["labels"])
        if configured_labels:
            return configured_labels

        # Auto-detect labels from data
        all_labels = set(true_labels + pred_labels)
        return sorted(all_labels)

    def _calculate_confusion_matrix_elements(
        self, true_labels: list[str], pred_labels: list[str], label: str
    ) -> dict[str, int]:
        """Calculate TP, FP, FN, TN for a specific label."""
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p == label)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != label and p == label)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p != label)
        tn = sum(1 for t, p in zip(true_labels, pred_labels) if t != label and p != label)

        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


class PrecisionGrader(ClassificationGrader):
    """
    Precision grader for classification tasks.

    Precision = TP / (TP + FP)
    Measures the accuracy of positive predictions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._accumulated_preds = []
        self._accumulated_trues = []
        self._use_batch_mode = kwargs.get('use_batch_mode', False)

    def __call__(self, example: Any, pred: Any, trace: Optional[Any] = None) -> Union[float, bool]:
        try:
            pred_label, true_label = self._extract_labels(example, pred)

            # If batch mode is enabled, accumulate and return batch precision
            if self._use_batch_mode:
                self._accumulated_preds.append(pred_label)
                self._accumulated_trues.append(true_label)
                
                # Calculate precision on accumulated data
                precision = self._calculate_precision(self._accumulated_trues, self._accumulated_preds)
            else:
                # For single example, calculate instance-level precision
                if pred_label == true_label:
                    precision = 1.0
                else:
                    precision = 0.0

            if trace is None:  # Evaluation mode
                return precision
            else:  # Optimization mode
                pass_threshold = getattr(self, "pass_threshold", self.DEFAULT_CONFIG["pass_threshold"])
                return precision >= pass_threshold

        except Exception as e:
            print(f"PrecisionGrader error: {e}")
            return 0.0 if trace is None else False

    def reset_accumulator(self):
        """Reset the accumulated predictions and true labels."""
        self._accumulated_preds = []
        self._accumulated_trues = []

    def batch_calculate(self, examples: list[Any], predictions: list[Any]) -> float:
        """Calculate precision over a batch of examples."""
        true_labels = []
        pred_labels = []

        for example, pred in zip(examples, predictions):
            pred_label, true_label = self._extract_labels(example, pred)
            true_labels.append(true_label)
            pred_labels.append(pred_label)

        return self._calculate_precision(true_labels, pred_labels)

    def _calculate_precision(self, true_labels: list[str], pred_labels: list[str]) -> float:
        """Calculate precision with averaging strategy."""
        if SKLEARN_AVAILABLE and precision_score is not None:
            try:
                average = getattr(self, "average", self.DEFAULT_CONFIG["average"])
                zero_division = getattr(self, "zero_division", self.DEFAULT_CONFIG["zero_division"])
                labels = self._get_labels(true_labels, pred_labels)

                return precision_score(
                    true_labels, pred_labels, labels=labels, average=average, zero_division=zero_division
                )
            except Exception:
                pass

        # Fallback implementation
        return self._manual_precision(true_labels, pred_labels)

    def _manual_precision(self, true_labels: list[str], pred_labels: list[str]) -> float:
        """Manual precision calculation."""
        labels = self._get_labels(true_labels, pred_labels)
        average = getattr(self, "average", self.DEFAULT_CONFIG["average"])
        zero_division = getattr(self, "zero_division", self.DEFAULT_CONFIG["zero_division"])

        if average == "micro":
            # Micro-average: calculate metrics globally
            total_tp = sum(
                self._calculate_confusion_matrix_elements(true_labels, pred_labels, label)["tp"] for label in labels
            )
            total_fp = sum(
                self._calculate_confusion_matrix_elements(true_labels, pred_labels, label)["fp"] for label in labels
            )

            if total_tp + total_fp == 0:
                return zero_division
            return total_tp / (total_tp + total_fp)

        else:
            # Macro or weighted average: calculate per-label then average
            label_precisions = []
            label_weights = []

            for label in labels:
                cm = self._calculate_confusion_matrix_elements(true_labels, pred_labels, label)
                if cm["tp"] + cm["fp"] == 0:
                    precision = zero_division
                else:
                    precision = cm["tp"] / (cm["tp"] + cm["fp"])

                label_precisions.append(precision)

                if average == "weighted":
                    # Weight by support (number of true instances)
                    support = cm["tp"] + cm["fn"]
                    label_weights.append(support)
                else:
                    label_weights.append(1.0)

            if sum(label_weights) == 0:
                return zero_division

            return sum(p * w for p, w in zip(label_precisions, label_weights)) / sum(label_weights)


class RecallGrader(ClassificationGrader):
    """
    Recall grader for classification tasks.

    Recall = TP / (TP + FN)
    Measures the ability to find all positive instances.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._accumulated_preds = []
        self._accumulated_trues = []
        self._use_batch_mode = kwargs.get('use_batch_mode', False)

    def __call__(self, example: Any, pred: Any, trace: Optional[Any] = None) -> Union[float, bool]:
        try:
            pred_label, true_label = self._extract_labels(example, pred)

            # If batch mode is enabled, accumulate and return batch recall
            if self._use_batch_mode:
                self._accumulated_preds.append(pred_label)
                self._accumulated_trues.append(true_label)
                
                # Calculate recall on accumulated data
                recall = self._calculate_recall(self._accumulated_trues, self._accumulated_preds)
            else:
                # For single example, calculate instance-level recall
                if pred_label == true_label:
                    recall = 1.0
                else:
                    recall = 0.0

            if trace is None:  # Evaluation mode
                return recall
            else:  # Optimization mode
                pass_threshold = getattr(self, "pass_threshold", self.DEFAULT_CONFIG["pass_threshold"])
                return recall >= pass_threshold

        except Exception as e:
            print(f"RecallGrader error: {e}")
            return 0.0 if trace is None else False

    def reset_accumulator(self):
        """Reset the accumulated predictions and true labels."""
        self._accumulated_preds = []
        self._accumulated_trues = []

    def batch_calculate(self, examples: list[Any], predictions: list[Any]) -> float:
        """Calculate recall over a batch of examples."""
        true_labels = []
        pred_labels = []

        for example, pred in zip(examples, predictions):
            pred_label, true_label = self._extract_labels(example, pred)
            true_labels.append(true_label)
            pred_labels.append(pred_label)

        return self._calculate_recall(true_labels, pred_labels)

    def _calculate_recall(self, true_labels: list[str], pred_labels: list[str]) -> float:
        """Calculate recall with averaging strategy."""
        if SKLEARN_AVAILABLE and recall_score is not None:
            try:
                average = getattr(self, "average", self.DEFAULT_CONFIG["average"])
                zero_division = getattr(self, "zero_division", self.DEFAULT_CONFIG["zero_division"])
                labels = self._get_labels(true_labels, pred_labels)

                return recall_score(
                    true_labels, pred_labels, labels=labels, average=average, zero_division=zero_division
                )
            except Exception:
                pass

        # Fallback implementation
        return self._manual_recall(true_labels, pred_labels)

    def _manual_recall(self, true_labels: list[str], pred_labels: list[str]) -> float:
        """Manual recall calculation."""
        labels = self._get_labels(true_labels, pred_labels)
        average = getattr(self, "average", self.DEFAULT_CONFIG["average"])
        zero_division = getattr(self, "zero_division", self.DEFAULT_CONFIG["zero_division"])

        if average == "micro":
            # Micro-average: calculate metrics globally
            total_tp = sum(
                self._calculate_confusion_matrix_elements(true_labels, pred_labels, label)["tp"] for label in labels
            )
            total_fn = sum(
                self._calculate_confusion_matrix_elements(true_labels, pred_labels, label)["fn"] for label in labels
            )

            if total_tp + total_fn == 0:
                return zero_division
            return total_tp / (total_tp + total_fn)

        else:
            # Macro or weighted average: calculate per-label then average
            label_recalls = []
            label_weights = []

            for label in labels:
                cm = self._calculate_confusion_matrix_elements(true_labels, pred_labels, label)
                if cm["tp"] + cm["fn"] == 0:
                    recall = zero_division
                else:
                    recall = cm["tp"] / (cm["tp"] + cm["fn"])

                label_recalls.append(recall)

                if average == "weighted":
                    # Weight by support (number of true instances)
                    support = cm["tp"] + cm["fn"]
                    label_weights.append(support)
                else:
                    label_weights.append(1.0)

            if sum(label_weights) == 0:
                return zero_division

            return sum(r * w for r, w in zip(label_recalls, label_weights)) / sum(label_weights)


class F1Grader(ClassificationGrader):
    """
    F1 score grader for classification tasks.

    F1 = 2 * (precision * recall) / (precision + recall)
    Harmonic mean of precision and recall.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._accumulated_preds = []
        self._accumulated_trues = []
        self._use_batch_mode = kwargs.get('use_batch_mode', False)

    def __call__(self, example: Any, pred: Any, trace: Optional[Any] = None) -> Union[float, bool]:
        try:
            pred_label, true_label = self._extract_labels(example, pred)

            # If batch mode is enabled, accumulate and return batch F1
            if self._use_batch_mode:
                self._accumulated_preds.append(pred_label)
                self._accumulated_trues.append(true_label)
                
                # Calculate F1 on accumulated data
                f1 = self._calculate_f1(self._accumulated_trues, self._accumulated_preds)
            else:
                # For single example, calculate instance-level F1
                if pred_label == true_label:
                    f1 = 1.0
                else:
                    f1 = 0.0

            if trace is None:  # Evaluation mode
                return f1
            else:  # Optimization mode
                pass_threshold = getattr(self, "pass_threshold", self.DEFAULT_CONFIG["pass_threshold"])
                return f1 >= pass_threshold

        except Exception as e:
            print(f"F1Grader error: {e}")
            return 0.0 if trace is None else False

    def reset_accumulator(self):
        """Reset the accumulated predictions and true labels."""
        self._accumulated_preds = []
        self._accumulated_trues = []

    def batch_calculate(self, examples: list[Any], predictions: list[Any]) -> float:
        """Calculate F1 score over a batch of examples."""
        true_labels = []
        pred_labels = []

        for example, pred in zip(examples, predictions):
            pred_label, true_label = self._extract_labels(example, pred)
            true_labels.append(true_label)
            pred_labels.append(pred_label)

        return self._calculate_f1(true_labels, pred_labels)

    def _calculate_f1(self, true_labels: list[str], pred_labels: list[str]) -> float:
        """Calculate F1 score with averaging strategy."""
        if SKLEARN_AVAILABLE and f1_score is not None:
            try:
                average = getattr(self, "average", self.DEFAULT_CONFIG["average"])
                zero_division = getattr(self, "zero_division", self.DEFAULT_CONFIG["zero_division"])
                labels = self._get_labels(true_labels, pred_labels)

                return f1_score(true_labels, pred_labels, labels=labels, average=average, zero_division=zero_division)
            except Exception:
                pass

        # Fallback implementation
        return self._manual_f1(true_labels, pred_labels)

    def _manual_f1(self, true_labels: list[str], pred_labels: list[str]) -> float:
        """Manual F1 calculation."""
        # Calculate precision and recall first
        precision_grader = PrecisionGrader(**self.config)
        recall_grader = RecallGrader(**self.config)

        precision = precision_grader._calculate_precision(true_labels, pred_labels)
        recall = recall_grader._calculate_recall(true_labels, pred_labels)

        # Calculate F1
        zero_division = getattr(self, "zero_division", self.DEFAULT_CONFIG["zero_division"])
        if precision + recall == 0:
            return zero_division

        return 2 * (precision * recall) / (precision + recall)


class AccuracyGrader(ClassificationGrader):
    """
    Accuracy grader for classification tasks.

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Overall correctness of the classifier.
    """

    def __call__(self, example: Any, pred: Any, trace: Optional[Any] = None) -> Union[float, bool]:
        try:
            pred_label, true_label = self._extract_labels(example, pred)

            # For single example, accuracy is 1 if correct, 0 if incorrect
            accuracy = 1.0 if pred_label == true_label else 0.0

            if trace is None:  # Evaluation mode
                return accuracy
            else:  # Optimization mode
                pass_threshold = getattr(self, "pass_threshold", self.DEFAULT_CONFIG["pass_threshold"])
                return accuracy >= pass_threshold

        except Exception as e:
            print(f"AccuracyGrader error: {e}")
            return 0.0 if trace is None else False

    def batch_calculate(self, examples: list[Any], predictions: list[Any]) -> float:
        """Calculate accuracy over a batch of examples."""
        true_labels = []
        pred_labels = []

        for example, pred in zip(examples, predictions):
            pred_label, true_label = self._extract_labels(example, pred)
            true_labels.append(true_label)
            pred_labels.append(pred_label)

        return self._calculate_accuracy(true_labels, pred_labels)

    def _calculate_accuracy(self, true_labels: list[str], pred_labels: list[str]) -> float:
        """Calculate accuracy."""
        if SKLEARN_AVAILABLE and accuracy_score is not None:
            try:
                return accuracy_score(true_labels, pred_labels)
            except Exception:
                pass

        # Fallback implementation
        if len(true_labels) == 0:
            return 0.0

        correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
        return correct / len(true_labels)


class ClassificationMetricsGrader(CompositeGrader):
    """
    Comprehensive classification evaluation combining all metrics.

    Ideal for customer support intent classification and similar tasks.
    """

    def __init__(self, weights: Optional[dict[str, float]] = None, average: str = "macro", **grader_kwargs):
        # Default weights if not provided
        if weights is None:
            weights = {"accuracy": 0.3, "precision": 0.25, "recall": 0.25, "f1": 0.2}

        # Create individual graders
        grader_config = {"average": average, **grader_kwargs}
        batch_grader_config = {"average": average, "use_batch_mode": True, **grader_kwargs}
        graders = {
            "accuracy": (AccuracyGrader(**grader_config), weights.get("accuracy", 0.3)),
            "precision": (PrecisionGrader(**batch_grader_config), weights.get("precision", 0.25)),
            "recall": (RecallGrader(**batch_grader_config), weights.get("recall", 0.25)),
            "f1": (F1Grader(**batch_grader_config), weights.get("f1", 0.2)),
        }

        super().__init__(graders, name="ClassificationMetrics")


class IntentClassificationGrader(ClassificationMetricsGrader):
    """
    Specialized grader for customer support intent classification.

    Optimized for common customer support scenarios.
    """

    def __init__(self, intents: Optional[list[str]] = None, **kwargs):
        # Common customer support intents
        if intents is None:
            intents = [
                "billing",
                "technical_support",
                "account_management",
                "product_inquiry",
                "complaint",
                "cancellation",
                "general_inquiry",
                "escalation",
            ]

        # Configure for intent classification
        grader_config = {
            "labels": intents,
            "pred": "intent",
            "ideal": "expected_intent",
            "average": "weighted",  # Weight by support since some intents are more common
            "normalize_labels": True,
            "case_sensitive": False,
            **kwargs,
        }

        # Weight F1 and recall higher for intent classification
        weights = {"accuracy": 0.2, "precision": 0.2, "recall": 0.3, "f1": 0.3}

        super().__init__(weights=weights, **grader_config)


# Convenience functions for creating classification graders
def create_intent_classifier_grader(
    intents: Optional[list[str]] = None, average: str = "weighted"
) -> IntentClassificationGrader:
    """Create a grader optimized for intent classification."""
    return IntentClassificationGrader(intents=intents, average=average)


def create_classification_grader(
    labels: Optional[list[str]] = None, average: str = "macro", weights: Optional[dict[str, float]] = None
) -> ClassificationMetricsGrader:
    """Create a general classification grader."""
    return ClassificationMetricsGrader(weights=weights, average=average, labels=labels)


def create_binary_classification_grader(
    positive_label: str = "positive", negative_label: str = "negative"
) -> ClassificationMetricsGrader:
    """Create a binary classification grader."""
    return ClassificationMetricsGrader(labels=[negative_label, positive_label], average="binary")
