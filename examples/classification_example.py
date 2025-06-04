"""
Example usage of classification graders for customer support intent classification.

This example demonstrates how to use precision, recall, F1, and accuracy graders
to evaluate a DSPy-based customer support intent classifier.
"""

import dspy
from dspy_kit import (
    AccuracyGrader,
    ClassificationMetricsGrader,
    F1Grader,
    IntentClassificationGrader,
    PrecisionGrader,
    RecallGrader,
    create_classification_grader,
    create_intent_classifier_grader,
)

# Configure DSPy
lm = dspy.OpenAI(model="gpt-4o-mini")
dspy.configure(lm=lm)


# Example customer support intent classifier
class CustomerSupportIntentClassifier(dspy.Module):
    def __init__(self):
        self.classifier = dspy.ChainOfThought(
            "customer_query -> intent", signature="Given a customer support query, classify the intent."
        )

    def forward(self, customer_query):
        result = self.classifier(customer_query=customer_query)
        return dspy.Prediction(
            customer_query=customer_query,
            intent=result.intent,
            predicted_label=result.intent,  # For grader compatibility
        )


# Create the classifier
intent_classifier = CustomerSupportIntentClassifier()

# Example dataset
examples = [
    {"customer_query": "I can't log into my account", "expected_intent": "technical_support"},
    {"customer_query": "What's my current bill amount?", "expected_intent": "billing"},
    {"customer_query": "I want to cancel my subscription", "expected_intent": "cancellation"},
    {"customer_query": "How do I reset my password?", "expected_intent": "technical_support"},
    {"customer_query": "I was charged twice this month", "expected_intent": "billing"},
    {"customer_query": "What features are included in the pro plan?", "expected_intent": "product_inquiry"},
]

# Convert to DSPy examples
dataset = []
for item in examples:
    example = dspy.Example(customer_query=item["customer_query"], expected_intent=item["expected_intent"]).with_inputs(
        "customer_query"
    )
    dataset.append(example)

# Generate predictions
predictions = []
for example in dataset:
    pred = intent_classifier(customer_query=example.customer_query)
    predictions.append(pred)

# ==========================================
# 1. Individual Classification Graders
# ==========================================

print("=== Individual Classification Metrics ===")

# Precision grader
precision_grader = PrecisionGrader(
    pred="predicted_label",
    ideal="expected_intent",
    average="macro",  # macro-average across all intent classes
)

# Calculate precision for single example
single_precision = precision_grader(dataset[0], predictions[0])
print(f"Single example precision: {single_precision:.3f}")

# Calculate precision across batch
batch_precision = precision_grader.batch_calculate(dataset, predictions)
print(f"Batch precision (macro): {batch_precision:.3f}")

# Recall grader
recall_grader = RecallGrader(
    pred="predicted_label",
    ideal="expected_intent",
    average="weighted",  # weighted by class frequency
)

batch_recall = recall_grader.batch_calculate(dataset, predictions)
print(f"Batch recall (weighted): {batch_recall:.3f}")

# F1 grader
f1_grader = F1Grader(pred="predicted_label", ideal="expected_intent", average="macro")

batch_f1 = f1_grader.batch_calculate(dataset, predictions)
print(f"Batch F1 (macro): {batch_f1:.3f}")

# Accuracy grader
accuracy_grader = AccuracyGrader(pred="predicted_label", ideal="expected_intent")

batch_accuracy = accuracy_grader.batch_calculate(dataset, predictions)
print(f"Batch accuracy: {batch_accuracy:.3f}")

# ==========================================
# 2. Composite Classification Grader
# ==========================================

print("\n=== Composite Classification Metrics ===")

# Create composite grader with custom weights
classification_grader = ClassificationMetricsGrader(
    weights={"accuracy": 0.3, "precision": 0.25, "recall": 0.25, "f1": 0.2},
    pred="predicted_label",
    ideal="expected_intent",
    average="macro",
)

# Use with DSPy evaluation
metric = classification_grader.to_dspy_metric()
evaluator = dspy.Evaluate(devset=dataset, metric=metric)

# Note: This would normally evaluate the program, but for demo we'll simulate
composite_score = 0.0
for example, pred in zip(dataset, predictions):
    score = classification_grader(example, pred)
    composite_score += score
composite_score /= len(dataset)

print(f"Composite classification score: {composite_score:.3f}")

# ==========================================
# 3. Intent-Specific Grader
# ==========================================

print("\n=== Intent Classification Grader ===")

# Create intent-specific grader
intent_grader = IntentClassificationGrader(
    intents=["billing", "technical_support", "cancellation", "product_inquiry", "general_inquiry"]
)

# Evaluate intent classification
intent_score = 0.0
for example, pred in zip(dataset, predictions):
    score = intent_grader(example, pred)
    intent_score += score
intent_score /= len(dataset)

print(f"Intent classification score: {intent_score:.3f}")

# ==========================================
# 4. Different Averaging Strategies
# ==========================================

print("\n=== Different Averaging Strategies ===")

averaging_strategies = ["macro", "micro", "weighted"]

for avg_strategy in averaging_strategies:
    grader = F1Grader(pred="predicted_label", ideal="expected_intent", average=avg_strategy)

    score = grader.batch_calculate(dataset, predictions)
    print(f"F1 ({avg_strategy}): {score:.3f}")

# ==========================================
# 5. Using with DSPy Optimization
# ==========================================

print("\n=== DSPy Optimization Integration ===")

# Create metric for optimization (returns bool for bootstrapping)
optimization_metric = f1_grader.to_dspy_metric()


# Example of how this would be used in optimization
def demo_optimization_metric(example, pred, trace=None):
    """Demo of how the grader works in optimization context."""
    f1_score = f1_grader(example, pred, trace)

    if trace is None:  # Evaluation mode
        return f1_score
    else:  # Optimization mode - return boolean
        return f1_score >= 0.7  # Pass threshold


print("Optimization metric created for DSPy BootstrapFewShot")

# ==========================================
# 6. Edge Cases and Error Handling
# ==========================================

print("\n=== Edge Cases ===")

# Test with empty/missing labels
edge_example = dspy.Example(customer_query="Test query", expected_intent="").with_inputs("customer_query")

edge_pred = dspy.Prediction(predicted_label="")

edge_score = precision_grader(edge_example, edge_pred)
print(f"Edge case (empty labels) precision: {edge_score:.3f}")

# Test with unknown label
unknown_example = dspy.Example(customer_query="Test query", expected_intent="unknown_intent").with_inputs(
    "customer_query"
)

unknown_pred = dspy.Prediction(predicted_label="billing")

unknown_score = precision_grader(unknown_example, unknown_pred)
print(f"Edge case (unknown intent) precision: {unknown_score:.3f}")

# ==========================================
# 7. Convenience Functions
# ==========================================

print("\n=== Convenience Functions ===")

# Quick intent classifier grader
quick_intent_grader = create_intent_classifier_grader()
quick_score = 0.0
for example, pred in zip(dataset, predictions):
    score = quick_intent_grader(example, pred)
    quick_score += score
quick_score /= len(dataset)

print(f"Quick intent grader score: {quick_score:.3f}")

# General classification grader
general_grader = create_classification_grader(
    labels=["billing", "technical_support", "cancellation", "product_inquiry"], average="weighted"
)

# ==========================================
# 8. Real-world Usage Pattern
# ==========================================

print("\n=== Real-world Usage Pattern ===")


class CustomerSupportEvaluator:
    """Real-world evaluator for customer support systems."""

    def __init__(self):
        self.intent_grader = create_intent_classifier_grader(
            intents=[
                "billing",
                "technical_support",
                "cancellation",
                "product_inquiry",
                "general_inquiry",
                "escalation",
            ],
            average="weighted",
        )

    def evaluate_intent_classifier(self, classifier, test_data):
        """Evaluate intent classifier performance."""
        predictions = []

        for example in test_data:
            pred = classifier(customer_query=example.customer_query)
            predictions.append(pred)

        # Calculate metrics
        accuracy = AccuracyGrader(pred="predicted_label", ideal="expected_intent").batch_calculate(
            test_data, predictions
        )

        precision = PrecisionGrader(
            pred="predicted_label", ideal="expected_intent", average="weighted"
        ).batch_calculate(test_data, predictions)

        recall = RecallGrader(pred="predicted_label", ideal="expected_intent", average="weighted").batch_calculate(
            test_data, predictions
        )

        f1 = F1Grader(pred="predicted_label", ideal="expected_intent", average="weighted").batch_calculate(
            test_data, predictions
        )

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "predictions": predictions}


# Use the evaluator
evaluator = CustomerSupportEvaluator()
results = evaluator.evaluate_intent_classifier(intent_classifier, dataset)

print("Final Evaluation Results:")
for metric, score in results.items():
    if metric != "predictions":
        print(f"{metric.capitalize()}: {score:.3f}")

print("\n=== Classification Graders Demo Complete ===")
