# """
# Enhanced Customer Support Integration Example for dspy-kit.

# This demonstrates comprehensive integration with DSPy customer support agents,
# including async evaluation, optimization workflows, and edge case handling.
# """

# import asyncio
# from pathlib import Path
# from typing import Any, Dict, List

# import dspy
# import yaml
# from dspy.evaluate import Evaluate

# # Import dspy-kit components
# from dspy_kit import (
#     BinaryClassificationGrader,
#     CompositeGrader,
#     EdgeCaseAwareGrader,
#     EmpathyEvaluationGrader,
#     ExactMatchGrader,
#     IntentAccuracyGrader,
#     ProblemResolutionGrader,
#     ScoreModelGrader,
# )


# # Example DSPy Signatures for Customer Support
# class IntentClassification(dspy.Signature):
#     """Classify customer query intent for proper routing."""

#     query: str = dspy.InputField(desc="Customer query text")
#     intent: str = dspy.OutputField(
#         desc="One of: billing, technical_support, account_management, product_inquiry, complaint, cancellation, other"
#     )


# class CustomerSupportResponse(dspy.Signature):
#     """Generate helpful, empathetic customer support response."""

#     query: str = dspy.InputField(desc="Customer query")
#     intent: str = dspy.InputField(desc="Classified intent")
#     customer_context: str = dspy.InputField(desc="Customer history and context", required=False)
#     response: str = dspy.OutputField(desc="Professional, helpful response addressing the customer's needs")


# class EscalationDecision(dspy.Signature):
#     """Determine if customer query needs human escalation."""

#     query: str = dspy.InputField(desc="Customer query")
#     response: str = dspy.InputField(desc="AI-generated response")
#     customer_sentiment: str = dspy.InputField(desc="Customer sentiment", required=False)
#     needs_escalation: bool = dspy.OutputField(desc="True if human escalation needed")


# # Customer Support Agent Implementation
# class CustomerSupportAgent(dspy.Module):
#     """Multi-step customer support agent with intent classification and response generation."""

#     def __init__(self, include_escalation: bool = True):
#         super().__init__()
#         self.intent_classifier = dspy.ChainOfThought(IntentClassification)
#         self.response_generator = dspy.ChainOfThought(CustomerSupportResponse)
#         self.include_escalation = include_escalation

#         if include_escalation:
#             self.escalation_detector = dspy.ChainOfThought(EscalationDecision)

#     def forward(self, query: str, customer_context: str = "") -> dspy.Prediction:
#         # Step 1: Classify intent
#         intent_result = self.intent_classifier(query=query)

#         # Step 2: Generate response
#         response_result = self.response_generator(
#             query=query, intent=intent_result.intent, customer_context=customer_context
#         )

#         # Step 3: Check for escalation (optional)
#         needs_escalation = False
#         if self.include_escalation:
#             escalation_result = self.escalation_detector(query=query, response=response_result.response)
#             needs_escalation = escalation_result.needs_escalation

#         return dspy.Prediction(
#             query=query,
#             intent=intent_result.intent,
#             response=response_result.response,
#             needs_escalation=needs_escalation,
#             customer_context=customer_context,
#         )


# # Evaluation Metrics Setup
# class CustomerSupportEvaluator:
#     """Comprehensive evaluator for customer support systems."""

#     def __init__(self, config_path: str = None):
#         self.config = self._load_config(config_path) if config_path else self._default_config()
#         self.metrics = self._setup_metrics()

#     def _load_config(self, config_path: str) -> Dict:
#         """Load evaluation configuration from YAML file."""
#         with open(config_path) as f:
#             return yaml.safe_load(f)

#     def _default_config(self) -> Dict:
#         """Default evaluation configuration."""
#         return {
#             "weights": {
#                 "intent_accuracy": 0.25,
#                 "response_quality": 0.25,
#                 "empathy": 0.15,
#                 "problem_resolution": 0.20,
#                 "safety": 0.10,
#                 "escalation_accuracy": 0.05,
#             },
#             "thresholds": {"pass_threshold": 0.7, "excellent_threshold": 0.9},
#             "edge_cases": {"check_out_of_scope": True, "check_abusive_content": True, "check_multilingual": False},
#         }

#     def _setup_metrics(self) -> Dict[str, Any]:
#         """Initialize all evaluation metrics."""
#         metrics = {}

#         # 1. Intent Classification Accuracy
#         metrics["intent_accuracy"] = IntentAccuracyGrader(
#             intent_field="intent",
#             reference_intent_field="true_intent",
#             valid_intents=[
#                 "billing",
#                 "technical_support",
#                 "account_management",
#                 "product_inquiry",
#                 "complaint",
#                 "cancellation",
#                 "other",
#             ],
#         )

#         # 2. Response Quality
#         metrics["response_quality"] = ScoreModelGrader(
#             prompt_template="""Rate the quality of this customer support response (1-5):

# Customer Query: {{item.question}}
# Support Response: {{sample.output_text}}
# Customer Context: {{item.customer_context}}

# Evaluate based on:
# - Helpfulness: Addresses customer's specific need
# - Clarity: Easy to understand and actionable
# - Completeness: Provides sufficient detail
# - Professionalism: Appropriate business tone

# Score (1-5):""",
#             model="gpt-4o-mini",
#             range=[1, 5],
#             pass_threshold=4.0,
#         )

#         # 3. Empathy Evaluation
#         metrics["empathy"] = EmpathyEvaluationGrader()

#         # 4. Problem Resolution
#         metrics["problem_resolution"] = ProblemResolutionGrader()

#         # 5. Safety Check
#         metrics["safety"] = BinaryClassificationGrader(
#             question="Is this customer support response safe, appropriate, and professional?",
#             labels=["safe", "unsafe"],
#             passing_labels=["safe"],
#         )

#         # 6. Escalation Accuracy (if applicable)
#         metrics["escalation_accuracy"] = ExactMatchGrader(pred="needs_escalation", ideal="should_escalate")

#         # 7. Edge Case Aware Wrapper
#         if self.config.get("edge_cases", {}).get("check_out_of_scope", False):
#             metrics["response_quality"] = EdgeCaseAwareGrader(
#                 base_grader=metrics["response_quality"],
#                 edge_case_handlers={
#                     "out_of_scope": self._is_out_of_scope,
#                     "abusive_input": self._is_abusive_input,
#                     "multilingual": self._is_multilingual,
#                 },
#             )

#         # 8. Composite Metric
#         weights = self.config["weights"]
#         metrics["composite"] = CompositeGrader(
#             {
#                 name: (grader, weights.get(name, 0.1))
#                 for name, grader in metrics.items()
#                 if name != "composite" and name in weights
#             }
#         )

#         return metrics

#     def _is_out_of_scope(self, example: Any, pred: Any) -> bool:
#         """Check if query is outside customer support scope."""
#         out_of_scope_keywords = ["weather", "sports", "politics", "recipe", "dating"]
#         query = getattr(example, "query", str(example)).lower()
#         return any(keyword in query for keyword in out_of_scope_keywords)

#     def _is_abusive_input(self, example: Any, pred: Any) -> bool:
#         """Check for abusive or inappropriate input."""
#         abusive_patterns = ["stupid", "idiot", "hate you", "worst company"]
#         query = getattr(example, "query", str(example)).lower()
#         return any(pattern in query for pattern in abusive_patterns)

#     def _is_multilingual(self, example: Any, pred: Any) -> bool:
#         """Check for non-English input."""
#         # Simple heuristic - could be enhanced with language detection
#         query = getattr(example, "query", str(example))
#         non_ascii_ratio = sum(1 for c in query if ord(c) > 127) / len(query) if query else 0
#         return non_ascii_ratio > 0.1

#     async def evaluate_async(self, agent: CustomerSupportAgent, examples: List[Any]) -> Dict[str, float]:
#         """Asynchronously evaluate agent on examples."""
#         results = {}

#         for metric_name, metric in self.metrics.items():
#             print(f"📊 Evaluating {metric_name}...")

#             # Run predictions
#             predictions = []
#             for example in examples:
#                 pred = agent(query=example.query, customer_context=getattr(example, "customer_context", ""))
#                 predictions.append(pred)

#             # Batch evaluate with async support
#             if hasattr(metric, "batch_evaluate"):
#                 scores = await metric.batch_evaluate(examples, predictions)
#                 avg_score = sum(scores) / len(scores) if scores else 0.0
#             else:
#                 scores = []
#                 for example, pred in zip(examples, predictions):
#                     score = metric(example, pred)
#                     scores.append(score)
#                 avg_score = sum(scores) / len(scores) if scores else 0.0

#             results[metric_name] = avg_score
#             print(f"   ✅ {metric_name}: {avg_score:.3f}")

#         return results

#     def evaluate_sync(self, agent: CustomerSupportAgent, examples: List[Any]) -> Dict[str, float]:
#         """Synchronously evaluate agent on examples."""
#         return asyncio.run(self.evaluate_async(agent, examples))

#     def get_dspy_metric(self, metric_name: str = "composite"):
#         """Get DSPy-compatible metric function."""
#         if metric_name not in self.metrics:
#             raise ValueError(f"Unknown metric: {metric_name}")
#         return self.metrics[metric_name].to_dspy_metric()


# # Example Usage and Integration
# def create_sample_dataset() -> List[dspy.Example]:
#     """Create sample evaluation dataset."""
#     examples = [
#         {
#             "query": "I want to cancel my subscription immediately",
#             "true_intent": "cancellation",
#             "should_escalate": False,
#             "customer_context": "Premium customer for 2 years, no previous complaints",
#             "expected_satisfaction": 4,
#         },
#         {
#             "query": "Your service is terrible and I'm extremely frustrated!",
#             "true_intent": "complaint",
#             "should_escalate": True,
#             "customer_context": "Multiple previous complaints, escalated twice",
#             "expected_satisfaction": 2,
#         },
#         {
#             "query": "How do I reset my password?",
#             "true_intent": "technical_support",
#             "should_escalate": False,
#             "customer_context": "New customer, first interaction",
#             "expected_satisfaction": 4,
#         },
#         {
#             "query": "I was charged twice this month, please fix this",
#             "true_intent": "billing",
#             "should_escalate": False,
#             "customer_context": "Regular customer, occasional billing questions",
#             "expected_satisfaction": 4,
#         },
#         {
#             "query": "I need help understanding my bill, there are charges I don't recognize",
#             "true_intent": "billing",
#             "should_escalate": False,
#             "customer_context": "Senior customer, prefers detailed explanations",
#             "expected_satisfaction": 4,
#         },
#     ]

#     return [dspy.Example(**ex).with_inputs("query", "customer_context") for ex in examples]


# async def main_async_example():
#     """Main example using async evaluation."""
#     print("🚀 Customer Support Agent Evaluation with dspy-kit")
#     print("=" * 60)

#     # Initialize agent
#     agent = CustomerSupportAgent(include_escalation=True)

#     # Create evaluator
#     evaluator = CustomerSupportEvaluator()

#     # Create sample dataset
#     devset = create_sample_dataset()
#     print(f"📝 Created evaluation dataset with {len(devset)} examples")

#     # Run async evaluation
#     print("\n📊 Running comprehensive evaluation...")
#     results = await evaluator.evaluate_async(agent, devset)

#     # Display results
#     print("\n📈 Evaluation Results:")
#     print("-" * 40)
#     for metric_name, score in results.items():
#         status = "✅ PASS" if score >= 0.7 else "❌ NEEDS IMPROVEMENT"
#         print(f"{metric_name:20s}: {score:.3f} {status}")

#     # Overall assessment
#     overall_score = results.get("composite", 0.0)
#     if overall_score >= 0.9:
#         assessment = "🌟 EXCELLENT"
#     elif overall_score >= 0.7:
#         assessment = "✅ GOOD"
#     elif overall_score >= 0.5:
#         assessment = "⚠️  NEEDS IMPROVEMENT"
#     else:
#         assessment = "❌ POOR"

#     print(f"\n🎯 Overall Assessment: {assessment} ({overall_score:.3f})")

#     return results


# def dspy_optimization_example():
#     """Example of using metrics for DSPy optimization."""
#     print("\n🔧 DSPy Optimization Example")
#     print("=" * 40)

#     # Initialize components
#     agent = CustomerSupportAgent()
#     evaluator = CustomerSupportEvaluator()

#     # Create datasets
#     devset = create_sample_dataset()
#     trainset = devset[:3]  # Small training set for demo

#     # Get DSPy-compatible metric
#     metric = evaluator.get_dspy_metric("composite")

#     # Baseline evaluation
#     print("📊 Baseline Evaluation:")
#     baseline_evaluator = Evaluate(devset=devset, metric=metric, num_threads=1, display_progress=True)
#     baseline_score = baseline_evaluator(agent)
#     print(f"🎯 Baseline Score: {baseline_score:.3f}")

#     # Optimization using DSPy
#     print("\n🚀 Running DSPy Optimization...")
#     try:
#         optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=2)
#         optimized_agent = optimizer.compile(agent, trainset=trainset)

#         # Evaluate optimized agent
#         print("📊 Optimized Agent Evaluation:")
#         optimized_score = baseline_evaluator(optimized_agent)
#         print(f"🎯 Optimized Score: {optimized_score:.3f}")

#         improvement = optimized_score - baseline_score
#         print(f"📈 Improvement: {improvement:.3f} ({improvement / baseline_score * 100:.1f}%)")

#         return optimized_agent, optimized_score

#     except Exception as e:
#         print(f"⚠️  Optimization failed: {e}")
#         return agent, baseline_score


# def config_driven_example():
#     """Example using configuration file."""
#     print("\n⚙️  Configuration-Driven Evaluation")
#     print("=" * 40)

#     # Create sample config
#     config = {
#         "weights": {"intent_accuracy": 0.3, "response_quality": 0.4, "empathy": 0.2, "safety": 0.1},
#         "thresholds": {"pass_threshold": 0.8, "excellent_threshold": 0.95},
#         "edge_cases": {"check_out_of_scope": True, "check_abusive_content": True},
#     }

#     # Save config temporarily
#     config_path = "temp_config.yaml"
#     with open(config_path, "w") as f:
#         yaml.dump(config, f)

#     try:
#         # Use config-driven evaluator
#         evaluator = CustomerSupportEvaluator(config_path=config_path)
#         agent = CustomerSupportAgent()
#         devset = create_sample_dataset()

#         results = evaluator.evaluate_sync(agent, devset)
#         print("📊 Config-driven results:")
#         for metric, score in results.items():
#             print(f"   {metric}: {score:.3f}")

#     finally:
#         # Cleanup
#         Path(config_path).unlink(missing_ok=True)


# if __name__ == "__main__":
#     # Configure DSPy (adjust based on your setup)
#     try:
#         lm = dspy.OpenAI(model="gpt-4o-mini")
#         dspy.settings.configure(lm=lm)
#         print("✅ DSPy configured with OpenAI")
#     except Exception as e:
#         print(f"⚠️  DSPy configuration failed: {e}")
#         print("   Please ensure OPENAI_API_KEY is set and dspy is properly installed")
#         exit(1)

#     # Run examples
#     print("🎯 Running Customer Support Integration Examples\n")

#     # 1. Async evaluation
#     results = asyncio.run(main_async_example())

#     # 2. DSPy optimization
#     optimized_agent, final_score = dspy_optimization_example()

#     # 3. Config-driven approach
#     config_driven_example()

#     print("\n🎉 All examples completed successfully!")
#     print("\n💡 Next steps:")
#     print("   1. Adapt the agent signatures to your specific use case")
#     print("   2. Create your own evaluation dataset")
#     print("   3. Customize the grader configurations")
#     print("   4. Integrate with your DSPy optimization pipeline")
#     print("   5. Set up continuous evaluation for production monitoring")
