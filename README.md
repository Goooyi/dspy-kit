# dspy-kit

[![PyPI version](https://badge.fury.io/py/dspy-kit.svg)](https://badge.fury.io/py/dspy-kit)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Comprehensive toolkit for DSPy programs: evaluation, synthetic data generation, and red teaming following OpenAI and Anthropic best practices.**

dspy-kit provides a batteries-included, easily extensible toolkit specifically designed for DSPy applications. Built following industry best practices from OpenAI's eval design guide and Anthropic's evaluation methodology, with plans to expand into synthetic data generation and red teaming capabilities.

## 🌟 Key Features

- **🔥 DSPy-Native Integration**: Drop-in compatibility with DSPy's evaluation and optimization workflows
- **🤖 LLM-as-a-Judge**: Advanced model-based evaluation with async support
- **📊 Comprehensive Metrics**: String matching, similarity scoring, custom Python evaluation, and model graders
- **🎯 Domain-Specific**: Pre-built evaluators for customer support, QA, summarization, and more
- **⚡ Async Support**: High-performance batch evaluation with concurrency control
- **🧩 Composable Design**: Mix and match graders with weighted combinations
- **🛡️ Edge Case Handling**: Built-in support for out-of-scope queries, safety checks, and error handling
- **⚙️ Configuration-Driven**: YAML-based configuration for easy customization
- **🔧 Trace-Aware**: Leverage DSPy's trace information for intermediate step validation

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (coming soon)
pip install dspy-kit

# Or install from source
git clone https://github.com/Goooyi/dspy-kit.git
cd dspy-kit
pip install -e .
```

### Basic Usage

```python
import dspy
from dspy_kit import ExactMatchGrader, ScoreModelGrader, CompositeGrader

# Configure DSPy
lm = dspy.OpenAI(model="gpt-4o-mini")
dspy.configure(lm=lm)

# Create your DSPy program
class QASystem(dspy.Module):
    def __init__(self):
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.qa(question=question)

# Set up evaluation
qa_system = QASystem()

# Individual graders
accuracy_grader = ExactMatchGrader(
    input_field="answer", 
    reference_field="gold_answer"
)

quality_grader = ScoreModelGrader(
    prompt_template="Rate this QA response quality (1-5): {{sample.output_text}}",
    range=[1, 5]
)

# Composite evaluation
composite_grader = CompositeGrader({
    "accuracy": (accuracy_grader, 0.6),
    "quality": (quality_grader, 0.4)
})

# Use with DSPy evaluation
evaluator = dspy.Evaluate(
    devset=your_dataset,
    metric=composite_grader.to_dspy_metric()
)

score = evaluator(qa_system)
print(f"Overall score: {score:.3f}")
```

## 📋 Grader Types

### String-Based Graders

```python
from dspy_kit import StringCheckGrader, TextSimilarityGrader

# Exact matching
exact_match = StringCheckGrader(operation="eq")

# Fuzzy matching
fuzzy_match = TextSimilarityGrader(metric="fuzzy_match", threshold=0.8)

# Contains check
contains_check = StringCheckGrader(operation="like")

# Regex matching
regex_match = StringCheckGrader(operation="regex")
```

### Model-Based Graders (LLM-as-a-Judge)

```python
from dspy_kit import (
    ScoreModelGrader, LabelModelGrader, LikertScaleGrader,
    FactualAccuracyGrader, SafetyGrader
)

# Score-based evaluation (1-5 scale)
quality_grader = ScoreModelGrader(
    prompt_template="Rate the helpfulness of this response (1-5): {{sample.output_text}}",
    model="gpt-4o-mini",
    range=[1, 5]
)

# Binary classification
safety_grader = LabelModelGrader(
    prompt_template="Is this response safe? {{sample.output_text}}",
    labels=["safe", "unsafe"],
    passing_labels=["safe"]
)

# Likert scale evaluation
empathy_grader = LikertScaleGrader(
    criteria="Empathy and emotional understanding"
)

# Specialized graders
factual_grader = FactualAccuracyGrader()
safety_grader = SafetyGrader()
```

### Python Code Graders

```python
from dspy_kit import PythonGrader, FuzzyMatchGrader, JSONValidationGrader

# Custom Python logic
custom_grader = PythonGrader("""
def grade(sample, item):
    output = sample.get("output_text", "")
    reference = item.get("reference_answer", "")
    
    # Custom scoring logic
    score = calculate_custom_similarity(output, reference)
    return score
""")

# Pre-built Python graders
fuzzy_grader = FuzzyMatchGrader(threshold=0.8)
json_grader = JSONValidationGrader(required_fields=["name", "email"])
```

## 🎯 Domain-Specific Examples

### Customer Support Agent Evaluation

```python
from dspy_kit import (
    CustomerSupportCompositeGrader,
    IntentAccuracyGrader,
    EscalationDetectionGrader,
    create_advanced_support_grader
)

# Comprehensive customer support evaluation
support_grader = CustomerSupportCompositeGrader(
    include_empathy=True,
    include_escalation=True
)

# Intent classification accuracy
intent_grader = IntentAccuracyGrader(
    valid_intents=["billing", "technical", "cancellation", "general"]
)

# Escalation detection
escalation_grader = EscalationDetectionGrader()

# Use with your DSPy customer support agent
class CustomerSupportAgent(dspy.Module):
    def __init__(self):
        self.classifier = dspy.Predict("query -> intent")
        self.responder = dspy.Predict("query, intent -> response")
    
    def forward(self, query):
        intent = self.classifier(query=query).intent
        response = self.responder(query=query, intent=intent).response
        return dspy.Prediction(query=query, intent=intent, response=response)

# Evaluate
agent = CustomerSupportAgent()
metric = support_grader.to_dspy_metric()
evaluator = dspy.Evaluate(devset=customer_support_data, metric=metric)
score = evaluator(agent)
```

### Async Batch Evaluation

```python
import asyncio
from dspy_kit import ScoreModelGrader

async def evaluate_large_dataset():
    grader = ScoreModelGrader(
        prompt_template="Rate this response (1-5): {{sample.output_text}}",
        range=[1, 5]
    )
    
    # Batch evaluation with concurrency control
    scores = await grader.batch_evaluate(
        examples=large_dataset,
        predictions=model_predictions,
        max_concurrent=10
    )
    
    return sum(scores) / len(scores)

# Run async evaluation
average_score = asyncio.run(evaluate_large_dataset())
```

## ⚙️ Configuration-Driven Evaluation

Create reusable evaluation configurations:

```yaml
# eval_config.yaml
graders:
  accuracy:
    type: ExactMatchGrader
    input_field: answer
    reference_field: gold_answer
    
  quality:
    type: ScoreModelGrader
    prompt_template: "Rate this answer quality (1-5): {{sample.output_text}}"
    model: gpt-4o-mini
    range: [1, 5]
    pass_threshold: 4.0
    
  safety:
    type: SafetyGrader
    
composite:
  weights:
    accuracy: 0.5
    quality: 0.3
    safety: 0.2
```

```python
from dspy_kit import CompositeGrader

# Load from configuration
grader = CompositeGrader.from_config("eval_config.yaml")

# Use with DSPy
metric = grader.to_dspy_metric()
evaluator = dspy.Evaluate(devset=dataset, metric=metric)
score = evaluator(program)
```

## 🔧 Edge Case Handling

```python
from dspy_kit import EdgeCaseAwareGrader, ScoreModelGrader

def is_out_of_scope(example, pred):
    """Check if query is outside domain scope."""
    out_of_scope_keywords = ['weather', 'sports', 'politics']
    query = getattr(example, 'query', '').lower()
    return any(keyword in query for keyword in out_of_scope_keywords)

def is_abusive(example, pred):
    """Check for abusive content."""
    abusive_patterns = ['stupid', 'hate', 'worst']
    query = getattr(example, 'query', '').lower()
    return any(pattern in query for pattern in abusive_patterns)

# Wrap base grader with edge case handling
base_grader = ScoreModelGrader(...)
edge_aware_grader = EdgeCaseAwareGrader(
    base_grader=base_grader,
    edge_case_handlers={
        'out_of_scope': is_out_of_scope,
        'abusive_input': is_abusive
    }
)
```

## 🚀 DSPy Optimization Integration

```python
import dspy
from dspy_kit import create_customer_support_grader

# Create your DSPy program
program = CustomerSupportAgent()

# Set up evaluation metric
metric = create_customer_support_grader().to_dspy_metric()

# Use for optimization
optimizer = dspy.BootstrapFewShot(metric=metric)
optimized_program = optimizer.compile(program, trainset=training_data)

# Evaluate improvement
evaluator = dspy.Evaluate(devset=eval_data, metric=metric)
baseline_score = evaluator(program)
optimized_score = evaluator(optimized_program)

print(f"Improvement: {optimized_score - baseline_score:.3f}")
```

## 📊 Evaluation Best Practices

### 1. **Start Simple, Iterate**
```python
# Begin with basic metrics
basic_grader = ExactMatchGrader()

# Gradually add complexity
advanced_grader = CompositeGrader({
    "accuracy": (ExactMatchGrader(), 0.4),
    "quality": (ScoreModelGrader(...), 0.4),
    "safety": (SafetyGrader(), 0.2)
})
```

### 2. **Use Trace-Aware Evaluation**
```python
def custom_metric(example, pred, trace=None):
    grader = ScoreModelGrader(...)
    
    if trace is not None:
        # Optimization mode - validate intermediate steps
        if not validate_reasoning_steps(trace):
            return False
        return grader(example, pred, trace) >= 0.8
    else:
        # Evaluation mode - return continuous score
        return grader(example, pred, trace)
```

### 3. **Handle Different Use Cases**
```python
# For classification tasks
classification_metric = CompositeGrader({
    "intent_accuracy": (IntentAccuracyGrader(), 0.7),
    "confidence": (ConfidenceGrader(), 0.3)
})

# For generation tasks  
generation_metric = CompositeGrader({
    "relevance": (RelevanceGrader(), 0.3),
    "quality": (ScoreModelGrader(), 0.3),
    "factuality": (FactualAccuracyGrader(), 0.2),
    "safety": (SafetyGrader(), 0.2)
})
```

## 🧪 Testing and Validation

```python
# Test your graders before production use
def test_grader():
    grader = ScoreModelGrader(...)
    
    # Test cases
    test_cases = [
        (good_example, good_prediction, "Should score high"),
        (good_example, bad_prediction, "Should score low"),
        (edge_case_example, any_prediction, "Should handle gracefully")
    ]
    
    for example, pred, description in test_cases:
        score = grader(example, pred)
        print(f"{description}: {score:.3f}")

test_grader()
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/Goooyi/dspy-kit.git
cd dspy-kit

# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check
black --check .
```

## 📖 Documentation

- [Full API Reference](docs/api/)
- [Tutorial Guide](docs/tutorials/)
- [Domain-Specific Examples](examples/)
- [Best Practices Guide](docs/guides/best-practices.md)

## 🎯 Roadmap

- [ ] **Additional Domains**: QA, Summarization, Code Generation
- [ ] **More Model Providers**: Support for Anthropic, Cohere, local models
- [ ] **Advanced Metrics**: BLEU, ROUGE, BERTScore integration
- [ ] **Evaluation Dashboard**: Web UI for evaluation results
- [ ] **Continuous Evaluation**: Integration with monitoring systems
- [ ] **Multi-modal Support**: Image and audio evaluation capabilities

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for the [DSPy](https://github.com/stanfordnlp/dspy) framework
- Follows [OpenAI's evaluation best practices](https://platform.openai.com/docs/guides/evals)
- Inspired by [Anthropic's evaluation methodology](https://docs.anthropic.com/en/docs/test-and-evaluate)

## 🚀 Roadmap

- [x] **Comprehensive Evaluation Framework**: String, model-based, and classification graders
- [ ] **Synthetic Data Generation**: Constitutional AI and alignment-based data generation
- [ ] **Red Teaming**: Adversarial testing and safety evaluation
- [ ] **Additional Domains**: More specialized evaluators for different use cases
- [ ] **Advanced Metrics**: BLEU, ROUGE, BERTScore integration
- [ ] **Evaluation Dashboard**: Web UI for evaluation results
- [ ] **Multi-modal Support**: Image and audio evaluation capabilities

## 🔗 Related Projects

- [DSPy](https://github.com/stanfordnlp/dspy) - The framework this library is built for
- [OpenAI Evals](https://github.com/openai/evals) - OpenAI's evaluation framework
- [LangChain Evaluation](https://python.langchain.com/docs/guides/evaluation/) - LangChain's evaluation tools

---

**Made with ❤️ for the DSPy community**