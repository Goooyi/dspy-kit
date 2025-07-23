> [!WARNING]
>  In construction, expect breaking changes

# dspy-kit

[![PyPI version](https://badge.fury.io/py/dspy-kit.svg)](https://badge.fury.io/py/dspy-kit)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Comprehensive toolkit for DSPy programs: evaluation, synthetic data generation, and red teaming following OpenAI and Anthropic best practices.**

dspy-kit provides a batteries-included, easily extensible toolkit specifically designed for DSPy applications. Built following industry best practices from OpenAI's eval design guide and Anthropic's evaluation methodology, with plans to expand into synthetic data generation and red teaming capabilities.

## 🌟 Key Features

- **🔥 DSPy-Optimizable Graders**: LLM-as-a-Judge graders that can be optimized with DSPy compilation
- **🎯 Flexible Field Extraction**: Works with any field naming scheme - no more hardcoded field requirements
- **🤖 Advanced Model-Based Evaluation**: Semantic similarity, factual accuracy, relevance, and safety graders
- **📊 Comprehensive Metrics**: String matching, similarity scoring, custom Python evaluation, and optimizable model graders
- **🚀 Best of Both Worlds**: Combines DSPy optimization capabilities with flexible field handling
- **⚡ Production-Ready**: Graceful error handling, async support, and batch evaluation
- **🧩 Composable Design**: Mix and match optimizable graders with weighted combinations
- **🛡️ Edge Case Handling**: Built-in support for out-of-scope queries, safety checks, and error handling
- **🔧 Trace-Aware**: Leverage DSPy's trace information for intermediate step validation
- **🌍 i18n Support**: Multi-language template system with cultural adaptations
- **📝 Modular Prompt Templates**: YAML + Jinja2 templates with inheritance and validation

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
from dspy_kit import (
    SemanticSimilarityGrader,
    DSPyFactualAccuracyGrader,
    create_dspy_qa_grader
)

# Configure DSPy
lm = dspy.OpenAI(model="gpt-4o-mini")
dspy.configure(lm=lm)

# Create your DSPy program with custom field names
class QASystem(dspy.Module):
    def __init__(self):
        self.qa = dspy.ChainOfThought("question -> detailed_answer")

    def forward(self, question):
        return self.qa(question=question)

# Set up evaluation with flexible field names
qa_system = QASystem()

# Works with ANY field names - no restructuring needed!
grader = create_dspy_qa_grader(
    answer_field="detailed_answer",    # Matches your program output
    question_field="user_question",    # Matches your dataset
    expected_field="reference_answer"  # Matches your dataset
)

# Use for evaluation AND optimization
evaluator = dspy.Evaluate(
    devset=your_dataset,
    metric=grader.to_dspy_metric()
)

# Evaluate current performance
score = evaluator(qa_system)
print(f"Current score: {score:.3f}")

# Optimize the program AND the grader!
optimizer = dspy.BootstrapFewShot(metric=grader.to_dspy_metric())
optimized_program = optimizer.compile(qa_system, trainset=training_data)

# Measure improvement
optimized_score = evaluator(optimized_program)
print(f"Improvement: {optimized_score - score:.3f}")
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

### DSPy-Optimizable Graders (LLM-as-a-Judge)

```python
from dspy_kit import (
    SemanticSimilarityGrader,
    DSPyFactualAccuracyGrader,
    DSPyRelevanceGrader,
    HelpfulnessGrader,
    DSPySafetyGrader
)

# Semantic similarity with flexible field names
similarity_grader = SemanticSimilarityGrader(
    pred_field="generated_response",    # Your field name
    ideal_field="reference_text",       # Your field name
    threshold=0.8
)

# Factual accuracy evaluation
accuracy_grader = DSPyFactualAccuracyGrader(
    pred_field="model_output",         # Any field name
    ideal_field="ground_truth",        # Any field name
    threshold=0.8
)

# Helpfulness assessment
helpfulness_grader = HelpfulnessGrader(
    pred_field="ai_response",          # Flexible naming
    query_field="user_query"           # Flexible naming
)

# Safety evaluation
safety_grader = DSPySafetyGrader(
    pred_field="generated_content"     # Works with any field
)

# All graders can be optimized with DSPy!
optimizer = dspy.BootstrapFewShot(metric=accuracy_grader.to_dspy_metric())
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

## 🎯 Flexible Field Extraction Examples

### The Key Innovation: No More Hardcoded Fields!

**Traditional DSPy metrics force specific field names:**
```python
# ❌ Traditional way - forces specific naming
example = dspy.Example(question="...", response="...")  # Must use these names
metric = SemanticF1()  # Hardcoded to expect 'question' and 'response'
```

**Our DSPy-optimizable graders work with ANY field names:**
```python
# ✅ New way - use your own field names!
from dspy_kit import create_dspy_customer_support_grader

# Your data with custom field names
example = {
    "customer_inquiry": "I can't access my account",
    "agent_reply": "I'll help you reset your password",
    "ideal_solution": "Guide user through password reset process"
}

# Grader adapts to YOUR field names
support_grader = create_dspy_customer_support_grader(
    response_field="agent_reply",       # Matches your data
    query_field="customer_inquiry",     # Matches your data
    reference_field="ideal_solution"    # Matches your data
)

# Works perfectly AND can be optimized!
class CustomerSupportAgent(dspy.Module):
    def forward(self, customer_inquiry):
        return dspy.Prediction(agent_reply=self.respond(customer_inquiry))

agent = CustomerSupportAgent()
evaluator = dspy.Evaluate(devset=data, metric=support_grader.to_dspy_metric())
score = evaluator(agent)

# The grader itself can be optimized too!
optimizer = dspy.BootstrapFewShot(metric=support_grader.to_dspy_metric())
```

### Multi-Domain Flexibility

```python
# Same grader logic, different field configurations
from dspy_kit import SemanticSimilarityGrader

# Research domain
research_grader = SemanticSimilarityGrader(
    pred_field="model_output",
    ideal_field="ground_truth_annotation"
)

# Production API domain
api_grader = SemanticSimilarityGrader(
    pred_field="api_response",
    ideal_field="expected_output"
)

# Customer support domain
support_grader = SemanticSimilarityGrader(
    pred_field="agent_message",
    ideal_field="ideal_response"
)

# All use the same optimizable logic, just different field names!
```

## ⚙️ Configuration-Driven Evaluation

Create reusable evaluation configurations:

```yaml
# eval_config.yaml
graders:
  accuracy:
    type: ExactMatchGrader
    pred: answer
    ideal: gold_answer

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
from dspy_kit import create_dspy_customer_support_grader

# Create grader that adapts to YOUR data format
support_grader = create_dspy_customer_support_grader(
    response_field="agent_response",     # Your field names
    query_field="customer_message",      # Your field names
    reference_field="ideal_response"     # Your field names
)

# Create your DSPy program
program = CustomerSupportAgent()

# The grader can be optimized alongside your program!
optimizer = dspy.BootstrapFewShot(metric=support_grader.to_dspy_metric())
optimized_program = optimizer.compile(program, trainset=training_data)

# Both program AND evaluation improved
evaluator = dspy.Evaluate(devset=eval_data, metric=support_grader.to_dspy_metric())
baseline_score = evaluator(program)
optimized_score = evaluator(optimized_program)

print(f"Total improvement: {optimized_score - baseline_score:.3f}")
```

## 📊 Evaluation Best Practices

### 1. **Start Simple, Iterate**
```python
# Begin with basic metrics
basic_grader = ExactMatchGrader()

# Gradually add DSPy-optimizable complexity
from dspy_kit import CompositeDSPyGrader, DSPyFactualAccuracyGrader, DSPySafetyGrader

advanced_grader = CompositeDSPyGrader({
    "accuracy": (DSPyFactualAccuracyGrader(pred_field="answer"), 0.5),
    "safety": (DSPySafetyGrader(pred_field="answer"), 0.3),
    "relevance": (DSPyRelevanceGrader(pred_field="answer"), 0.2)
})
```

### 2. **Use Trace-Aware Evaluation**
```python
def custom_metric(example, pred, trace=None):
    grader = DSPyFactualAccuracyGrader(
        pred_field="generated_answer",  # Flexible field names
        ideal_field="reference_answer"
    )

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
# For classification tasks with flexible fields
from dspy_kit import CompositeDSPyGrader, HelpfulnessGrader

classification_metric = CompositeDSPyGrader({
    "helpfulness": (HelpfulnessGrader(
        pred_field="predicted_intent",
        query_field="user_message"
    ), 0.7),
    "safety": (DSPySafetyGrader(pred_field="predicted_intent"), 0.3)
})

# For generation tasks with custom field names
generation_metric = CompositeDSPyGrader({
    "relevance": (DSPyRelevanceGrader(
        pred_field="generated_text",
        query_field="prompt"
    ), 0.3),
    "factuality": (DSPyFactualAccuracyGrader(
        pred_field="generated_text",
        ideal_field="reference_text"
    ), 0.4),
    "safety": (DSPySafetyGrader(pred_field="generated_text"), 0.3)
})
```

## 🧪 Testing and Validation

```python
# Test your graders before production use
def test_grader():
    grader = DSPyFactualAccuracyGrader(
        pred_field="model_response",    # Your field names
        ideal_field="expected_output"   # Your field names
    )

    # Test cases with your data format
    test_cases = [
        ({"expected_output": "Paris is the capital"},
         {"model_response": "Paris is France's capital"}, "Should score high"),
        ({"expected_output": "Paris is the capital"},
         {"model_response": "London is the capital"}, "Should score low"),
        ({}, {"model_response": "Some text"}, "Should handle missing fields")
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
- [i18n Guide](docs/i18n_guide.md)
- [Template System Guide](docs/template_guide.md)

## 🎯 Roadmap

- [x] **DSPy-Optimizable Graders**: LLM-as-a-Judge graders that can be optimized with DSPy
- [x] **Flexible Field Extraction**: No more hardcoded field requirements - works with any naming scheme
- [ ] **Additional Domains**: More specialized graders for code generation, summarization
- [ ] **Advanced Metrics**: BLEU, ROUGE, BERTScore integration with flexible fields
- [ ] **Auto-Field Detection**: Automatically detect common field patterns in datasets
- [ ] **Evaluation Dashboard**: Web UI for evaluation results and optimization tracking
- [ ] **Multi-modal Support**: Image and audio evaluation capabilities

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for the [DSPy](https://github.com/stanfordnlp/dspy) framework
- Follows [OpenAI's evaluation best practices](https://platform.openai.com/docs/guides/evals)
- Inspired by [Anthropic's evaluation methodology](https://docs.anthropic.com/en/docs/test-and-evaluate)
- Solves the fundamental trade-off between field flexibility and optimization capability in DSPy evaluation

## 💡 Key Innovation

**Before dspy-kit**: Choose between flexible field handling OR DSPy optimization
**With dspy-kit**: Get both flexible fields AND full DSPy optimization capabilities

This eliminates the need to restructure your data to match evaluation library expectations while maintaining all the benefits of DSPy's powerful optimization framework.

## 🚀 Future Plans

- [x] **Revolutionary Grader Design**: Combines DSPy optimization with flexible field extraction
- [ ] **Synthetic Data Generation**: Constitutional AI and alignment-based data generation
- [ ] **Red Teaming**: Adversarial testing and safety evaluation
- [ ] **Schema Validation**: Optional validation that required fields exist
- [ ] **Performance Optimization**: Caching and batch processing improvements
- [ ] **More Model Providers**: Support for Anthropic, Cohere, local models

## 🔗 Related Projects

- [DSPy](https://github.com/stanfordnlp/dspy) - The framework this library is built for
- [OpenAI Evals](https://github.com/openai/evals) - OpenAI's evaluation framework
- [LangChain Evaluation](https://python.langchain.com/docs/guides/evaluation/) - LangChain's evaluation tools

---

**Made with ❤️ for the DSPy community**

## Docs

#### Overview

```
/Users/yigao/Developer/Personal/dspy-kit/
├── demo_overview.py                    # Complete system demo
├── demo_template_inheritance.py        # Inheritance demo
├── demo_tool_integration.py            # Tool integration demo
├── demo_i18n_templates.py              # i18n system demo
```

Test Files (Validate Implementation)
```
├── test_migration.py                   # Migration testing
├── test_inheritance.py                 # Inheritance testing
├── test_query_info_migration.py        # Real prompt migration
├── test_rendered_output.py             # Output validation
├── test_i18n_system.py                # i18n testing
```

#### Example Files (Usage Patterns)
```
├── examples/
│   ├── chinese_chat_adapter_comparison.py
│   ├── simple_chinese_adapter.py
│   ├── i18n_chinese_ecommerce.py
│   ├── complex_routing_template_demo.py
│   └── simple_routing_usage.py
```

#### Template Files (YAML Examples)
```
├── templates/
│   ├── query_item_info_template.yaml   # Migrated template
│   ├── intent_classification_template.yaml
│   ├── ecommerce_support_with_tools.yaml
│   └── examples/
│       └── customer_support_router.yaml
```

---

## Key Files to Review First

1. **Core Implementation**
   - `/dspy-kit/dspy_kit/templates/core/template.py` - The heart of the system
   - `/dspy-kit/dspy_kit/templates/adapters/dspy_adapter.py` - DSPy integration

2. **Critical Features**
   - `/dspy-kit/dspy_kit/templates/utils/migrator.py` - Migration logic
   - `/dspy-kit/dspy_kit/templates/i18n/adapter.py` - i18n system

3. **Integration Tests**
   - `/dspy-kit/final_demo.py` - Shows everything working together
   - `/dspy-kit/test_query_info_migration.py` - Real-world validation
