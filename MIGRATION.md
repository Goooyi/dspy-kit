# Migration Guide: dspy-evals → dspy-kit

This guide helps you migrate from `dspy-evals` to `dspy-kit`. The project has been renamed and restructured to support a broader toolkit including evaluation, synthetic data generation, and red teaming capabilities.

## 🔄 What Changed

### Project Rename
- **Old**: `dspy-evals` 
- **New**: `dspy-kit`

### Package Structure
```
Old:                    New:
dspy_evals/            dspy_kit/
├── core/              ├── evaluation/
│   ├── base.py        │   ├── graders/
│   ├── string_graders.py   │   │   ├── base.py
│   ├── model_graders.py    │   │   ├── string_graders.py
│   └── python_graders.py   │   │   ├── model_graders.py
└── domains/           │   │   ├── python_graders.py
                       │   │   └── classification_graders.py (NEW)
                       │   └── domains/
                       ├── synthetic/ (NEW - Coming Soon)
                       ├── red_team/ (NEW - Coming Soon)
                       └── utils/ (NEW)
```

### New Features Added
- **Classification Graders**: Precision, Recall, F1, Accuracy metrics
- **Intent Classification**: Specialized graders for customer support
- **Shared Utilities**: Common functions across modules
- **Future Modules**: Placeholders for synthetic data and red teaming

## 🚀 Quick Migration

### 1. Update Installation

```bash
# Uninstall old package
pip uninstall dspy-evals

# Install new package
pip install dspy-kit
```

### 2. Update Import Statements

**Simple Find & Replace:**
- Find: `from dspy_evals`
- Replace: `from dspy_kit`

**Example:**
```python
# Old
from dspy_evals import ExactMatchGrader, ScoreModelGrader
from dspy_evals.core.string_graders import TextSimilarityGrader

# New  
from dspy_kit import ExactMatchGrader, ScoreModelGrader, TextSimilarityGrader
```

## 📋 Detailed Migration Steps

### Step 1: Update Requirements Files

**requirements.txt / pyproject.toml:**
```diff
- dspy-evals>=0.1.0
+ dspy-kit>=0.1.0
```

### Step 2: Update Import Statements

All imports now use the simplified `dspy_kit` namespace:

```python
# ✅ New unified imports
from dspy_kit import (
    # String graders
    ExactMatchGrader, 
    TextSimilarityGrader,
    StringCheckGrader,
    
    # Model graders  
    ScoreModelGrader,
    LabelModelGrader,
    SafetyGrader,
    
    # Classification graders (NEW!)
    PrecisionGrader,
    RecallGrader, 
    F1Grader,
    AccuracyGrader,
    IntentClassificationGrader,
    
    # Composite graders
    CompositeGrader,
    EdgeCaseAwareGrader,
    
    # Convenience functions
    create_exact_match,
    create_intent_classifier_grader,
)
```

### Step 3: Update Domain-Specific Imports

```python
# Old
from dspy_evals.domains.customer_support import CustomerSupportCompositeGrader

# New - now available from main package
from dspy_kit import CustomerSupportCompositeGrader
```

### Step 4: Take Advantage of New Features

#### Classification Metrics for Intent Classification:
```python
from dspy_kit import create_intent_classifier_grader

# Create specialized intent classification grader
grader = create_intent_classifier_grader(
    intents=["billing", "technical", "general"],
    average="weighted"
)

# Use with DSPy
metric = grader.to_dspy_metric()
evaluator = dspy.Evaluate(devset=dataset, metric=metric)
```

#### Individual Classification Metrics:
```python
from dspy_kit import F1Grader, PrecisionGrader, RecallGrader

# Precision for intent classification
precision_grader = PrecisionGrader(
    predicted_field="intent",
    true_field="expected_intent",
    average="macro"
)

# Batch evaluation
score = precision_grader.batch_calculate(examples, predictions)
```

### Step 5: Update Configuration Files

If you have YAML configuration files:

```yaml
# Old
graders:
  accuracy:
    type: ExactMatchGrader
    input_field: answer
    reference_field: gold_answer

# New - same format, but can now include classification graders
graders:
  accuracy:
    type: ExactMatchGrader
    input_field: answer
    reference_field: gold_answer
    
  intent_f1:
    type: F1Grader
    predicted_field: intent
    true_field: expected_intent
    average: weighted
```

## 🔧 Breaking Changes

### None! 
The migration is designed to be **100% backward compatible**. All existing functionality works exactly the same.

### What Stays the Same:
- ✅ All grader APIs unchanged
- ✅ All function signatures unchanged  
- ✅ All configuration formats unchanged
- ✅ All DSPy integration unchanged

### What's New:
- ✅ Classification graders (Precision, Recall, F1, Accuracy)
- ✅ Intent classification graders
- ✅ Simplified import structure
- ✅ Shared utilities
- ✅ Future-ready for synthetic data & red teaming

## 📚 Updated Examples

### Basic Usage (Unchanged)
```python
import dspy
from dspy_kit import ExactMatchGrader, ScoreModelGrader, CompositeGrader

# Everything works exactly the same!
accuracy_grader = ExactMatchGrader(
    input_field="answer",
    reference_field="gold_answer"
)

quality_grader = ScoreModelGrader(
    prompt_template="Rate this response (1-5): {{sample.output_text}}",
    range=[1, 5]
)

composite = CompositeGrader({
    "accuracy": (accuracy_grader, 0.6),
    "quality": (quality_grader, 0.4)
})
```

### New Classification Example
```python
from dspy_kit import create_intent_classifier_grader

# NEW: Intent classification evaluation  
intent_grader = create_intent_classifier_grader(
    intents=["billing", "technical_support", "cancellation"],
    average="weighted"
)

# Use with your intent classifier
metric = intent_grader.to_dspy_metric()
evaluator = dspy.Evaluate(devset=intent_data, metric=metric)
score = evaluator(intent_classifier)
```

## 🚨 Common Migration Issues

### Issue 1: Import Errors
```python
# ❌ This will fail
from dspy_evals.core.base import BaseGrader

# ✅ Use this instead  
from dspy_kit import BaseGrader
```

### Issue 2: Relative Imports
```python
# ❌ This will fail
from dspy_evals.core.string_graders import ExactMatchGrader

# ✅ Use this instead
from dspy_kit import ExactMatchGrader
```

### Issue 3: Domain Imports
```python
# ❌ This will fail
from dspy_evals.domains.customer_support import IntentAccuracyGrader

# ✅ Use this instead
from dspy_kit import IntentAccuracyGrader
```

## 🔮 Future Features Preview

The expanded `dspy-kit` will include:

### Synthetic Data Generation (Coming Soon)
```python
# Future functionality
from dspy_kit.synthetic import ConstitutionalDataGenerator

generator = ConstitutionalDataGenerator(
    principles=["helpful", "harmless", "honest"]
)
synthetic_data = generator.generate_qa_pairs(topics=["customer_support"])
```

### Red Teaming (Coming Soon)  
```python
# Future functionality
from dspy_kit.red_team import AdversarialGenerator

red_team = AdversarialGenerator()
adversarial_prompts = red_team.generate_jailbreaks(target_model=my_model)
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Goooyi/dspy-kit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Goooyi/dspy-kit/discussions)
- **Documentation**: [README](https://github.com/Goooyi/dspy-kit#readme)

## ✅ Migration Checklist

- [ ] Update package installation (`pip install dspy-kit`)
- [ ] Update import statements (`dspy_evals` → `dspy_kit`)
- [ ] Test existing functionality still works
- [ ] Consider using new classification graders
- [ ] Update documentation/README references
- [ ] Update CI/CD configurations
- [ ] Update requirements files

---

**Ready to migrate? The process should take less than 5 minutes for most projects!** 🚀