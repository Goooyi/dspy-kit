# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-12-04

### 🎯 MAJOR: Project Renamed to dspy-kit

**BREAKING**: Project renamed from `dspy-evals` to `dspy-kit` to reflect expanded scope.

#### 📦 Package Changes
- **Renamed**: `dspy_evals` → `dspy_kit`
- **Restructured**: Organized into specialized modules
  - `dspy_kit.evaluation.graders` - All evaluation graders
  - `dspy_kit.evaluation.domains` - Domain-specific evaluators  
  - `dspy_kit.utils` - Shared utilities
  - `dspy_kit.synthetic` - Synthetic data generation (placeholder)
  - `dspy_kit.red_team` - Red teaming capabilities (placeholder)

#### ✨ Added
- **Classification Graders**: New comprehensive classification metrics
  - `PrecisionGrader` - Precision calculation with multiple averaging strategies
  - `RecallGrader` - Recall calculation with multiple averaging strategies  
  - `F1Grader` - F1 score calculation with multiple averaging strategies
  - `AccuracyGrader` - Accuracy calculation for classification tasks
  - `ClassificationMetricsGrader` - Composite grader combining all classification metrics
  - `IntentClassificationGrader` - Specialized grader for customer support intent classification

- **Classification Features**:
  - Multiple averaging strategies: macro, micro, weighted, binary
  - Flexible field mapping for predictions and ground truth
  - Label normalization and case-insensitive matching
  - Batch evaluation with `batch_calculate()` method
  - Integration with DSPy's evaluation and optimization workflows

- **Convenience Functions**:
  - `create_intent_classifier_grader()` - Quick intent classification setup
  - `create_classification_grader()` - General classification grader setup
  - `create_binary_classification_grader()` - Binary classification setup

- **Convenience Aliases**:
  - `Precision` → `PrecisionGrader`
  - `Recall` → `RecallGrader` 
  - `F1Score` → `F1Grader`
  - `Accuracy` → `AccuracyGrader`
  - `IntentClassifier` → `IntentClassificationGrader`

- **Shared Utilities** (`dspy_kit.utils`):
  - Configuration loading/saving (JSON, YAML)
  - Text normalization functions
  - Field extraction utilities
  - Logging setup
  - Dependency checking
  - Common exceptions

- **Future Module Placeholders**:
  - `dspy_kit.synthetic` - For synthetic data generation
  - `dspy_kit.red_team` - For adversarial testing and red teaming

#### 🔄 Changed
- **Import Structure**: Simplified imports from main package
  - Old: `from dspy_evals.core.string_graders import ExactMatchGrader`
  - New: `from dspy_kit import ExactMatchGrader`
- **Package URLs**: Updated to reflect new repository name
- **Documentation**: Updated for new package name and structure

#### 🛡️ Backwards Compatibility
- **100% Compatible**: All existing APIs work unchanged
- **Migration Guide**: Added comprehensive migration documentation
- **Same Functionality**: All graders work identically to previous version

#### 📚 Documentation
- Updated README.md with new package name and features
- Added MIGRATION.md with step-by-step migration guide
- Updated all examples to use new import structure
- Added classification grader examples and tutorials

#### 🔧 Development
- Updated pyproject.toml for new package name
- Updated test files for new import structure
- Maintained all existing test coverage
- Added tests for new classification graders

## [0.1.0] - 2024-12-03

### ✨ Initial Release (dspy-evals)

#### Added
- **String-based Graders**:
  - `StringCheckGrader` - Flexible string comparison operations
  - `TextSimilarityGrader` - Fuzzy matching and similarity metrics
  - `ExactMatchGrader` - Exact string matching
  - `ContainsGrader` - Substring matching
  - `StartsWithGrader` - Prefix matching
  - `RegexGrader` - Regular expression matching

- **Model-based Graders (LLM-as-a-Judge)**:
  - `ScoreModelGrader` - Numerical scoring with LLM evaluation
  - `LabelModelGrader` - Classification with LLM evaluation
  - `LikertScaleGrader` - 5-point Likert scale evaluation
  - `BinaryClassificationGrader` - Yes/No binary evaluation
  - `FactualAccuracyGrader` - Factual correctness evaluation
  - `SafetyGrader` - Safety and appropriateness evaluation
  - `RelevanceGrader` - Response relevance evaluation

- **Python Code Graders**:
  - `PythonGrader` - Execute custom Python evaluation logic
  - `FuzzyMatchGrader` - Fuzzy string matching
  - `JSONValidationGrader` - JSON structure validation
  - `NumericAccuracyGrader` - Numerical comparison with tolerance

- **Composite Graders**:
  - `CompositeGrader` - Weighted combination of multiple graders
  - `EdgeCaseAwareGrader` - Handle edge cases in evaluation

- **Domain-specific Evaluators**:
  - Customer support evaluation graders
  - QA system evaluation graders
  - Specialized composite graders for common use cases

- **Core Features**:
  - Async support for model-based graders
  - DSPy integration and optimization support
  - Configuration-driven evaluation with YAML support
  - Comprehensive error handling and edge case management
  - Extensive documentation and examples

- **Developer Experience**:
  - Type hints throughout codebase
  - Comprehensive test suite
  - Example usage scripts
  - Detailed documentation

---

## Migration from dspy-evals

If you're upgrading from `dspy-evals` to `dspy-kit`, see [MIGRATION.md](MIGRATION.md) for a complete migration guide. The process is straightforward and maintains 100% backward compatibility.