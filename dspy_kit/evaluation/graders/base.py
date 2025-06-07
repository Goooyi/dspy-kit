"""Base grader interface with async support and DSPy integration."""

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional, Union

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None


class BaseGrader(ABC):
    """
    Base class for all DSPy graders following OpenAI and Anthropic best practices.

    Graders evaluate the quality of DSPy program outputs and return scores
    that can be used for optimization and evaluation.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        self.name = name or self.__class__.__name__
        self.config = kwargs
        self._cache = {}

    @abstractmethod
    def __call__(
        self, example: Any, pred: Any, trace: Optional[Any] = None
    ) -> Union[float, bool]:
        """
        Evaluate a prediction against an example.

        Args:
            example: The ground truth example from your dataset
            pred: The prediction from your DSPy program
            trace: Optional trace from DSPy optimization (for intermediate step validation)

        Returns:
            Score as float (0.0-1.0) for evaluation/optimization,
            or bool for strict pass/fail during bootstrapping
        """
        pass

    async def acall(
        self, example: Any, pred: Any, trace: Optional[Any] = None
    ) -> Union[float, bool]:
        """
        Async version of __call__. Override this for model-based graders.
        Default implementation calls the sync version.
        """
        return self(example, pred, trace)

    def to_dspy_metric(self) -> Callable:
        """Convert this grader to a DSPy-compatible metric function."""

        def metric(example, pred, trace=None):
            return self(example, pred, trace)

        # Preserve function name for debugging
        metric.__name__ = f"{self.name}_metric"
        return metric

    def to_async_dspy_metric(self) -> Callable:
        """Convert this grader to an async DSPy-compatible metric function."""

        async def async_metric(example, pred, trace=None):
            return await self.acall(example, pred, trace)

        async_metric.__name__ = f"{self.name}_async_metric"
        return async_metric

    async def batch_evaluate(
        self,
        examples: list[Any],
        predictions: list[Any],
        traces: Optional[list[Any]] = None,
        max_concurrent: int = 10,
    ) -> list[Union[float, bool]]:
        """
        Batch evaluation with concurrency control.

        Args:
            examples: List of ground truth examples
            predictions: List of predictions from DSPy program
            traces: Optional list of traces from optimization
            max_concurrent: Maximum number of concurrent evaluations

        Returns:
            List of scores
        """
        if traces is None:
            traces = [None] * len(examples)

        semaphore = asyncio.Semaphore(max_concurrent)

        async def evaluate_with_semaphore(example, pred, trace):
            async with semaphore:
                return await self.acall(example, pred, trace)

        tasks = [
            evaluate_with_semaphore(example, pred, trace)
            for example, pred, trace in zip(examples, predictions, traces)
        ]

        return await asyncio.gather(*tasks)

    @classmethod
    def from_config(cls, config_path: Union[str, Path, dict]) -> "BaseGrader":
        """
        Create grader from configuration file or dict.

        Args:
            config_path: Path to YAML config file or config dict

        Returns:
            Configured grader instance
        """
        if isinstance(config_path, (str, Path)):
            if not YAML_AVAILABLE:
                raise ImportError(
                    "PyYAML is required for loading YAML config files"
                )
            with open(config_path) as f:
                config = yaml.safe_load(f)  # type:ignore
        else:
            config = config_path

        grader_type = config.get("type", cls.__name__)
        grader_config = config.get("config", {})

        # Import and instantiate the grader class
        if grader_type == cls.__name__:
            return cls(**grader_config)
        else:
            # Dynamic import for other grader types
            from . import dspy_model_graders, python_graders, string_graders

            for module in [string_graders, dspy_model_graders, python_graders]:
                if hasattr(module, grader_type):
                    grader_class = getattr(module, grader_type)
                    return grader_class(**grader_config)

            raise ValueError(f"Unknown grader type: {grader_type}")

    def extract_field(self, obj: Any, field: str, default: str = "") -> str:
        """
        Extract field from object, handling various formats.

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

    def validate_trace(
        self, trace: Any, step_validators: Optional[dict[str, Callable]] = None
    ) -> bool:
        """
        Validate intermediate steps in trace for optimization.

        Args:
            trace: DSPy trace object
            step_validators: Dict mapping step names to validation functions

        Returns:
            True if all validations pass
        """
        if trace is None or step_validators is None:
            return True

        try:
            # Extract trace steps (this depends on DSPy's trace format)
            for validator in step_validators.values():
                if not validator(trace):
                    return False
            return True
        except Exception:
            return False


class CompositeGrader(BaseGrader):
    """
    Combines multiple graders into a single score following OpenAI's multigrader pattern.
    see: https://platform.openai.com/docs/api-reference/graders/multi

    Example:
        composite = CompositeGrader({
            "accuracy": (ExactMatchGrader(), 0.4),
            "quality": (ScoreModelGrader(), 0.4),
            "safety": (BinaryGrader(), 0.2)
        })
    """

    def __init__(
        self,
        graders: dict[str, tuple[BaseGrader, float]],
        aggregation_fn: Optional[Callable] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.graders = graders
        self.aggregation_fn = aggregation_fn or self._weighted_average

        # Validate weights sum to 1
        total_weight = sum(weight for _, weight in graders.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    def __call__(
        self, example: Any, pred: Any, trace: Optional[Any] = None
    ) -> Union[float, bool]:
        scores = {}

        for name, (grader, weight) in self.graders.items():
            score = grader(example, pred, trace)
            scores[name] = (score, weight)

        return self.aggregation_fn(scores, trace)

    async def acall(
        self, example: Any, pred: Any, trace: Optional[Any] = None
    ) -> Union[float, bool]:
        scores = {}

        for name, (grader, weight) in self.graders.items():
            score = await grader.acall(example, pred, trace)
            scores[name] = (score, weight)

        return self.aggregation_fn(scores, trace)

    def _weighted_average(
        self, scores: dict[str, tuple[float, float]], trace: Optional[Any]
    ) -> Union[float, bool]:
        """Default aggregation: weighted average."""
        if trace is None:  # Evaluation mode
            return sum(score * weight for score, weight in scores.values())
        else:  # Optimization mode - all must pass threshold
            return all(score > 0.7 for score, _ in scores.values())


class EdgeCaseAwareGrader(BaseGrader):
    """
    Wrapper grader that handles edge cases before delegating to base grader.
    Implements OpenAI's edge case handling recommendations.
    see: https://platform.openai.com/docs/guides/evals-design#handle-edge-cases
    """

    def __init__(
        self,
        base_grader: BaseGrader,
        edge_case_handlers: dict[str, Callable],
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.base_grader = base_grader
        self.edge_case_handlers = edge_case_handlers

    def __call__(
        self, example: Any, pred: Any, trace: Optional[Any] = None
    ) -> Union[float, bool]:
        # Check for edge cases first
        for case_name, handler in self.edge_case_handlers.items():
            if handler(example, pred):
                return self._handle_edge_case(case_name, example, pred, trace)

        # No edge case detected, use base grader
        return self.base_grader(example, pred, trace)

    async def acall(
        self, example: Any, pred: Any, trace: Optional[Any] = None
    ) -> Union[float, bool]:
        # Check for edge cases first
        for case_name, handler in self.edge_case_handlers.items():
            if handler(example, pred):
                return self._handle_edge_case(case_name, example, pred, trace)

        # No edge case detected, use base grader
        return await self.base_grader.acall(example, pred, trace)

    def _handle_edge_case(
        self, case_name: str, example: Any, pred: Any, trace: Optional[Any]
    ) -> Union[float, bool]:
        """Handle specific edge case. Override for custom behavior."""
        if case_name == "out_of_scope":
            return 0.0 if trace is None else False
        elif case_name == "abusive_input":
            return 0.0 if trace is None else False
        elif case_name == "malformed_output":
            return 0.0 if trace is None else False
        else:
            return 0.0 if trace is None else False


def dspy_metric(grader_class: type) -> type:
    """
    Decorator to make a grader class easily usable as a DSPy metric.

    Usage:
        @dspy_metric
        class MyMetric(ScoreModelGrader):
            prompt_template = "Rate this response 1-5..."
            model = "gpt-4"

        # Can now be used directly as:
        metric = MyMetric()
        evaluator = Evaluate(devset, metric=metric)
    """
    original_init = grader_class.__init__

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Make the instance callable as a DSPy metric
        self.__call__ = self.to_dspy_metric()

    grader_class.__init__ = new_init
    return grader_class


class ConfigurableGrader(BaseGrader):
    """
    Base class for graders that can be easily configured via YAML or dict.
    """

    # Default configuration - override in subclasses
    DEFAULT_CONFIG = {}

    def __init__(self, name: Optional[str] = None, **kwargs):
        # Merge with default config
        config = {**self.DEFAULT_CONFIG, **kwargs}
        super().__init__(name=name, **config)

        # Set all config items as attributes
        for key, value in config.items():
            if key != "name": # 'name' is handled by BaseGrader
                setattr(self, key, value)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "ConfigurableGrader":
        """Create grader from YAML configuration."""
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for loading YAML config files"
            )
        with open(yaml_path) as f:
            config = yaml.safe_load(f)  # type:ignore
        return cls(**config)
