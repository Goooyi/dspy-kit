"""Python-based graders for custom evaluation logic following OpenAI's pattern."""

import ast
import importlib
import inspect
from typing import Any, Callable, Optional, Union

from .base import BaseGrader, ConfigurableGrader


class PythonGrader(BaseGrader):
    """
    Python code grader following OpenAI's pattern.
    Executes arbitrary Python code with a grade function that takes (sample, item) -> float.
    """

    def __init__(
        self,
        source_code: str,
        allowed_imports: Optional[list[str]] = None,
        timeout: float = 30.0,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.source_code = source_code
        self.timeout = timeout
        self.allowed_imports = allowed_imports or [
            "math",
            "re",
            "json",
            "statistics",
            "collections",
            "rapidfuzz",
            "numpy",
            "scipy",
            "sklearn",
            "rouge_score",
        ]

        # Compile and validate the code
        self.grade_function = self._compile_grade_function()

    def _compile_grade_function(self) -> Callable:
        """Compile the source code and extract the grade function."""
        try:
            # Parse AST to validate syntax
            ast.parse(self.source_code)

            # Create a restricted execution environment
            exec_globals = self._create_safe_globals()

            # Execute the code
            exec(self.source_code, exec_globals)

            # Extract the grade function
            if "grade" not in exec_globals:
                raise ValueError("Python grader must define a 'grade' function")

            grade_func = exec_globals["grade"]

            # Validate function signature
            sig = inspect.signature(grade_func)
            if len(sig.parameters) != 2:
                raise ValueError("Grade function must take exactly 2 parameters: (sample, item)")

            return grade_func

        except Exception as e:
            raise ValueError(f"Failed to compile grade function: {e}") from e

    def _create_safe_globals(self) -> dict[str, Any]:
        """Create a restricted global namespace for code execution."""
        safe_globals: dict[str, Any] = {
            "__builtins__": {
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "reversed": reversed,
                "enumerate": enumerate,
                "zip": zip,
                "range": range,
                "isinstance": isinstance,
                "hasattr": hasattr,
                "getattr": getattr,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "Exception": Exception,
                "print": print,
            }
        }

        # Add allowed imports
        for module_name in self.allowed_imports:
            try:
                module = importlib.import_module(module_name)
                safe_globals[module_name] = module
            except ImportError:
                # Module not available, skip
                pass

        return safe_globals

    def __call__(self, example: Any, pred: Any, trace: Optional[Any] = None) -> Union[float, bool]:
        """Execute the grade function."""
        try:
            # Prepare sample and item dictionaries
            sample = self._prepare_sample(pred)
            item = self._prepare_item(example)

            # Execute the grade function
            result = self.grade_function(sample, item)

            # Validate result
            if not isinstance(result, (int, float)):
                return 0.0

            result = float(result)

            # Clamp to valid range
            result = max(0.0, min(1.0, result))

            if trace is None:  # Evaluation mode
                return result
            else:  # Optimization mode
                return result >= 0.7  # Default threshold

        except Exception as e:
            print(f"Python grader execution failed: {e}")
            return 0.0 if trace is None else False

    def _prepare_sample(self, pred: Any) -> dict[str, Any]:
        """Prepare sample dictionary from prediction."""
        sample = {
            "output_text": self.extract_field(pred, "output", ""),
            "output_json": pred if isinstance(pred, dict) else {},
        }

        # Add all fields from pred if it's an object
        if hasattr(pred, "__dict__"):
            sample.update(pred.__dict__)
        elif isinstance(pred, dict):
            sample.update(pred)

        return sample

    def _prepare_item(self, example: Any) -> dict[str, Any]:
        """Prepare item dictionary from example."""
        item = {
            "reference_answer": self.extract_field(example, "answer", ""),
            "question": self.extract_field(example, "question", ""),
        }

        # Add all fields from example if it's an object
        if hasattr(example, "__dict__"):
            item.update(example.__dict__)
        elif isinstance(example, dict):
            item.update(example)

        return item


class FuzzyMatchGrader(PythonGrader):
    """Pre-built fuzzy matching grader using rapidfuzz."""

    def __init__(self, threshold: float = 0.8, **kwargs):
        source_code = """
from rapidfuzz import fuzz, utils

def grade(sample, item):
    output_text = sample.get("output_text", "")
    reference_answer = item.get("reference_answer", "")

    if not output_text or not reference_answer:
        return 0.0

    score = fuzz.WRatio(output_text, reference_answer, processor=utils.default_process) / 100.0
    return score
"""
        super().__init__(source_code, **kwargs)
        self.threshold = threshold


class RegexMatchGrader(PythonGrader):
    """Pre-built regex pattern matching grader."""

    def __init__(self, pattern: str, flags: int = 0, **kwargs):
        source_code = f"""
import re

def grade(sample, item):
    output_text = sample.get("output_text", "")
    pattern = r"{pattern}"
    flags = {flags}

    match = re.search(pattern, output_text, flags)
    return 1.0 if match else 0.0
"""
        super().__init__(source_code, **kwargs)


class JSONValidationGrader(PythonGrader):
    """Pre-built JSON structure validation grader."""

    def __init__(self, required_fields: Optional[list[str]] = None, **kwargs):
        required_fields = required_fields or []
        fields_str = str(required_fields)

        source_code = f"""
import json

def grade(sample, item):
    output_text = sample.get("output_text", "")
    required_fields = {fields_str}

    try:
        data = json.loads(output_text)

        if not isinstance(data, dict):
            return 0.0

        # Check required fields
        for field in required_fields:
            if field not in data:
                return 0.0

        return 1.0

    except json.JSONDecodeError:
        return 0.0
"""
        super().__init__(source_code, **kwargs)


class NumericAccuracyGrader(PythonGrader):
    """Pre-built numeric accuracy grader with tolerance."""

    def __init__(self, tolerance: float = 1e-6, relative: bool = False, **kwargs):
        source_code = f"""
import re

def grade(sample, item):
    output_text = sample.get("output_text", "")
    reference_answer = item.get("reference_answer", "")
    tolerance = {tolerance}
    relative = {relative}

    # Extract numbers from both texts
    output_numbers = re.findall(r'-?\\d+\\.?\\d*', output_text)
    reference_numbers = re.findall(r'-?\\d+\\.?\\d*', reference_answer)

    if not output_numbers or not reference_numbers:
        return 0.0

    try:
        output_val = float(output_numbers[0])
        reference_val = float(reference_numbers[0])

        if relative:
            # Relative tolerance
            if reference_val == 0:
                return 1.0 if output_val == 0 else 0.0
            diff = abs(output_val - reference_val) / abs(reference_val)
        else:
            # Absolute tolerance
            diff = abs(output_val - reference_val)

        return 1.0 if diff <= tolerance else 0.0

    except ValueError:
        return 0.0
"""
        super().__init__(source_code, **kwargs)


class ListComparisonGrader(PythonGrader):
    """Pre-built list comparison grader (order-sensitive or not)."""

    def __init__(self, order_sensitive: bool = True, **kwargs):
        source_code = f"""
import json
import re

def grade(sample, item):
    output_text = sample.get("output_text", "")
    reference_answer = item.get("reference_answer", "")
    order_sensitive = {order_sensitive}

    def extract_list(text):
        # Try JSON parsing first
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except:
            pass

        # Try comma-separated values
        if ',' in text:
            return [item.strip() for item in text.split(',')]

        # Try space-separated
        return text.split()

    output_list = extract_list(output_text)
    reference_list = extract_list(reference_answer)

    if not output_list or not reference_list:
        return 0.0

    if order_sensitive:
        return 1.0 if output_list == reference_list else 0.0
    else:
        return 1.0 if set(output_list) == set(reference_list) else 0.0
"""
        super().__init__(source_code, **kwargs)


class SQLExecutionGrader(PythonGrader):
    """Pre-built SQL execution grader (syntax checking)."""

    def __init__(self, **kwargs):
        source_code = """
import re

def grade(sample, item):
    output_text = sample.get("output_text", "")

    # Basic SQL syntax validation
    output_text = output_text.strip()

    if not output_text:
        return 0.0

    # Check for basic SQL keywords
    sql_keywords = ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE']
    has_keyword = any(keyword in output_text.upper() for keyword in sql_keywords)

    if not has_keyword:
        return 0.0

    # Check for balanced parentheses
    open_count = output_text.count('(')
    close_count = output_text.count(')')

    if open_count != close_count:
        return 0.5  # Partial credit for syntax issues

    # Check for semicolon ending (optional)
    if output_text.rstrip().endswith(';'):
        return 1.0
    else:
        return 0.8  # Good but missing semicolon
"""
        super().__init__(source_code, **kwargs)


class CustomMetricGrader(ConfigurableGrader, PythonGrader):
    """Configurable Python grader that can be defined via config."""

    DEFAULT_CONFIG = {
        "source_code": """
def grade(sample, item):
    # Default implementation: exact match
    output_text = sample.get("output_text", "").strip().lower()
    reference_answer = item.get("reference_answer", "").strip().lower()
    return 1.0 if output_text == reference_answer else 0.0
""",
        "allowed_imports": ["math", "re", "json", "rapidfuzz", "numpy"],
        "timeout": 30.0,
    }


# Utility functions for building custom graders
def create_python_grader_from_file(file_path: str, **kwargs) -> PythonGrader:
    """Create a Python grader from a source file."""
    with open(file_path) as f:
        source_code = f.read()
    return PythonGrader(source_code, **kwargs)


def create_lambda_grader(func: Callable[[dict, dict], float], **kwargs) -> PythonGrader:
    """Create a Python grader from a lambda function."""
    # Get the function source (limited support)
    try:
        source = inspect.getsource(func)
        # Extract the lambda body
        lambda_body = source.split("lambda")[1].split(":")[1].strip()

        source_code = f"""
def grade(sample, item):
    return {lambda_body}
"""
        return PythonGrader(source_code, **kwargs)
    except Exception as e:
        raise ValueError("Cannot extract source from lambda function") from e


# Common utility functions for use in custom graders
UTILITY_FUNCTIONS = """
# Common utility functions for custom graders

def extract_numbers(text):
    import re
    return [float(x) for x in re.findall(r'-?\\d+\\.?\\d*', text)]

def normalize_text(text):
    import re
    return re.sub(r'\\s+', ' ', text.lower().strip())

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def contains_all_keywords(text, keywords):
    text_lower = text.lower()
    return all(keyword.lower() in text_lower for keyword in keywords)

def extract_json_field(text, field):
    import json
    try:
        data = json.loads(text)
        return data.get(field, "")
    except:
        return ""
"""
