"""
Template-aware evaluation graders.

These graders integrate with dspy-kit's evaluation system to provide
template-specific evaluation capabilities including format compliance,
factual recall, and domain-specific quality assessment.
"""

import json
import re
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

# Import dspy-kit evaluation components
from dspy_kit.evaluation.graders.base import ConfigurableGrader, CompositeGrader
from dspy_kit.evaluation.graders.dspy_model_graders import BaseDSPyGrader
from dspy_kit.evaluation.graders.python_graders import JSONValidationGrader

# Import our template system
from ..core.template import PromptTemplate

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


@dataclass
class TemplateEvaluationResult:
    """Result of template-aware evaluation."""
    score: float
    passed: bool
    details: Dict[str, Any]
    errors: List[str] = None


class TemplateFormatComplianceGrader(ConfigurableGrader):
    """
    Evaluates whether the output complies with the template's output_schema.
    
    This grader validates:
    - JSON structure matches output_schema
    - Required fields are present
    - Field types are correct
    - Value constraints are satisfied
    """
    
    def __init__(self, template: PromptTemplate, **kwargs):
        """
        Initialize the format compliance grader.
        
        Args:
            template: PromptTemplate with output_schema to validate against
        """
        super().__init__(**kwargs)
        self.template = template
        self.output_schema = template.output_schema
        self.required_fields = {
            name: config for name, config in self.output_schema.items()
            if config.get("required", True)
        }
        
    def __call__(self, example, pred, trace=None) -> Union[float, bool]:
        """
        Evaluate format compliance.
        
        Args:
            example: Input example
            pred: Model prediction to evaluate
            trace: Optional trace information
            
        Returns:
            float: Compliance score (0.0-1.0) in evaluation mode
            bool: Pass/fail in optimization mode
        """
        result = self._evaluate_format_compliance(example, pred, trace)
        
        if self.mode == "evaluation":
            return result.score
        else:
            return result.passed
    
    def _evaluate_format_compliance(self, example, pred, trace=None) -> TemplateEvaluationResult:
        """Internal method to evaluate format compliance."""
        errors = []
        details = {}
        score = 1.0
        
        # Extract response from prediction
        response = self._extract_response(pred)
        if not response:
            return TemplateEvaluationResult(
                score=0.0,
                passed=False,
                details={"error": "No response found"},
                errors=["Failed to extract response from prediction"]
            )
        
        # Try to parse as JSON if expected
        if self._expects_json_format():
            try:
                parsed_response = json.loads(response)
                details["parsed_json"] = True
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON format: {e}")
                score *= 0.5
                parsed_response = {"raw_response": response}
        else:
            parsed_response = {"raw_response": response}
        
        # Validate required fields
        missing_fields = []
        for field_name, field_config in self.required_fields.items():
            if field_name not in parsed_response:
                missing_fields.append(field_name)
        
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
            score *= max(0.1, 1.0 - len(missing_fields) / len(self.required_fields))
        
        # Validate field types
        type_errors = []
        for field_name, field_config in self.output_schema.items():
            if field_name in parsed_response:
                expected_type = field_config.get("type", "string")
                actual_value = parsed_response[field_name]
                
                if not self._validate_field_type(actual_value, expected_type):
                    type_errors.append(f"{field_name}: expected {expected_type}, got {type(actual_value).__name__}")
        
        if type_errors:
            errors.append(f"Type validation errors: {type_errors}")
            score *= max(0.3, 1.0 - len(type_errors) / len(self.output_schema))
        
        # Validate enum constraints
        enum_errors = []
        for field_name, field_config in self.output_schema.items():
            if field_name in parsed_response and "enum" in field_config:
                value = parsed_response[field_name]
                allowed_values = field_config["enum"]
                if value not in allowed_values:
                    enum_errors.append(f"{field_name}: '{value}' not in {allowed_values}")
        
        if enum_errors:
            errors.append(f"Enum validation errors: {enum_errors}")
            score *= max(0.5, 1.0 - len(enum_errors) / len([f for f in self.output_schema.values() if "enum" in f]))
        
        details.update({
            "missing_fields": missing_fields,
            "type_errors": type_errors,
            "enum_errors": enum_errors,
            "schema_compliance": score
        })
        
        return TemplateEvaluationResult(
            score=score,
            passed=score >= 0.8,  # 80% compliance threshold
            details=details,
            errors=errors
        )
    
    def _extract_response(self, pred) -> str:
        """Extract response string from prediction."""
        if isinstance(pred, str):
            return pred
        elif hasattr(pred, 'response'):
            return pred.response
        elif hasattr(pred, 'answer'):
            return pred.answer
        elif isinstance(pred, dict):
            # Try common response field names
            for field in ['response', 'answer', 'output', 'result']:
                if field in pred:
                    return str(pred[field])
        return str(pred)
    
    def _expects_json_format(self) -> bool:
        """Check if the output schema expects JSON format."""
        return len(self.output_schema) > 1 or any(
            config.get("type") in ["object", "array"] 
            for config in self.output_schema.values()
        )
    
    def _validate_field_type(self, value, expected_type: str) -> bool:
        """Validate field type."""
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "number":
            return isinstance(value, (int, float))
        elif expected_type == "integer":
            return isinstance(value, int)
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, list)
        elif expected_type == "object":
            return isinstance(value, dict)
        else:
            return True  # Unknown type, assume valid


class TemplateFactualRecallGrader(ConfigurableGrader):
    """
    Evaluates factual recall based on template context and expected knowledge.
    
    Uses DSPy's factual accuracy grader but adapted for template-specific context.
    """
    
    def __init__(self, template: PromptTemplate, **kwargs):
        """Initialize factual recall grader."""
        super().__init__(**kwargs)
        self.template = template
        
        if DSPY_AVAILABLE:
            # Create DSPy signature for factual evaluation
            self._create_factual_evaluation_signature()
    
    def _create_factual_evaluation_signature(self):
        """Create DSPy signature for factual evaluation."""
        class FactualRecallSignature(dspy.Signature):
            """Evaluate factual accuracy of response against provided context."""
            
            context: str = dspy.InputField(desc="Reference context or knowledge")
            response: str = dspy.InputField(desc="Response to evaluate")
            template_domain: str = dspy.InputField(desc="Domain of the template")
            
            factual_accuracy: float = dspy.OutputField(desc="Factual accuracy score 0.0-1.0")
            missing_facts: str = dspy.OutputField(desc="Key facts that are missing or incorrect")
            
        self.signature = FactualRecallSignature
        if DSPY_AVAILABLE:
            self.predictor = dspy.Predict(FactualRecallSignature)
    
    def __call__(self, example, pred, trace=None) -> Union[float, bool]:
        """Evaluate factual recall."""
        if not DSPY_AVAILABLE:
            # Fallback to simple evaluation
            return self._simple_factual_evaluation(example, pred, trace)
        
        # Extract context and response
        context = self._extract_context(example)
        response = self._extract_response(pred)
        template_domain = self.template.domain
        
        try:
            # Use DSPy predictor for evaluation
            result = self.predictor(
                context=context,
                response=response,
                template_domain=template_domain
            )
            
            score = float(result.factual_accuracy)
            
            if self.mode == "evaluation":
                return score
            else:
                return score >= 0.7  # 70% factual accuracy threshold
                
        except Exception as e:
            # Fallback to simple evaluation
            return self._simple_factual_evaluation(example, pred, trace)
    
    def _extract_context(self, example) -> str:
        """Extract context from example."""
        if isinstance(example, dict):
            # Look for context in common field names
            for field in ['context', 'reference', 'knowledge', 'facts']:
                if field in example:
                    return str(example[field])
            
            # Use template rendering as context
            rendered_template = self.template.render(**example)
            return rendered_template
        
        return str(example)
    
    def _extract_response(self, pred) -> str:
        """Extract response from prediction."""
        if isinstance(pred, str):
            return pred
        elif hasattr(pred, 'response'):
            return pred.response
        elif isinstance(pred, dict) and 'response' in pred:
            return pred['response']
        return str(pred)
    
    def _simple_factual_evaluation(self, example, pred, trace=None) -> Union[float, bool]:
        """Simple factual evaluation fallback."""
        # Basic keyword matching for factual recall
        context = self._extract_context(example).lower()
        response = self._extract_response(pred).lower()
        
        # Extract key terms from context
        key_terms = re.findall(r'\b\w{4,}\b', context)
        key_terms = [term for term in key_terms if term not in ['客户', '商品', '店铺']]  # Filter common words
        
        if not key_terms:
            return 1.0 if self.mode == "evaluation" else True
        
        # Check how many key terms appear in response
        recall_count = sum(1 for term in key_terms if term in response)
        recall_score = recall_count / len(key_terms)
        
        if self.mode == "evaluation":
            return recall_score
        else:
            return recall_score >= 0.5


class ChineseEcommerceGrader(CompositeGrader):
    """
    Specialized grader for Chinese e-commerce customer support templates.
    
    Combines multiple evaluation aspects specific to Chinese e-commerce:
    - Format compliance
    - Chinese language quality
    - E-commerce terminology usage
    - Customer service tone
    """
    
    def __init__(self, template: PromptTemplate, **kwargs):
        """Initialize Chinese e-commerce grader."""
        self.template = template
        
        # Extract mode from kwargs if present
        mode = kwargs.pop('mode', 'evaluation')
        
        # Define grader weights for e-commerce evaluation
        grader_weights = {
            "format_compliance": (TemplateFormatComplianceGrader(template, mode=mode), 0.3),
            "factual_recall": (TemplateFactualRecallGrader(template, mode=mode), 0.3),
            "chinese_quality": (self._create_chinese_quality_grader(mode), 0.2),
            "ecommerce_terms": (self._create_ecommerce_terms_grader(mode), 0.2)
        }
        
        super().__init__(grader_weights, **kwargs)
    
    def _create_chinese_quality_grader(self, mode="evaluation"):
        """Create grader for Chinese language quality."""
        class ChineseQualityGrader(ConfigurableGrader):
            def __init__(self, mode="evaluation"):
                super().__init__(mode=mode)
                
            def __call__(self, example, pred, trace=None):
                response = str(pred.response if hasattr(pred, 'response') else pred)
                
                # Check for Chinese characters
                chinese_pattern = r'[\u4e00-\u9fff]+'
                chinese_chars = len(re.findall(chinese_pattern, response))
                total_chars = len(response)
                
                if total_chars == 0:
                    return 0.0
                
                chinese_ratio = chinese_chars / total_chars
                
                # Check for polite expressions
                polite_expressions = ['您好', '请', '谢谢', '不好意思', '很抱歉']
                politeness_score = sum(1 for expr in polite_expressions if expr in response) / len(polite_expressions)
                
                overall_score = (chinese_ratio * 0.7 + politeness_score * 0.3)
                
                if self.mode == "evaluation":
                    return min(1.0, overall_score)
                else:
                    return overall_score >= 0.6
        
        return ChineseQualityGrader(mode)
    
    def _create_ecommerce_terms_grader(self, mode="evaluation"):
        """Create grader for e-commerce terminology."""
        class EcommerceTermsGrader(ConfigurableGrader):
            def __init__(self, mode="evaluation"):
                super().__init__(mode=mode)
                
            def __call__(self, example, pred, trace=None):
                response = str(pred.response if hasattr(pred, 'response') else pred)
                
                # E-commerce terms to look for
                ecommerce_terms = [
                    '商品', '产品', '价格', '优惠', '活动', '库存', 
                    '订单', '配送', '物流', '客服', '售后', '退换'
                ]
                
                # Count relevant terms used
                terms_used = sum(1 for term in ecommerce_terms if term in response)
                relevance_score = min(1.0, terms_used / 3)  # Expect at least 3 relevant terms
                
                if self.mode == "evaluation":
                    return relevance_score
                else:
                    return relevance_score >= 0.5
        
        return EcommerceTermsGrader(mode)


class TemplateCompositeGrader(CompositeGrader):
    """
    General-purpose composite grader for template evaluation.
    
    Automatically configures appropriate graders based on template domain and language.
    """
    
    def __init__(self, template: PromptTemplate, **kwargs):
        """Initialize template composite grader."""
        self.template = template
        
        # Configure graders based on template properties
        grader_weights = self._configure_graders()
        
        super().__init__(grader_weights, **kwargs)
    
    def _configure_graders(self) -> Dict[str, tuple]:
        """Configure graders based on template properties."""
        graders = {}
        
        # Always include format compliance
        graders["format_compliance"] = (TemplateFormatComplianceGrader(self.template), 0.4)
        
        # Always include factual recall
        graders["factual_recall"] = (TemplateFactualRecallGrader(self.template), 0.3)
        
        # Add domain-specific graders
        if self.template.domain == "e_commerce" and self.template.language.startswith("zh"):
            # Use specialized Chinese e-commerce grader
            return ChineseEcommerceGrader(self.template).grader_weights
        
        # Add language-specific graders
        if self.template.language.startswith("zh"):
            graders["chinese_quality"] = (self._create_chinese_quality_grader(), 0.3)
        
        # Normalize weights
        total_weight = sum(weight for _, weight in graders.values())
        normalized_graders = {
            name: (grader, weight / total_weight)
            for name, (grader, weight) in graders.items()
        }
        
        return normalized_graders
    
    def _create_chinese_quality_grader(self):
        """Create basic Chinese quality grader."""
        class BasicChineseQualityGrader(ConfigurableGrader):
            def __call__(self, example, pred, trace=None):
                response = str(pred.response if hasattr(pred, 'response') else pred)
                chinese_pattern = r'[\u4e00-\u9fff]+'
                has_chinese = bool(re.search(chinese_pattern, response))
                
                if self.mode == "evaluation":
                    return 1.0 if has_chinese else 0.5
                else:
                    return has_chinese
        
        return BasicChineseQualityGrader()


def create_template_grader(template: PromptTemplate, 
                          grader_type: str = "auto",
                          **kwargs) -> Union[TemplateFormatComplianceGrader, 
                                           TemplateFactualRecallGrader,
                                           ChineseEcommerceGrader,
                                           TemplateCompositeGrader]:
    """
    Factory function to create appropriate grader for a template.
    
    Args:
        template: PromptTemplate to create grader for
        grader_type: Type of grader ("auto", "format", "factual", "chinese_ecommerce", "composite")
        **kwargs: Additional arguments for grader initialization
        
    Returns:
        Appropriate grader instance
    """
    if grader_type == "auto":
        # Auto-select based on template properties
        if template.domain == "e_commerce" and template.language.startswith("zh"):
            return ChineseEcommerceGrader(template, **kwargs)
        else:
            return TemplateCompositeGrader(template, **kwargs)
    
    elif grader_type == "format":
        return TemplateFormatComplianceGrader(template, **kwargs)
    
    elif grader_type == "factual":
        return TemplateFactualRecallGrader(template, **kwargs)
    
    elif grader_type == "chinese_ecommerce":
        return ChineseEcommerceGrader(template, **kwargs)
    
    elif grader_type == "composite":
        return TemplateCompositeGrader(template, **kwargs)
    
    else:
        raise ValueError(f"Unknown grader type: {grader_type}")