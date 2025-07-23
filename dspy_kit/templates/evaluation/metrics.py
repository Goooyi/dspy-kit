"""
Template-aware evaluation metrics.

Provides convenient metric functions that work with dspy-kit's evaluation system
and integrate with our template system for format compliance and quality assessment.
"""

import json
import re
from typing import Dict, Any, List, Optional, Union

from ..core.template import PromptTemplate


def template_format_compliance(example, pred, trace=None, template: PromptTemplate = None) -> float:
    """
    Evaluate format compliance against template output_schema.
    
    Args:
        example: Input example
        pred: Model prediction
        trace: Optional trace information  
        template: Template with output_schema to validate against
        
    Returns:
        float: Compliance score (0.0-1.0)
    """
    if template is None:
        raise ValueError("Template is required for format compliance evaluation")
    
    from .template_graders import TemplateFormatComplianceGrader
    
    grader = TemplateFormatComplianceGrader(template, mode="evaluation")
    return grader(example, pred, trace)


def template_factual_recall(example, pred, trace=None, template: PromptTemplate = None) -> float:
    """
    Evaluate factual recall based on template context.
    
    Args:
        example: Input example with context
        pred: Model prediction
        trace: Optional trace information
        template: Template for context understanding
        
    Returns:
        float: Factual recall score (0.0-1.0)
    """
    if template is None:
        raise ValueError("Template is required for factual recall evaluation")
    
    from .template_graders import TemplateFactualRecallGrader
    
    grader = TemplateFactualRecallGrader(template, mode="evaluation")
    return grader(example, pred, trace)


def chinese_ecommerce_quality(example, pred, trace=None, template: PromptTemplate = None) -> float:
    """
    Evaluate Chinese e-commerce specific quality metrics.
    
    Args:
        example: Input example
        pred: Model prediction  
        trace: Optional trace information
        template: E-commerce template
        
    Returns:
        float: Quality score (0.0-1.0)
    """
    if template is None:
        raise ValueError("Template is required for Chinese e-commerce evaluation")
    
    from .template_graders import ChineseEcommerceGrader
    
    grader = ChineseEcommerceGrader(template, mode="evaluation")
    return grader(example, pred, trace)


def validate_customer_support_response_format(example, pred, trace=None) -> bool:
    """
    Legacy format validation for customer support responses.
    
    This maintains compatibility with existing evaluation code while
    providing enhanced validation through templates.
    
    Args:
        example: Input example
        pred: Model prediction
        trace: Optional trace information
        
    Returns:
        bool: True if format is valid
    """
    try:
        # Extract response
        if hasattr(pred, 'response'):
            response = pred.response
        elif isinstance(pred, dict) and 'response' in pred:
            response = pred['response']
        else:
            response = str(pred)
        
        # Basic validation - check if response is not empty
        if not response or not response.strip():
            return False
        
        # Check for Chinese characters if expected
        if hasattr(example, 'language') and example.language.startswith('zh'):
            chinese_pattern = r'[\u4e00-\u9fff]+'
            if not re.search(chinese_pattern, response):
                return False
        
        # Check for basic structure
        if len(response) < 10:  # Minimum response length
            return False
        
        return True
        
    except Exception:
        return False


def intentTopOneEM(example, pred, trace=None) -> bool:
    """
    Legacy exact match evaluation for intent classification.
    
    Args:
        example: Input example with expected intent
        pred: Model prediction with intent
        trace: Optional trace information
        
    Returns:
        bool: True if top intent matches exactly
    """
    try:
        # Extract expected intent
        if hasattr(example, 'intent'):
            expected_intent = example.intent
        elif isinstance(example, dict) and 'intent' in example:
            expected_intent = example['intent']
        else:
            return False
        
        # Extract predicted intent
        if hasattr(pred, 'intent'):
            predicted_intent = pred.intent
        elif isinstance(pred, dict) and 'intent' in pred:
            predicted_intent = pred['intent']
        elif hasattr(pred, 'response'):
            # Try to parse intent from response
            response = pred.response
            intent_match = re.search(r'intent["\']?\s*:\s*["\']?([^"\'\\n,}]+)', response, re.IGNORECASE)
            if intent_match:
                predicted_intent = intent_match.group(1).strip()
            else:
                return False
        else:
            return False
        
        # Normalize and compare
        expected_intent = str(expected_intent).strip().lower()
        predicted_intent = str(predicted_intent).strip().lower()
        
        return expected_intent == predicted_intent
        
    except Exception:
        return False


def validate_intents_format(example, pred, trace=None) -> bool:
    """
    Legacy validation for intent list format.
    
    Args:
        example: Input example
        pred: Model prediction with intents
        trace: Optional trace information
        
    Returns:
        bool: True if intents format is valid
    """
    try:
        # Extract intents
        if hasattr(pred, 'intents'):
            intents = pred.intents
        elif isinstance(pred, dict) and 'intents' in pred:
            intents = pred['intents']
        elif hasattr(pred, 'response'):
            # Try to parse intents from response
            response = pred.response
            try:
                parsed = json.loads(response)
                if 'intents' in parsed:
                    intents = parsed['intents']
                else:
                    return False
            except json.JSONDecodeError:
                return False
        else:
            return False
        
        # Validate intents format
        if not isinstance(intents, list):
            return False
        
        if len(intents) == 0:
            return False
        
        # Check each intent has required structure
        for intent in intents:
            if not isinstance(intent, dict):
                return False
            
            # Basic required fields
            if 'name' not in intent:
                return False
            
            if not isinstance(intent['name'], str) or not intent['name'].strip():
                return False
        
        return True
        
    except Exception:
        return False


def template_variable_coverage(example, pred, trace=None, template: PromptTemplate = None) -> float:
    """
    Evaluate how well the response covers template variables.
    
    Args:
        example: Input example with template variables
        pred: Model prediction
        trace: Optional trace information
        template: Template with variable definitions
        
    Returns:
        float: Variable coverage score (0.0-1.0)
    """
    if template is None:
        return 1.0  # No template to validate against
    
    try:
        # Extract response
        if hasattr(pred, 'response'):
            response = pred.response.lower()
        elif isinstance(pred, dict) and 'response' in pred:
            response = pred['response'].lower()
        else:
            response = str(pred).lower()
        
        # Get template variables from input_schema
        template_vars = list(template.input_schema.keys())
        
        if not template_vars:
            return 1.0  # No variables to check
        
        # Check how many variables are referenced in the response
        coverage_count = 0
        for var_name in template_vars:
            # Check if variable content appears in response
            if var_name in example:
                var_value = str(example[var_name]).lower()
                # Check if any part of the variable value appears in response
                if var_value and any(word in response for word in var_value.split() if len(word) > 2):
                    coverage_count += 1
        
        return coverage_count / len(template_vars) if template_vars else 1.0
        
    except Exception:
        return 0.0


def template_tool_utilization(example, pred, trace=None, template: PromptTemplate = None) -> float:
    """
    Evaluate how well the response indicates tool utilization.
    
    Args:
        example: Input example
        pred: Model prediction
        trace: Optional trace information  
        template: Template with tool definitions
        
    Returns:
        float: Tool utilization score (0.0-1.0)
    """
    if template is None or not template.tools:
        return 1.0  # No tools to utilize
    
    try:
        # Extract response
        if hasattr(pred, 'response'):
            response = pred.response.lower()
        elif isinstance(pred, dict) and 'response' in pred:
            response = pred['response'].lower()
        else:
            response = str(pred).lower()
        
        # Check for tool usage indicators
        tool_indicators = [
            '查询', '检查', '获取', '调用', '使用工具', 
            'tool', 'function', 'call', 'api'
        ]
        
        # Check if response mentions tool usage
        mentions_tools = any(indicator in response for indicator in tool_indicators)
        
        # Check if response mentions specific tools
        tool_mentions = sum(1 for tool in template.tools if tool.replace('_', '') in response)
        
        # Combine indicators
        if mentions_tools or tool_mentions > 0:
            return min(1.0, 0.5 + (tool_mentions / len(template.tools)) * 0.5)
        else:
            return 0.0
            
    except Exception:
        return 0.0