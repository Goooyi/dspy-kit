"""
Template-aware evaluation system.

Integrates the template system with dspy-kit's evaluation graders
to provide template-specific evaluation metrics including:
- Format compliance based on template output_schema
- Factual recall evaluation
- Domain-specific grading (Chinese e-commerce)
- Template rendering quality assessment
"""

from .template_graders import (
    TemplateFormatComplianceGrader,
    TemplateFactualRecallGrader,
    ChineseEcommerceGrader,
    TemplateCompositeGrader,
    create_template_grader
)

from .metrics import (
    template_format_compliance,
    template_factual_recall,
    chinese_ecommerce_quality
)

__all__ = [
    "TemplateFormatComplianceGrader",
    "TemplateFactualRecallGrader", 
    "ChineseEcommerceGrader",
    "TemplateCompositeGrader",
    "create_template_grader",
    "template_format_compliance",
    "template_factual_recall",
    "chinese_ecommerce_quality"
]