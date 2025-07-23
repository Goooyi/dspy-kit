"""
DSPy-Kit Template System

A modular prompt template system with YAML frontmatter and Jinja2 content,
designed for easy integration with DSPy and LLM-as-judge evaluation.
"""

from .core.template import PromptTemplate
from .core.parser import TemplateParser  
from .core.inheritance import InheritablePromptTemplate, TemplateResolver
from .adapters.dspy_adapter import DSPySignatureAdapter, create_dspy_signature, create_dspy_module
from .utils.migrator import PromptMigrator

# i18n components (optional import)
try:
    from .i18n import I18nAdapter, LanguageSelector, MultilingualTemplate
    I18N_AVAILABLE = True
except ImportError:
    I18N_AVAILABLE = False

# Validation components (optional import)
try:
    from .validation.validator import TemplateValidator, ValidationResult, ValidationError
    from .validation.schemas import get_template_schema, get_chinese_ecommerce_schema
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

__all__ = [
    "PromptTemplate",
    "TemplateParser", 
    "InheritablePromptTemplate",
    "TemplateResolver",
    "DSPySignatureAdapter",
    "create_dspy_signature",
    "create_dspy_module",
    "PromptMigrator"
]

# Add i18n exports if available
if I18N_AVAILABLE:
    __all__.extend([
        "I18nAdapter",
        "LanguageSelector",
        "MultilingualTemplate"
    ])

# Add validation exports if available
if VALIDATION_AVAILABLE:
    __all__.extend([
        "TemplateValidator",
        "ValidationResult", 
        "ValidationError",
        "get_template_schema",
        "get_chinese_ecommerce_schema"
    ])