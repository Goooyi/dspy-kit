"""
Template validation implementation.
"""

import re
import json
from typing import List, Dict, Any, Optional, Set, Union
from dataclasses import dataclass, field
from pathlib import Path
import jinja2
from jinja2 import meta, Environment, BaseLoader

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

from .schemas import (
    get_template_schema, 
    get_chinese_ecommerce_schema,
    get_input_schema_schema,
    get_output_schema_schema,
    get_tool_schema
)


@dataclass
class ValidationError:
    """Represents a validation error."""
    level: str  # "error", "warning", "info"
    category: str  # "schema", "syntax", "reference", "tool", "domain"
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of template validation."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    info: List[ValidationError] = field(default_factory=list)
    
    def add_error(self, category: str, message: str, location: str = None, suggestion: str = None):
        """Add an error to the validation result."""
        self.errors.append(ValidationError("error", category, message, location, suggestion))
        self.is_valid = False
    
    def add_warning(self, category: str, message: str, location: str = None, suggestion: str = None):
        """Add a warning to the validation result."""
        self.warnings.append(ValidationError("warning", category, message, location, suggestion))
    
    def add_info(self, category: str, message: str, location: str = None, suggestion: str = None):
        """Add an info message to the validation result."""
        self.info.append(ValidationError("info", category, message, location, suggestion))
    
    def summary(self) -> str:
        """Get a summary of validation results."""
        if self.is_valid and not self.warnings:
            return "✅ Template validation passed"
        
        parts = []
        if self.errors:
            parts.append(f"❌ {len(self.errors)} errors")
        if self.warnings:
            parts.append(f"⚠️ {len(self.warnings)} warnings")
        if self.info:
            parts.append(f"ℹ️ {len(self.info)} info")
            
        return " | ".join(parts)


class TemplateValidator:
    """Comprehensive template validator."""
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.
        
        Args:
            strict_mode: If True, warnings become errors
        """
        self.strict_mode = strict_mode
        self.jinja_env = Environment(loader=BaseLoader())
        
        # Check dependencies
        if not JSONSCHEMA_AVAILABLE:
            print("⚠️ jsonschema not available - schema validation disabled")
    
    def validate_template(self, template_data: Dict[str, Any], 
                         template_content: str,
                         domain: str = "general") -> ValidationResult:
        """
        Validate a complete template.
        
        Args:
            template_data: Parsed YAML frontmatter
            template_content: Jinja2 template content 
            domain: Template domain for domain-specific validation
            
        Returns:
            ValidationResult with errors, warnings, and info
        """
        result = ValidationResult(is_valid=True)
        
        # 1. Schema validation
        self._validate_schema(template_data, domain, result)
        
        # 2. Jinja2 syntax validation
        self._validate_jinja_syntax(template_content, result)
        
        # 3. Cross-reference validation
        self._validate_cross_references(template_data, template_content, result)
        
        # 4. Tool validation
        self._validate_tools(template_data, result)
        
        # 5. Domain-specific validation
        if domain == "e_commerce":
            self._validate_chinese_ecommerce(template_data, template_content, result)
        
        # 6. Best practices validation
        self._validate_best_practices(template_data, template_content, result)
        
        return result
    
    def _validate_schema(self, template_data: Dict[str, Any], 
                        domain: str, result: ValidationResult):
        """Validate template against JSON schema."""
        if not JSONSCHEMA_AVAILABLE:
            result.add_warning("schema", "Schema validation skipped - jsonschema not available")
            return
        
        try:
            # Choose appropriate schema
            if domain == "e_commerce" and template_data.get("language", "").startswith("zh"):
                schema = get_chinese_ecommerce_schema()
            else:
                schema = get_template_schema()
            
            # Validate against schema
            jsonschema.validate(template_data, schema)
            result.add_info("schema", "Template structure validation passed")
            
        except jsonschema.ValidationError as e:
            error_path = " -> ".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"
            result.add_error(
                "schema", 
                f"Schema validation failed: {e.message}",
                error_path,
                "Check template YAML structure against schema requirements"
            )
        except Exception as e:
            result.add_error("schema", f"Schema validation error: {e}")
    
    def _validate_jinja_syntax(self, template_content: str, result: ValidationResult):
        """Validate Jinja2 template syntax."""
        try:
            # Parse the template
            ast = self.jinja_env.parse(template_content)
            result.add_info("syntax", "Jinja2 syntax validation passed")
            
            # Extract variables
            variables = meta.find_undeclared_variables(ast)
            if variables:
                result.add_info(
                    "syntax", 
                    f"Template uses {len(variables)} variables: {', '.join(sorted(variables))}"
                )
            
        except jinja2.TemplateSyntaxError as e:
            result.add_error(
                "syntax",
                f"Jinja2 syntax error: {e.message}",
                f"Line {e.lineno}",
                "Check Jinja2 template syntax"
            )
        except Exception as e:
            result.add_error("syntax", f"Template parsing error: {e}")
    
    def _validate_cross_references(self, template_data: Dict[str, Any], 
                                  template_content: str, result: ValidationResult):
        """Validate cross-references between schema and template."""
        try:
            # Extract template variables
            ast = self.jinja_env.parse(template_content)
            template_vars = meta.find_undeclared_variables(ast)
            
            # Get schema variables
            input_schema = template_data.get("input_schema", {})
            schema_vars = set(input_schema.keys())
            
            # Tool-related variables (auto-generated)
            tools = template_data.get("tools", [])
            tool_vars = set()
            for tool in tools:
                tool_vars.add(f"has_{tool}")
            tool_vars.add("available_tools")
            
            # Check for undefined variables
            undefined_vars = template_vars - schema_vars - tool_vars
            if undefined_vars:
                result.add_error(
                    "reference",
                    f"Template uses undefined variables: {', '.join(sorted(undefined_vars))}",
                    suggestion="Add missing variables to input_schema or check for typos"
                )
            
            # Check for unused schema variables
            unused_vars = schema_vars - template_vars
            if unused_vars:
                result.add_warning(
                    "reference",
                    f"Schema defines unused variables: {', '.join(sorted(unused_vars))}",
                    suggestion="Remove unused variables or use them in template"
                )
            
            # Validate required variables
            required_vars = {
                name for name, config in input_schema.items() 
                if config.get("required", False)
            }
            missing_required = required_vars - template_vars
            if missing_required:
                result.add_error(
                    "reference",
                    f"Template missing required variables: {', '.join(sorted(missing_required))}",
                    suggestion="Use all required variables in template"
                )
            
        except Exception as e:
            result.add_warning("reference", f"Cross-reference validation failed: {e}")
    
    def _validate_tools(self, template_data: Dict[str, Any], result: ValidationResult):
        """Validate tool configuration."""
        tools = template_data.get("tools", [])
        
        if not tools:
            return
        
        # Check tool name format
        for tool in tools:
            if not re.match(r"^[a-z_][a-z0-9_]*$", tool):
                result.add_error(
                    "tool",
                    f"Invalid tool name format: '{tool}'",
                    suggestion="Tool names should be lowercase with underscores"
                )
        
        # Check for common e-commerce tools
        common_tools = {"get_product_info", "get_shop_activities", "check_inventory"}
        domain = template_data.get("domain", "")
        language = template_data.get("language", "")
        
        if domain == "e_commerce" and language.startswith("zh"):
            available_common = set(tools) & common_tools
            if not available_common:
                result.add_warning(
                    "tool",
                    "Chinese e-commerce template without common tools",
                    suggestion=f"Consider adding: {', '.join(common_tools)}"
                )
        
        result.add_info("tool", f"Template configured with {len(tools)} tools")
    
    def _validate_chinese_ecommerce(self, template_data: Dict[str, Any], 
                                   template_content: str, result: ValidationResult):
        """Validate Chinese e-commerce specific requirements."""
        language = template_data.get("language", "")
        if not language.startswith("zh"):
            result.add_warning(
                "domain",
                "E-commerce template not using Chinese language",
                suggestion="Set language to 'zh-CN' for Chinese e-commerce"
            )
        
        # Check for Chinese content
        chinese_pattern = r"[\u4e00-\u9fff]+"
        if not re.search(chinese_pattern, template_content):
            result.add_warning(
                "domain",
                "E-commerce template contains no Chinese characters",
                suggestion="Add Chinese content for Chinese e-commerce scenarios"
            )
        
        # Check for common e-commerce terms
        ecommerce_terms = ["客服", "商品", "优惠", "活动", "库存", "订单"]
        found_terms = [term for term in ecommerce_terms if term in template_content]
        
        if len(found_terms) < 2:
            result.add_warning(
                "domain",
                "Template lacks common Chinese e-commerce terminology",
                suggestion=f"Consider including terms like: {', '.join(ecommerce_terms)}"
            )
        else:
            result.add_info(
                "domain",
                f"Template includes {len(found_terms)} e-commerce terms: {', '.join(found_terms)}"
            )
    
    def _validate_best_practices(self, template_data: Dict[str, Any], 
                                template_content: str, result: ValidationResult):
        """Validate against best practices."""
        # Check template length
        if len(template_content) > 5000:
            result.add_warning(
                "best_practice",
                f"Template is very long ({len(template_content)} chars)",
                suggestion="Consider breaking into smaller modules"
            )
        
        # Check for description
        if not template_data.get("metadata", {}).get("description"):
            result.add_info(
                "best_practice",
                "Template lacks description",
                suggestion="Add description in metadata for better documentation"
            )
        
        # Check input schema descriptions
        input_schema = template_data.get("input_schema", {})
        missing_descriptions = [
            name for name, config in input_schema.items()
            if not config.get("description")
        ]
        if missing_descriptions:
            result.add_warning(
                "best_practice",
                f"Input fields lack descriptions: {', '.join(missing_descriptions)}",
                suggestion="Add descriptions to all input fields"
            )
        
        # Check for version
        version = template_data.get("version", "1.0")
        if version == "1.0":
            result.add_info(
                "best_practice",
                "Template using default version",
                suggestion="Consider updating version for better tracking"
            )
    
    def validate_file(self, template_path: Union[str, Path]) -> ValidationResult:
        """Validate a template file."""
        from ..core.parser import TemplateParser
        
        try:
            parser = TemplateParser()
            template = parser.parse_file(template_path)
            
            # Convert template back to data format for validation
            template_data = template.to_dict()
            template_content = template_data.pop("content_template", "")
            
            domain = template_data.get("domain", "general")
            return self.validate_template(template_data, template_content, domain)
            
        except Exception as e:
            result = ValidationResult(is_valid=False)
            result.add_error("file", f"Failed to load template file: {e}")
            return result