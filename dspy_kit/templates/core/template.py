"""
Core template system for modular prompt management.
"""

import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import jinja2

from .parser import TemplateParser
from .tools import ToolRegistry, ToolAwareTemplateEngine


@dataclass
class PromptTemplate:
    """
    A modular prompt template with YAML frontmatter and Jinja2 content.

    Supports:
    - Modular structure with reusable components
    - YAML frontmatter for configuration
    - Jinja2 templating for dynamic content
    - DSPy signature generation
    - Multiple concatenation styles
    """

    name: str
    version: str = "1.0"
    domain: str = "general"
    language: str = "en"

    # Schema definition
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)

    # Module configuration
    modules: List[Dict[str, Any]] = field(default_factory=list)

    # Template configuration
    concatenation_style: str = "sections"  # "sections", "xml", "minimal"
    separator: str = "----"
    include_headers: bool = True

    # Content
    content_template: str = ""

    # Tool configuration
    tools: List[str] = field(default_factory=list)

    # Runtime state
    _jinja_env: Optional[jinja2.Environment] = field(default=None, init=False)
    _parser: Optional[TemplateParser] = field(default=None, init=False)
    _tool_registry: Optional[ToolRegistry] = field(default=None, init=False)
    _tool_engine: Optional[ToolAwareTemplateEngine] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize Jinja2 environment and tool support."""
        self._jinja_env = jinja2.Environment(loader=jinja2.BaseLoader(), undefined=jinja2.StrictUndefined)
        self._parser = TemplateParser()

        # Initialize tool support if tools are defined
        if self.tools:
            from .tools import create_default_tool_registry

            self._tool_registry = create_default_tool_registry()
            self._tool_engine = ToolAwareTemplateEngine(self._tool_registry)

    @classmethod
    def from_file(cls, template_path: Union[str, Path]) -> "PromptTemplate":
        """Load template from YAML file with frontmatter."""
        parser = TemplateParser()
        return parser.parse_file(template_path)

    @classmethod
    def from_string(cls, template_string: str) -> "PromptTemplate":
        """Load template from string with YAML frontmatter."""
        parser = TemplateParser()
        return parser.parse_string(template_string)

    def render(self, **kwargs) -> str:
        """
        Render the complete prompt with given variables.

        Args:
            **kwargs: Template variables

        Returns:
            Rendered prompt string
        """
        # Validate required inputs
        self._validate_inputs(kwargs)

        # Process modules to render their templates with context
        rendered_modules = []
        for module in self.modules:
            rendered_module = module.copy()
            if "template" in rendered_module and isinstance(rendered_module["template"], str):
                # Render the module template with the context
                module_template = self._jinja_env.from_string(rendered_module["template"]) # type: ignore
                rendered_module["template"] = module_template.render(**kwargs)
            rendered_modules.append(rendered_module)

        # Include template's own modules in the render context
        render_context = {
            **kwargs,
            "modules": rendered_modules,
            "name": self.name,
            "version": self.version,
            "domain": self.domain,
        }

        # Use tool-aware rendering if tools are available
        if self.tools and self._tool_engine:
            return self._tool_engine.render_with_tools(self, render_context, self.tools)

        # Standard rendering
        template = self._jinja_env.from_string(self.content_template) # type: ignore
        rendered_content = template.render(**render_context)

        return rendered_content

    def render_module(self, module_name: str, **kwargs) -> str:
        """Render a specific module by name."""
        module = self._find_module(module_name)
        if not module:
            raise ValueError(f"Module '{module_name}' not found")

        if "template" in module:
            template = self._jinja_env.from_string(module["template"]) # type: ignore
            return template.render(**kwargs)

        return ""

    def get_required_inputs(self) -> List[str]:
        """Get list of required input variables."""
        required = []
        for name, config in self.input_schema.items():
            if config.get("required", False):
                required.append(name)
        return required

    def get_concatenation_format(self) -> str:
        """Get the concatenation format string."""
        if self.concatenation_style == "xml":
            return "<{header}>\n{content}\n</{header}>"
        elif self.concatenation_style == "sections":
            return "{separator} {header}\n{content}" if self.include_headers else "{content}"
        else:  # minimal
            return "{content}"

    def _validate_inputs(self, kwargs: Dict[str, Any]) -> None:
        """Validate that required inputs are provided."""
        required = self.get_required_inputs()
        missing = [key for key in required if key not in kwargs]
        if missing:
            raise ValueError(f"Missing required inputs: {missing}")

    def _find_module(self, name: str) -> Optional[Dict[str, Any]]:
        """Find module by name."""
        for module in self.modules:
            if module.get("name") == name:
                return module
        return None

    def set_tool_registry(self, tool_registry: ToolRegistry) -> None:
        """Set a custom tool registry."""
        self._tool_registry = tool_registry
        if self.tools:
            self._tool_engine = ToolAwareTemplateEngine(tool_registry)

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        if self._tool_registry:
            return self._tool_registry.list_tools()
        return []

    def add_tool(self, tool_name: str) -> None:
        """Add a tool to the template."""
        if tool_name not in self.tools:
            self.tools.append(tool_name)

            # Reinitialize tool engine if needed
            if self._tool_registry and not self._tool_engine:
                self._tool_engine = ToolAwareTemplateEngine(self._tool_registry)

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "domain": self.domain,
            "language": self.language,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "modules": self.modules,
            "concatenation_style": self.concatenation_style,
            "separator": self.separator,
            "include_headers": self.include_headers,
            "tools": self.tools,
            "content_template": self.content_template,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save template to YAML file."""
        data = self.to_dict()
        content = data.pop("content_template", "")

        # Create YAML frontmatter
        yaml_content = yaml.dump(data, allow_unicode=True, default_flow_style=False)

        # Combine with content
        full_content = f"---\n{yaml_content}---\n{content}"

        with open(path, "w", encoding="utf-8") as f:
            f.write(full_content)

    def validate(self, strict_mode: bool = False):
        """
        Validate the template structure and content.

        Args:
            strict_mode: If True, warnings become errors

        Returns:
            ValidationResult object with errors, warnings, and info
        """
        try:
            from ..validation.validator import TemplateValidator

            validator = TemplateValidator(strict_mode=strict_mode)
            template_data = self.to_dict()
            template_content = template_data.pop("content_template", "")

            return validator.validate_template(template_data, template_content, self.domain)
        except ImportError:
            # Fallback if validation module not available
            from ..validation.validator import ValidationResult

            result = ValidationResult(is_valid=True)
            result.add_warning("validation", "Template validation module not available")
            return result
