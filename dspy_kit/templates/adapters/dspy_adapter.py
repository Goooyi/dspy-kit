"""
DSPy integration adapter for prompt templates.
"""

from dataclasses import make_dataclass
from typing import Any, Dict, Optional, Type

import dspy

from ..core.template import PromptTemplate


class DSPySignatureAdapter:
    """Adapter to convert PromptTemplate to DSPy signature and module."""

    def __init__(self, template: PromptTemplate):
        self.template = template

    def to_signature(self) -> Type[dspy.Signature]:
        """Convert template to DSPy signature class."""
        # Build fields with proper type annotations
        annotations = {}
        class_dict = {}

        # Input fields
        for name, config in self.template.input_schema.items():
            annotations[name] = str  # Use str type annotation
            class_dict[name] = dspy.InputField(desc=config.get("description", ""))

        # Output fields
        for name, config in self.template.output_schema.items():
            annotations[name] = str  # Use str type annotation
            class_dict[name] = dspy.OutputField(desc=config.get("description", ""))

        # Add annotations and docstring
        class_dict["__annotations__"] = annotations
        class_dict["__doc__"] = self._generate_docstring()

        # Create signature class dynamically
        signature_name = f"{self.template.name.replace('_', '').title()}Signature"
        signature_class = type(signature_name, (dspy.Signature,), class_dict)

        return signature_class

    def to_module(self, predictor_class: Optional[Type] = None) -> Type[dspy.Module]:
        """Convert template to DSPy module class with tool support."""
        if predictor_class is None:
            predictor_class = dspy.ChainOfThought

        signature_class = self.to_signature()
        template = self.template  # Capture for closure

        class TemplateModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.template = template

                # Get tools if available
                tools = []
                if template.tools and template._tool_registry:
                    tools = template._tool_registry.get_tools_for_template(template.tools)

                # Create predictor with tools
                if tools:
                    self.predictor = predictor_class(signature_class, tools=tools)
                else:
                    self.predictor = predictor_class(signature_class)

            def forward(self, **kwargs):
                # Render the template with variables (includes tool context)
                rendered_prompt = self.template.render(**kwargs)

                # Use the predictor with tool support
                result = self.predictor(**kwargs)

                # Add rendered template to result for debugging
                if hasattr(result, "rendered_template"):
                    result.rendered_template = rendered_prompt

                return result

        TemplateModule.__name__ = f"{self.template.name.replace('_', '').title()}Module"

        return TemplateModule

    def _generate_docstring(self) -> str:
        """Generate docstring for the signature."""
        lines = [f"Generated from template: {self.template.name}"]

        if self.template.domain != "general":
            lines.append(f"Domain: {self.template.domain}")

        if self.template.language != "en":
            lines.append(f"Language: {self.template.language}")

        return "\n".join(lines)


def create_dspy_signature(template: PromptTemplate) -> Type[dspy.Signature]:
    """Convenience function to create DSPy signature from template."""
    adapter = DSPySignatureAdapter(template)
    return adapter.to_signature()


def create_dspy_module(template: PromptTemplate, predictor_class: Optional[Type] = None) -> Type[dspy.Module]:
    """Convenience function to create DSPy module from template."""
    adapter = DSPySignatureAdapter(template)
    return adapter.to_module(predictor_class)
