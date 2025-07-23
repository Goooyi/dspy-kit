"""
Template parser for YAML frontmatter + Jinja2 content.
"""

import re
from pathlib import Path
from typing import Any, Dict, Union

import yaml

# Avoid circular import - import at runtime


class TemplateParser:
    """Parser for template files with YAML frontmatter."""

    FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.MULTILINE | re.DOTALL)

    def parse_file(self, template_path: Union[str, Path]):
        """Parse template from file."""
        path = Path(template_path)
        if not path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        return self.parse_string(content)

    def parse_string(self, template_string: str):
        """Parse template from string."""
        match = self.FRONTMATTER_PATTERN.match(template_string.strip())

        if not match:
            # No frontmatter, treat entire content as template
            from .template import PromptTemplate

            return PromptTemplate(name="untitled", content_template=template_string)

        yaml_content, jinja_content = match.groups()

        # Parse YAML frontmatter
        try:
            metadata = yaml.safe_load(yaml_content) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML frontmatter: {e}")

        # Extract template configuration
        template_config = self._extract_template_config(metadata)
        template_config["content_template"] = jinja_content.strip()

        from .template import PromptTemplate

        return PromptTemplate(**template_config)

    def _extract_template_config(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract template configuration from metadata."""
        config = {}

        # Basic metadata
        config["name"] = metadata.get("name", "untitled")
        config["version"] = metadata.get("version", "1.0")
        config["domain"] = metadata.get("domain", "general")
        config["language"] = metadata.get("language", "en")

        # Schema
        config["input_schema"] = metadata.get("input_schema", {})
        config["output_schema"] = metadata.get("output_schema", {})

        # Modules
        config["modules"] = metadata.get("modules", [])

        # Configuration
        config["concatenation_style"] = metadata.get("concatenation_style", "sections")
        config["separator"] = metadata.get("separator", "----")
        config["include_headers"] = metadata.get("include_headers", True)

        # Tools
        config["tools"] = metadata.get("tools", [])

        return config
