"""
Multilingual template support with seamless integration to existing template system.

This module extends our PromptTemplate to support multiple languages
in a single template definition.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import copy

from ..core.template import PromptTemplate
from ..core.parser import TemplateParser
from .adapter import LanguageSelector, LanguageConfig


@dataclass
class MultilingualModule:
    """A module that supports multiple language variants."""

    name: str
    priority: int
    description: str = ""
    conditional: Optional[str] = None

    # Language variants of the template
    templates: Dict[str, str] = field(default_factory=dict)

    # Optional language-specific conditionals
    conditionals: Dict[str, str] = field(default_factory=dict)

    def get_template(self, language: str, fallback_chain: List[str]) -> str:
        """Get template content for a specific language with fallback."""
        # Try requested language
        if language in self.templates:
            return self.templates[language]

        # Try fallback languages
        for lang in fallback_chain:
            if lang in self.templates:
                return self.templates[lang]

        # Return first available template
        if self.templates:
            return next(iter(self.templates.values()))

        return ""

    def get_conditional(self, language: str) -> Optional[str]:
        """Get conditional for a specific language."""
        return self.conditionals.get(language, self.conditional)

    def to_monolingual_module(self, language: str, fallback_chain: List[str]) -> Dict[str, Any]:
        """Convert to a standard module dict for a specific language."""
        return {
            "name": self.name,
            "priority": self.priority,
            "description": self.description,
            "template": self.get_template(language, fallback_chain),
            "conditional": self.get_conditional(language),
        }


class MultilingualTemplate(PromptTemplate):
    """
    Extended PromptTemplate that supports multiple languages.

    Can be used as a drop-in replacement for PromptTemplate with
    added multilingual capabilities.
    """

    def __init__(
        self, language_selector: Optional[LanguageSelector] = None, current_language: Optional[str] = None, **kwargs
    ):
        """
        Initialize multilingual template.

        Args:
            language_selector: Language selection logic
            current_language: Currently selected language
            **kwargs: Standard PromptTemplate arguments
        """
        # Store original modules before conversion
        self._multilingual_modules: List[MultilingualModule] = []
        self.language_selector = language_selector or LanguageSelector()
        self.current_language = current_language or self.language_selector.default_language

        # Process modules if they contain language variants
        if "modules" in kwargs:
            kwargs["modules"] = self._process_modules(kwargs["modules"])

        # Store supported languages
        self.supported_languages = kwargs.pop("supported_languages", ["en"])
        self.default_language = kwargs.pop("default_language", "en")

        # Language-specific metadata
        self.language_configs = kwargs.pop("languages", {})

        super().__init__(**kwargs)

    def _process_modules(self, modules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process modules to handle multilingual content."""
        processed_modules = []

        for module in modules:
            if isinstance(module.get("template"), dict):
                # This is a multilingual module
                ml_module = MultilingualModule(
                    name=module["name"],
                    priority=module.get("priority", 100),
                    description=module.get("description", ""),
                    conditional=module.get("conditional"),
                    templates=module["template"],
                )

                # Store for later use
                self._multilingual_modules.append(ml_module)

                # Convert to monolingual for current language
                fallback_chain = self.language_selector.get_fallback_chain(self.current_language)
                processed_modules.append(ml_module.to_monolingual_module(self.current_language, fallback_chain))
            else:
                # Regular module, use as-is
                processed_modules.append(module)

        return processed_modules

    def switch_language(self, language: str) -> "MultilingualTemplate":
        """
        Create a new template instance for a different language.

        Args:
            language: Target language code

        Returns:
            New MultilingualTemplate instance in the target language
        """
        if language == self.current_language:
            return self

        # Create new instance with same configuration but different language
        new_template = copy.deepcopy(self)
        new_template.current_language = language

        # Re-process modules for new language
        fallback_chain = self.language_selector.get_fallback_chain(language)
        new_modules = []

        for ml_module in new_template._multilingual_modules:
            new_modules.append(ml_module.to_monolingual_module(language, fallback_chain))

        # Add non-multilingual modules
        for module in self.modules:
            if not any(ml.name == module["name"] for ml in self._multilingual_modules):
                new_modules.append(module)

        new_template.modules = new_modules

        return new_template

    def render(self, language: Optional[str] = None, **kwargs) -> str:
        """
        Render template with optional language override.

        Args:
            language: Optional language override
            **kwargs: Template variables

        Returns:
            Rendered template string
        """
        if language and language != self.current_language:
            # Render in different language
            temp_template = self.switch_language(language)
            return temp_template.render(**kwargs)

        # Add language-specific formatting to context
        lang_config = self.language_selector.get_language_config(self.current_language)
        if lang_config:
            kwargs.update(
                {
                    "_lang": self.current_language,
                    "_currency": lang_config.currency,
                    "_currency_symbol": lang_config.currency_symbol,
                    "_date_format": lang_config.date_format,
                    "_formality": lang_config.formality_level,
                }
            )

        return super().render(**kwargs)

    def get_available_languages(self) -> List[str]:
        """Get list of languages this template supports."""
        languages = set(self.supported_languages)

        # Check multilingual modules
        for ml_module in self._multilingual_modules:
            languages.update(ml_module.templates.keys())

        return sorted(list(languages))

    def validate_language_coverage(self) -> Dict[str, List[str]]:
        """
        Check which modules have missing language variants.

        Returns:
            Dict mapping module names to missing languages
        """
        missing = {}

        for ml_module in self._multilingual_modules:
            missing_langs = []
            for lang in self.supported_languages:
                if lang not in ml_module.templates:
                    missing_langs.append(lang)

            if missing_langs:
                missing[ml_module.name] = missing_langs

        return missing

    @classmethod
    def from_file(
        cls,
        template_path: str,
        language_selector: Optional[LanguageSelector] = None,
        current_language: Optional[str] = None,
    ) -> "MultilingualTemplate":
        """
        Load multilingual template from file.

        Args:
            template_path: Path to template file
            language_selector: Language selection logic
            current_language: Initial language

        Returns:
            MultilingualTemplate instance
        """
        parser = TemplateParser()
        template_dict = parser.parse_file(template_path).to_dict()

        return cls(language_selector=language_selector, current_language=current_language, **template_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including multilingual information."""
        base_dict = super().to_dict()

        # Add multilingual specific fields
        base_dict.update(
            {
                "supported_languages": self.supported_languages,
                "default_language": self.default_language,
                "current_language": self.current_language,
                "languages": self.language_configs,
            }
        )

        # Convert modules back to multilingual format
        if self._multilingual_modules:
            ml_modules = []
            ml_module_names = {ml.name for ml in self._multilingual_modules}

            for module in base_dict["modules"]:
                if module["name"] in ml_module_names:
                    # Find corresponding multilingual module
                    ml_module = next(ml for ml in self._multilingual_modules if ml.name == module["name"])
                    ml_modules.append(
                        {
                            "name": ml_module.name,
                            "priority": ml_module.priority,
                            "description": ml_module.description,
                            "template": ml_module.templates,
                            "conditional": ml_module.conditional,
                        }
                    )
                else:
                    ml_modules.append(module)

            base_dict["modules"] = ml_modules

        return base_dict
