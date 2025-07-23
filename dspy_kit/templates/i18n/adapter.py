"""
Core i18n adapter for multi-language template support.

This module provides the infrastructure for managing templates
in multiple languages with automatic selection and fallback.
"""

import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

from ..core.template import PromptTemplate
from ..core.inheritance import InheritablePromptTemplate


@dataclass
class LanguageConfig:
    """Configuration for a specific language."""

    code: str  # ISO 639-1 code (e.g., 'en', 'zh', 'ja')
    name: str  # Full name (e.g., 'English', '中文', '日本語')
    direction: str = "ltr"  # 'ltr' or 'rtl'

    # Formatting preferences
    date_format: str = "YYYY-MM-DD"
    time_format: str = "HH:mm"
    currency: str = "USD"
    currency_symbol: str = "$"
    decimal_separator: str = "."
    thousands_separator: str = ","

    # Cultural preferences
    formality_level: str = "medium"  # low, medium, high

    # Fallback language if content missing
    fallback: Optional[str] = None


class LanguageSelector:
    """
    Selects appropriate language based on various inputs.

    Selection priority:
    1. Explicit language parameter
    2. User profile preference
    3. Session context
    4. Accept-Language header
    5. Default language
    """

    def __init__(self, default_language: str = "en"):
        self.default_language = default_language
        self._supported_languages: Dict[str, LanguageConfig] = {}
        self._initialize_common_languages()

    def _initialize_common_languages(self):
        """Initialize common language configurations."""
        # English
        self._supported_languages["en"] = LanguageConfig(code="en", name="English", formality_level="medium")

        # Chinese (Simplified)
        self._supported_languages["zh"] = LanguageConfig(
            code="zh",
            name="中文",
            date_format="YYYY年MM月DD日",
            currency="CNY",
            currency_symbol="¥",
            formality_level="medium",
            fallback="en",
        )

        # Chinese (Traditional)
        self._supported_languages["zh-TW"] = LanguageConfig(
            code="zh-TW",
            name="繁體中文",
            date_format="YYYY年MM月DD日",
            currency="TWD",
            currency_symbol="NT$",
            formality_level="medium",
            fallback="zh",
        )

        # Japanese
        self._supported_languages["ja"] = LanguageConfig(
            code="ja",
            name="日本語",
            date_format="YYYY年MM月DD日",
            currency="JPY",
            currency_symbol="¥",
            formality_level="high",
            fallback="en",
        )

        # Spanish
        self._supported_languages["es"] = LanguageConfig(
            code="es",
            name="Español",
            date_format="DD/MM/YYYY",
            currency="EUR",
            currency_symbol="€",
            decimal_separator=",",
            thousands_separator=".",
            formality_level="medium",
            fallback="en",
        )

        # Arabic
        self._supported_languages["ar"] = LanguageConfig(
            code="ar",
            name="العربية",
            direction="rtl",
            date_format="DD/MM/YYYY",
            currency="SAR",
            currency_symbol="ر.س",
            formality_level="high",
            fallback="en",
        )

    def add_language(self, config: LanguageConfig):
        """Add or update a language configuration."""
        self._supported_languages[config.code] = config

    def select_language(
        self,
        explicit_lang: Optional[str] = None,
        user_preference: Optional[str] = None,
        session_context: Optional[Dict[str, Any]] = None,
        accept_language: Optional[str] = None,
    ) -> str:
        """
        Select the most appropriate language.

        Args:
            explicit_lang: Explicitly requested language
            user_preference: User's saved language preference
            session_context: Current session information
            accept_language: HTTP Accept-Language header value

        Returns:
            Selected language code
        """
        # Priority 1: Explicit language
        if explicit_lang and self._is_supported(explicit_lang):
            return explicit_lang

        # Priority 2: User preference
        if user_preference and self._is_supported(user_preference):
            return user_preference

        # Priority 3: Session context
        if session_context:
            session_lang = session_context.get("language") or session_context.get("lang")
            if session_lang and self._is_supported(session_lang):
                return session_lang

        # Priority 4: Accept-Language header
        if accept_language:
            parsed_lang = self._parse_accept_language(accept_language)
            if parsed_lang and self._is_supported(parsed_lang):
                return parsed_lang

        # Priority 5: Default
        return self.default_language

    def _is_supported(self, lang_code: str) -> bool:
        """Check if a language is supported."""
        return lang_code in self._supported_languages

    def _parse_accept_language(self, header: str) -> Optional[str]:
        """Parse Accept-Language header and return best match."""
        # Simple parsing - in production, use a proper parser
        # Example: "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7"
        languages = []
        for lang_q in header.split(","):
            parts = lang_q.strip().split(";")
            lang = parts[0]
            q = 1.0
            if len(parts) > 1 and parts[1].startswith("q="):
                try:
                    q = float(parts[1][2:])
                except ValueError:
                    q = 0.0
            languages.append((lang, q))

        # Sort by quality descending
        languages.sort(key=lambda x: x[1], reverse=True)

        # Find first supported language
        for lang, _ in languages:
            # Try exact match
            if lang in self._supported_languages:
                return lang
            # Try base language (e.g., 'en' for 'en-US')
            base_lang = lang.split("-")[0]
            if base_lang in self._supported_languages:
                return base_lang

        return None

    def get_fallback_chain(self, lang_code: str) -> List[str]:
        """Get fallback language chain for a given language."""
        chain = [lang_code]

        if lang_code in self._supported_languages:
            config = self._supported_languages[lang_code]
            if config.fallback:
                chain.append(config.fallback)

        # Always fall back to default if not already in chain
        if self.default_language not in chain:
            chain.append(self.default_language)

        return chain

    def get_language_config(self, lang_code: str) -> Optional[LanguageConfig]:
        """Get language configuration."""
        return self._supported_languages.get(lang_code)


class I18nAdapter:
    """
    Main i18n adapter for template system integration.

    Handles:
    - Multi-language template loading
    - Language selection
    - Content fallback
    - Variable formatting per language
    """

    def __init__(self, template_base_dir: Union[str, Path] = "templates", selector: Optional[LanguageSelector] = None):
        """
        Initialize i18n adapter.

        Args:
            template_base_dir: Base directory for templates
            selector: Language selector instance
        """
        self.template_base_dir = Path(template_base_dir)
        self.selector = selector or LanguageSelector()
        self._template_cache: Dict[str, Dict[str, PromptTemplate]] = {}

    def get_template(
        self, template_name: str, language: Optional[str] = None, context: Optional[Dict[str, Any]] = None
    ) -> PromptTemplate:
        """
        Get template in the appropriate language.

        Args:
            template_name: Base template name (without language suffix)
            language: Requested language (optional)
            context: Context for language selection

        Returns:
            Template in selected language
        """
        # Select language
        selected_lang = self.selector.select_language(
            explicit_lang=language,
            user_preference=context.get("user_language") if context else None,
            session_context=context,
        )

        # Get fallback chain
        fallback_chain = self.selector.get_fallback_chain(selected_lang)

        # Try to load template in order of preference
        for lang in fallback_chain:
            template = self._load_template(template_name, lang)
            if template:
                return template

        # If no template found, raise error
        raise FileNotFoundError(f"Template '{template_name}' not found in any language: {fallback_chain}")

    def _load_template(self, template_name: str, language: str) -> Optional[PromptTemplate]:
        """
        Load a template for a specific language.

        Supports multiple file organization patterns:
        1. Language suffix: customer_support_zh.yaml, customer_support_en.yaml
        2. Language folders: i18n/zh/customer_support.yaml, i18n/en/customer_support.yaml
        3. Single file with language sections (detected by content)
        """
        # Check cache first
        cache_key = f"{template_name}:{language}"
        if cache_key in self._template_cache:
            return self._template_cache.get(cache_key)

        # Try pattern 1: Language suffix
        suffix_path = self.template_base_dir / f"{template_name}_{language}.yaml"
        if suffix_path.exists():
            template = self._load_and_cache(suffix_path, cache_key)
            if template:
                return template

        # Try pattern 2: Language folders
        folder_path = self.template_base_dir / "i18n" / language / f"{template_name}.yaml"
        if folder_path.exists():
            template = self._load_and_cache(folder_path, cache_key)
            if template:
                return template

        # Try pattern 3: Single file with language sections
        single_path = self.template_base_dir / f"{template_name}.yaml"
        if single_path.exists():
            template = self._load_multilingual_template(single_path, language)
            if template:
                self._template_cache[cache_key] = template
                return template

        return None

    def _load_and_cache(self, path: Path, cache_key: str) -> Optional[PromptTemplate]:
        """Load template from file and cache it."""
        try:
            template = InheritablePromptTemplate.from_file(str(path))
            self._template_cache[cache_key] = template
            return template
        except Exception:
            return None

    def _load_multilingual_template(self, path: Path, language: str) -> Optional[PromptTemplate]:
        """
        Load a multilingual template and extract specific language.

        This handles templates where multiple languages are in one file.
        """
        try:
            # This would require extending our template parser
            # For now, return None to try other patterns
            return None
        except Exception:
            return None

    def register_template_directory(self, directory: Union[str, Path], pattern: str = "suffix"):
        """
        Register a directory containing templates with a specific pattern.

        Args:
            directory: Directory path
            pattern: Organization pattern ('suffix', 'folder', 'single')
        """
        # Implementation for scanning and registering templates
        pass

    def clear_cache(self):
        """Clear the template cache."""
        self._template_cache.clear()

    def get_available_languages(self, template_name: str) -> List[str]:
        """Get list of available languages for a template."""
        available = []

        # Check suffix pattern
        for lang_config in self.selector._supported_languages.values():
            suffix_path = self.template_base_dir / f"{template_name}_{lang_config.code}.yaml"
            if suffix_path.exists():
                available.append(lang_config.code)

        # Check folder pattern
        i18n_dir = self.template_base_dir / "i18n"
        if i18n_dir.exists():
            for lang_dir in i18n_dir.iterdir():
                if lang_dir.is_dir():
                    template_path = lang_dir / f"{template_name}.yaml"
                    if template_path.exists():
                        available.append(lang_dir.name)

        return available
