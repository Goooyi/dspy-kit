"""
i18n (internationalization) adapter system for multi-language templates.
"""

from .adapter import I18nAdapter, LanguageSelector
from .multilingual_template import MultilingualTemplate

__all__ = [
    "I18nAdapter",
    "LanguageSelector", 
    "MultilingualTemplate"
]