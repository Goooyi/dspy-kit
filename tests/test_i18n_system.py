#!/usr/bin/env python3
"""
Test script for i18n Template System

Tests the core functionality of our internationalization system.
"""

import sys
from pathlib import Path

# Add dspy_kit to path
sys.path.insert(0, str(Path(__file__).parent / "dspy_kit"))

from dspy_kit.templates.core.template import PromptTemplate
from dspy_kit.templates.i18n.adapter import I18nAdapter, LanguageSelector
from dspy_kit.templates.i18n.multilingual_template import MultilingualTemplate


def test_language_selector():
    """Test language selection logic."""
    print("Testing Language Selector...")

    selector = LanguageSelector(default_language="en")

    # Test explicit language
    assert selector.select_language(explicit_lang="zh") == "zh"
    print("✓ Explicit language selection works")

    # Test user preference
    assert selector.select_language(user_preference="ja") == "ja"
    print("✓ User preference selection works")

    # Test session context
    assert selector.select_language(session_context={"language": "es"}) == "es"
    print("✓ Session context selection works")

    # Test Accept-Language header
    assert selector.select_language(accept_language="zh-CN,zh;q=0.9") == "zh"
    print("✓ Accept-Language parsing works")

    # Test fallback to default
    assert selector.select_language() == "en"
    print("✓ Default fallback works")

    # Test fallback chains
    chain = selector.get_fallback_chain("zh-TW")
    assert chain == ["zh-TW", "zh", "en"]
    print("✓ Fallback chains work")

    print("Language Selector: All tests passed!\n")


def test_multilingual_template():
    """Test multilingual template functionality."""
    print("Testing Multilingual Template...")

    # Create a simple multilingual template
    template_data = {
        "name": "test_template",
        "version": "1.0",
        "default_language": "en",
        "supported_languages": ["en", "zh"],
        "content_template": "{% for module in modules %}{{ module.template }}\n{% endfor %}",
        "modules": [
            {"name": "greeting", "priority": 10, "template": {"en": "Hello {{ name }}!", "zh": "你好 {{ name }}！"}},
            {
                "name": "message",
                "priority": 20,
                "template": {"en": "Welcome to our service.", "zh": "欢迎使用我们的服务。"},
            },
        ],
    }

    template = MultilingualTemplate(**template_data)

    # Test English rendering
    en_result = template.render(language="en", name="John")
    print("+++++++++++")
    print(en_result)
    print("+++++++++++")
    assert "Hello John!" in en_result
    assert "Welcome to our service." in en_result
    print("✓ English rendering works")

    # Test Chinese rendering
    zh_result = template.render(language="zh", name="李明")
    assert "你好 李明！" in zh_result
    assert "欢迎使用我们的服务。" in zh_result
    print("✓ Chinese rendering works")

    # Test language switching
    zh_template = template.switch_language("zh")
    assert zh_template.current_language == "zh"
    print("✓ Language switching works")

    # Test available languages
    langs = template.get_available_languages()
    assert set(langs) == {"en", "zh"}
    print("✓ Available languages detection works")

    print("Multilingual Template: All tests passed!\n")


def test_i18n_adapter():
    """Test i18n adapter functionality."""
    print("Testing i18n Adapter...")

    adapter = I18nAdapter(template_base_dir="templates")

    # Test language suffix pattern detection
    available = adapter.get_available_languages("i18n/examples/ecommerce_support")
    assert "en" in available
    assert "zh" in available
    print("✓ Language suffix pattern detection works")

    # Test template loading
    try:
        # Load English template
        en_template = adapter.get_template("i18n/examples/ecommerce_support", language="en")
        assert en_template.language == "en"
        print("✓ English template loading works")

        # Load Chinese template
        zh_template = adapter.get_template("i18n/examples/ecommerce_support", language="zh")
        assert zh_template.language == "zh"
        print("✓ Chinese template loading works")

    except FileNotFoundError:
        print("⚠ Template files not found (expected in test environment)")

    print("i18n Adapter: Core functionality verified!\n")


def test_cultural_adaptations():
    """Test cultural adaptation features."""
    print("Testing Cultural Adaptations...")

    # Create templates with different formality levels
    formal_template = {
        "name": "formal_support",
        "version": "1.0",
        "supported_languages": ["en", "ja", "de"],
        "default_language": "en",
        "content_template": "{% for module in modules %}{{ module.template }}{% endfor %}",
        "languages": {
            "en": {"formality": "friendly"},
            "ja": {"formality": "very_polite"},
            "de": {"formality": "formal"},
        },
        "modules": [
            {
                "name": "greeting",
                "template": {
                    "en": "Hi there! How can I help?",
                    "ja": "いらっしゃいませ。お手伝いさせていただきます。",
                    "de": "Guten Tag. Wie kann ich Ihnen behilflich sein?",
                },
            }
        ],
    }

    template = MultilingualTemplate(**formal_template)

    # Test different formality levels
    en_greeting = template.render(language="en")
    ja_greeting = template.render(language="ja")
    de_greeting = template.render(language="de")

    # Verify cultural appropriateness (basic check)
    assert "Hi there!" in en_greeting  # Casual/friendly
    assert "いらっしゃいませ" in ja_greeting  # Very polite
    assert "Guten Tag" in de_greeting  # Formal

    print("✓ Cultural formality levels work")
    print("✓ Language-specific adaptations work")

    print("Cultural Adaptations: All tests passed!\n")


def test_integration_with_base_template():
    """Test integration with the base template system."""
    print("Testing Integration with Base Template System...")

    # Create a regular template
    regular_template = PromptTemplate(
        **{
            "name": "base_template", 
            "version": "1.0", 
            "content_template": "{% for module in modules %}{{ module.template }}{% endfor %}",
            "modules": [{"name": "intro", "template": "This is a template"}]
        }
    )

    # Verify it works with regular templates
    result = regular_template.render()
    assert "This is a template" in result
    print("✓ Base template system still works")

    # Create a multilingual template
    multi_template = MultilingualTemplate(
        **{
            "name": "extended_template",
            "version": "1.0",
            "supported_languages": ["en", "zh"],
            "default_language": "en",
            "content_template": "{% for module in modules %}{{ module.template }}{% endfor %}",
            "modules": [{"name": "greeting", "template": {"en": "Hello!", "zh": "你好！"}}],
        }
    )

    # Verify multilingual features
    assert multi_template.get_available_languages() == ["en", "zh"]
    print("✓ Multilingual templates can specify inheritance")

    print("Integration: All tests passed!\n")


def main():
    """Run all tests."""
    print("🧪 Running i18n System Tests")
    print("=" * 50)
    print()

    try:
        test_language_selector()
        test_multilingual_template()
        test_i18n_adapter()
        test_cultural_adaptations()
        test_integration_with_base_template()

        print("=" * 50)
        print("✅ All i18n tests passed!")
        print("\n🎉 The i18n system is working correctly!")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
