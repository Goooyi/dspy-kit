#!/usr/bin/env python3
"""
Demo: i18n (Internationalization) Template System

This demo shows how to use multi-language templates with:
1. Language selection logic
2. Fallback mechanisms
3. Cultural adaptations
4. Integration with existing template system
"""

import sys
from pathlib import Path

# Add dspy_kit to path
sys.path.insert(0, str(Path(__file__).parent / "dspy_kit"))

import os

os.chdir(str(Path(__file__).parent))

from dspy_kit.templates import I18nAdapter, LanguageSelector, MultilingualTemplate, InheritablePromptTemplate


def demo_language_selection():
    """Demo the language selection logic."""
    print("🌍 Language Selection Logic Demo")
    print("=" * 50)
    print()

    selector = LanguageSelector(default_language="en")

    # Test cases for language selection
    test_cases = [
        {"name": "Explicit language", "params": {"explicit_lang": "zh"}, "expected": "zh"},
        {"name": "User preference", "params": {"user_preference": "ja"}, "expected": "ja"},
        {"name": "Session context", "params": {"session_context": {"language": "es"}}, "expected": "es"},
        {"name": "Accept-Language header", "params": {"accept_language": "zh-CN,zh;q=0.9,en;q=0.8"}, "expected": "zh"},
        {"name": "Fallback to default", "params": {}, "expected": "en"},
        {
            "name": "Unsupported language fallback",
            "params": {"explicit_lang": "ko"},  # Korean not supported
            "expected": "en",
        },
    ]

    print("📋 Selection Priority Tests:")
    for test in test_cases:
        result = selector.select_language(**test["params"])
        status = "✅" if result == test["expected"] else "❌"
        print(f"   {status} {test['name']}: {result}")

    # Show fallback chains
    print("\n🔄 Fallback Chains:")
    languages = ["zh-TW", "ja", "ar", "ko"]
    for lang in languages:
        chain = selector.get_fallback_chain(lang)
        print(f"   {lang}: {' → '.join(chain)}")


def demo_multilingual_template():
    """Demo multilingual template with language switching."""
    print("\n\n🗣️ Multilingual Template Demo")
    print("=" * 50)
    print()

    # Load multilingual template
    print("📋 Loading multilingual customer support template...")
    template = MultilingualTemplate.from_file("templates/i18n/examples/customer_support_multilingual.yaml")

    print(f"   Supported languages: {', '.join(template.get_available_languages())}")
    print()

    # Test data
    test_data = {"customer_query": "shipping time", "product_id": "PROD-123"}

    # Render in different languages
    languages = ["en", "zh", "ja", "es"]

    print("🌐 Rendering in Different Languages:")
    print("-" * 50)

    for lang in languages:
        print(f"\n[{lang.upper()}] {template.language_selector.get_language_config(lang).name}:")
        print("-" * 30)

        # Render template in specific language
        rendered = template.render(language=lang, **test_data)
        print(rendered)

    print("-" * 50)


def demo_i18n_adapter():
    """Demo i18n adapter with file organization patterns."""
    print("\n\n📁 i18n Adapter File Organization Demo")
    print("=" * 50)
    print()

    adapter = I18nAdapter(template_base_dir="templates")

    print("📋 Supported File Organization Patterns:")
    print()
    print("1️⃣ Language Suffix Pattern:")
    print("   templates/")
    print("   ├── ecommerce_support_en.yaml")
    print("   ├── ecommerce_support_zh.yaml")
    print("   └── ecommerce_support_ja.yaml")
    print()
    print("2️⃣ Language Folder Pattern:")
    print("   templates/i18n/")
    print("   ├── en/")
    print("   │   └── ecommerce_support.yaml")
    print("   ├── zh/")
    print("   │   └── ecommerce_support.yaml")
    print("   └── ja/")
    print("       └── ecommerce_support.yaml")
    print()
    print("3️⃣ Single File Pattern (multilingual):")
    print("   templates/")
    print("   └── customer_support_multilingual.yaml")
    print()

    # Test loading with different patterns
    print("🔍 Testing Template Loading:")

    # Pattern 1: Language suffix
    try:
        en_template = adapter.get_template("i18n/examples/ecommerce_support", language="en")
        zh_template = adapter.get_template("i18n/examples/ecommerce_support", language="zh")

        print(f"   ✅ Loaded English template: {en_template.name}")
        print(f"   ✅ Loaded Chinese template: {zh_template.name}")
    except Exception as e:
        print(f"   ❌ Error loading templates: {e}")

    # Check available languages
    available = adapter.get_available_languages("i18n/examples/ecommerce_support")
    print(f"\n   Available languages for 'ecommerce_support': {', '.join(available)}")


def demo_cultural_adaptation():
    """Demo cultural adaptations in templates."""
    print("\n\n🎭 Cultural Adaptation Demo")
    print("=" * 50)
    print()

    # Create templates with different formality levels
    print("📋 Formality Levels by Culture:")
    print()

    # English - Direct and friendly
    print("🇺🇸 English (Friendly/Direct):")
    print("   'Hi! I'll help you right away.'")
    print()

    # Chinese - Professional and balanced
    print("🇨🇳 Chinese (Professional):")
    print("   '您好！我马上为您处理。'")
    print("   (Hello! I'll handle this for you immediately.)")
    print()

    # Japanese - Very polite with honorifics
    print("🇯🇵 Japanese (Very Polite):")
    print("   '恐れ入りますが、お手伝いさせていただきます。'")
    print("   (I'm terribly sorry to trouble you, but please allow me to assist.)")
    print()

    # German - Formal and precise
    print("🇩🇪 German (Formal):")
    print("   'Guten Tag. Ich werde Ihnen gerne behilflich sein.'")
    print("   (Good day. I will gladly be of assistance to you.)")
    print()

    print("💡 Key Adaptations:")
    print("   • Language ≠ Translation")
    print("   • Consider cultural expectations")
    print("   • Adjust formality levels")
    print("   • Respect local customs")


def demo_integration_with_inheritance():
    """Demo how i18n works with template inheritance."""
    print("\n\n🔗 Integration with Template Inheritance")
    print("=" * 50)
    print()

    print("📋 Inheritance + i18n Pattern:")
    print()
    print("   base_template.yaml")
    print("   ├── chinese_ecommerce_template.yaml")
    print("   │   ├── example_template_zh.yaml")
    print("   │   └── example_template_en.yaml")
    print("   └── global_ecommerce_template.yaml")
    print("       ├── amazon_template_en.yaml")
    print("       ├── amazon_template_es.yaml")
    print("       └── amazon_template_de.yaml")
    print()

    print("✅ Benefits:")
    print("   • Base templates define structure")
    print("   • Domain templates add specialization")
    print("   • Language variants provide localization")
    print("   • Single source of truth for logic")
    print()

    # Example: Loading inherited multilingual template
    print("📊 Example Usage:")
    print("```python")
    print("# Load Chinese version of template")
    print("zh_template = adapter.get_template('example_template', language='zh')")
    print("")
    print("# Automatically inherits from:")
    print("# - example_template_zh.yaml")
    print("# - chinese_ecommerce_template.yaml")
    print("# - base_template.yaml")
    print("")
    print("# Switch to English while maintaining inheritance")
    print("en_template = zh_template.switch_language('en')")
    print("```")


def demo_practical_example():
    """Show a practical e-commerce example."""
    print("\n\n🛍️ Practical E-commerce Example")
    print("=" * 50)
    print()

    # Simulate a customer interaction
    print("📋 Scenario: Customer from different regions")
    print()

    customers = [
        {
            "name": "John (US)",
            "language": "en",
            "query": "What's your return policy?",
            "context": {"region": "US", "currency": "USD"},
        },
        {
            "name": "李明 (China)",
            "language": "zh",
            "query": "退货政策是什么？",
            "context": {"region": "CN", "currency": "CNY"},
        },
        {
            "name": "田中 (Japan)",
            "language": "ja",
            "query": "返品ポリシーは何ですか？",
            "context": {"region": "JP", "currency": "JPY"},
        },
    ]

    # Load appropriate template
    adapter = I18nAdapter()

    for customer in customers:
        print(f"\n👤 {customer['name']}:")
        print(f"   Query: {customer['query']}")
        print(f"   Response language: {customer['language']}")
        print(f"   Currency: {customer['context']['currency']}")
        print()

        # In real implementation, this would load and render the template
        print("   [Template would render appropriate response with:]")
        print(f"   - Language: {customer['language']}")
        print(f"   - Currency format: {customer['context']['currency']}")
        print("   - Cultural tone: Appropriate for region")


def main():
    """Run the i18n demo."""
    print("🌍 i18n Template System Demo")
    print("=" * 70)
    print()

    try:
        # Run all demos
        demo_language_selection()
        demo_multilingual_template()
        demo_i18n_adapter()
        demo_cultural_adaptation()
        demo_integration_with_inheritance()
        demo_practical_example()

        print("\n\n✅ i18n demo completed!")
        print("\n🎯 Key Features Demonstrated:")
        print("   • Smart language selection with fallback")
        print("   • Multiple file organization patterns")
        print("   • Cultural adaptations beyond translation")
        print("   • Seamless integration with inheritance")
        print("   • Practical multi-region support")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
