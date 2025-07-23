# i18n (Internationalization) Template Guide

## Overview

The i18n adapter system enables multi-language support in the DSPy-Kit template system. It provides:

- 🌍 Multi-language template variants
- 🔄 Smart language selection with fallback chains
- 🎭 Cultural adaptations beyond translation
- 📁 Multiple file organization patterns
- 🔗 Seamless integration with template inheritance

## Core Components

### 1. LanguageSelector

Handles intelligent language selection based on multiple inputs:

```python
from dspy_kit.templates import LanguageSelector

selector = LanguageSelector(
    default_language="en",
    supported_languages=["en", "zh", "ja", "es"]
)

# Selection priority (first match wins):
# 1. Explicit language
language = selector.select_language(explicit_lang="zh")  # => "zh"

# 2. User preference
language = selector.select_language(user_preference="ja")  # => "ja"

# 3. Session context
language = selector.select_language(
    session_context={"language": "es"}
)  # => "es"

# 4. Accept-Language header
language = selector.select_language(
    accept_language="zh-CN,zh;q=0.9,en;q=0.8"
)  # => "zh"

# 5. Default fallback
language = selector.select_language()  # => "en"
```

### 2. MultilingualTemplate

Extends PromptTemplate with language switching capabilities:

```python
from dspy_kit.templates import MultilingualTemplate

# Load from file
template = MultilingualTemplate.from_file(
    "templates/customer_support_multilingual.yaml"
)

# Render in different languages
en_response = template.render(language="en", customer_name="John")
zh_response = template.render(language="zh", customer_name="李明")

# Switch language for a template instance
zh_template = template.switch_language("zh")
```

### 3. I18nAdapter

Manages template loading with language variants:

```python
from dspy_kit.templates import I18nAdapter

adapter = I18nAdapter(template_base_dir="templates")

# Get template in specific language
zh_template = adapter.get_template(
    "ecommerce_support", 
    language="zh"
)

# Check available languages
languages = adapter.get_available_languages("ecommerce_support")
# => ["en", "zh", "ja"]
```

## File Organization Patterns

### Pattern 1: Language Suffix

```
templates/
├── ecommerce_support_en.yaml
├── ecommerce_support_zh.yaml
└── ecommerce_support_ja.yaml
```

### Pattern 2: Language Folders

```
templates/i18n/
├── en/
│   └── ecommerce_support.yaml
├── zh/
│   └── ecommerce_support.yaml
└── ja/
    └── ecommerce_support.yaml
```

### Pattern 3: Single Multilingual File

```yaml
# customer_support_multilingual.yaml
---
name: "customer_support"
default_language: "en"
supported_languages: ["en", "zh", "ja"]

modules:
  - name: "greeting"
    template:
      en: "Hello! How can I help?"
      zh: "您好！有什么可以帮助您的？"
      ja: "いらっしゃいませ！ご用件をお聞かせください。"
---
```

## Language Fallback Chains

The system automatically handles language variants:

```python
# Fallback chains for common scenarios:
"zh-TW" → "zh" → "en"  # Traditional Chinese → Chinese → English
"en-GB" → "en" → default  # British English → English → Default
"ja-JP" → "ja" → "en"  # Japanese (Japan) → Japanese → English
```

## Cultural Adaptations

Beyond translation, templates can adapt to cultural expectations:

```yaml
# Define cultural settings
languages:
  en:
    formality: "friendly"
    date_format: "MM/DD/YYYY"
    currency_symbol: "$"
  zh:
    formality: "professional"
    date_format: "YYYY年MM月DD日"
    currency_symbol: "¥"
  ja:
    formality: "very_polite"
    date_format: "YYYY年MM月DD日"
    currency_symbol: "¥"
```

## Integration with Template Inheritance

Combine i18n with template inheritance for powerful combinations:

```yaml
# Base template (base_ecommerce.yaml)
---
name: "base_ecommerce"
modules:
  - name: "structure"
    template: "..."
---

# Chinese e-commerce template
---
name: "chinese_ecommerce"
extends: "base_ecommerce"
default_language: "zh"
---

# Shop-specific multilingual template
---
name: "example_shop"
extends: "chinese_ecommerce"
supported_languages: ["zh", "en"]
modules:
  - name: "greeting"
    template:
      zh: "欢迎光临示例官方旗舰店！"
      en: "Welcome to Example Official Store!"
---
```

## Best Practices

### 1. Language Detection Strategy

```python
def get_user_language(request):
    """Recommended language detection flow."""
    return selector.select_language(
        # Check URL parameter first
        explicit_lang=request.args.get("lang"),
        # Then user profile preference
        user_preference=request.user.language_preference,
        # Then session data
        session_context=request.session,
        # Finally HTTP headers
        accept_language=request.headers.get("Accept-Language")
    )
```

### 2. Template Design

- **Keep modules language-agnostic**: Structure and logic should be the same across languages
- **Use language-specific variables**: Store cultural data in language configs
- **Plan for missing translations**: Always have fallback content

### 3. Performance Optimization

```python
# Cache templates by language
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_template(template_name, language):
    return adapter.get_template(template_name, language)
```

## Example: E-commerce Customer Support

### Multilingual Template Definition

```yaml
---
name: "ecommerce_support"
version: "1.0"
default_language: "en"
supported_languages: ["en", "zh", "ja"]

# Cultural configurations
languages:
  en:
    formality: "friendly"
    support_hours: "24/7"
    phone_format: "1-800-XXXX"
  zh:
    formality: "professional"
    support_hours: "全天候"
    phone_format: "400-XXX-XXXX"
  ja:
    formality: "very_polite"
    support_hours: "24時間対応"
    phone_format: "0120-XXX-XXX"

# Input/output schemas with translations
input_schema:
  query:
    type: "string"
    description:
      en: "Customer question"
      zh: "客户问题"
      ja: "お客様のご質問"

# Multilingual modules
modules:
  - name: "greeting"
    priority: 10
    template:
      en: |
        Hello! I'm here to help with your {{ query_type }} question.
        Our support team is available {{ _support_hours }}.
      zh: |
        您好！我来帮助您解决关于{{ query_type }}的问题。
        我们的客服团队{{ _support_hours }}为您服务。
      ja: |
        いらっしゃいませ。{{ query_type }}についてのご質問を承ります。
        サポートチームは{{ _support_hours }}いたします。

  - name: "contact"
    priority: 80
    template:
      en: "For immediate assistance, call {{ _phone_format }}"
      zh: "如需即时帮助，请拨打 {{ _phone_format }}"
      ja: "お急ぎの場合は {{ _phone_format }} までお電話ください"
---
```

### Usage in Code

```python
# Initialize adapter
adapter = I18nAdapter()

# Detect user language
user_lang = selector.select_language(
    session_context={"language": user.preferred_language},
    accept_language=request.headers.get("Accept-Language")
)

# Load and render template
template = adapter.get_template("ecommerce_support", language=user_lang)
response = template.render(
    query_type="shipping",
    language=user_lang  # Pass language for runtime selection
)

# Or use DSPy integration
dspy_module = create_dspy_module(template, language=user_lang)
```

## Testing i18n Templates

```python
def test_multilingual_template():
    """Test template in all supported languages."""
    template = MultilingualTemplate.from_file("template.yaml")
    
    test_data = {"product": "laptop", "issue": "shipping"}
    
    for lang in template.get_available_languages():
        result = template.render(language=lang, **test_data)
        
        # Verify language-specific content
        assert_contains_language_markers(result, lang)
        
        # Check cultural appropriateness
        assert_formality_level(result, lang)
        
        # Validate completeness
        assert_all_modules_rendered(result)
```

## Migration Guide

To add i18n to existing templates:

1. **Identify translatable content**
   ```yaml
   # Before
   modules:
     - name: "greeting"
       template: "Hello!"
   
   # After
   modules:
     - name: "greeting"
       template:
         en: "Hello!"
         zh: "你好！"
   ```

2. **Add language metadata**
   ```yaml
   default_language: "en"
   supported_languages: ["en", "zh", "ja"]
   ```

3. **Configure cultural settings**
   ```yaml
   languages:
     en:
       formality: "casual"
     zh:
       formality: "professional"
   ```

4. **Test with language switching**
   ```python
   for lang in ["en", "zh", "ja"]:
       result = template.render(language=lang)
       validate_language_output(result, lang)
   ```

## Future Enhancements

- [ ] Automatic translation suggestions using LLMs
- [ ] Language-specific module ordering
- [ ] RTL (Right-to-Left) language support
- [ ] Pluralization rules by language
- [ ] Number and date formatting by locale
- [ ] Integration with standard i18n libraries