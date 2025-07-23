#!/usr/bin/env python3
"""
Example: i18n for Chinese E-commerce Customer Support

This example shows how to build a multilingual customer support system
that adapts to Chinese e-commerce cultural expectations.
"""

import sys
from pathlib import Path

# Add dspy_kit to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dspy_kit"))

from dspy_kit.templates import (
    MultilingualTemplate,
    LanguageSelector,
    I18nAdapter
)


def create_chinese_ecommerce_template():
    """Create a multilingual template for Chinese e-commerce."""
    
    template_data = {
        "name": "chinese_ecommerce_support",
        "version": "1.0",
        "default_language": "zh",
        "supported_languages": ["zh", "en"],
        
        # Cultural configurations
        "languages": {
            "zh": {
                "formality": "professional",
                "currency": "¥",
                "support_hours": "9:00-21:00",
                "date_format": "YYYY年MM月DD日"
            },
            "en": {
                "formality": "friendly", 
                "currency": "$",
                "support_hours": "9AM-9PM CST",
                "date_format": "MM/DD/YYYY"
            }
        },
        
        # Input schema
        "input_schema": {
            "customer_name": {
                "type": "string",
                "description": {
                    "zh": "客户姓名",
                    "en": "Customer name"
                }
            },
            "order_id": {
                "type": "string",
                "description": {
                    "zh": "订单号",
                    "en": "Order ID"
                }
            },
            "issue_type": {
                "type": "string",
                "enum": ["shipping", "refund", "product", "other"],
                "description": {
                    "zh": "问题类型",
                    "en": "Issue type"
                }
            }
        },
        
        # Multilingual modules
        "modules": [
            {
                "name": "greeting",
                "priority": 10,
                "template": {
                    "zh": "{{ customer_name }}您好！欢迎联系客服中心。",
                    "en": "Hello {{ customer_name }}! Welcome to our customer service."
                }
            },
            {
                "name": "acknowledgment",
                "priority": 20,
                "conditional": "{% if issue_type %}",
                "template": {
                    "zh": "我了解您遇到了{{ issue_type }}相关的问题。让我来帮助您。",
                    "en": "I understand you have a {{ issue_type }} related issue. Let me help you with that."
                }
            },
            {
                "name": "order_lookup",
                "priority": 30,
                "conditional": "{% if order_id %}",
                "template": {
                    "zh": "我正在查询您的订单 #{{ order_id }}...",
                    "en": "I'm looking up your order #{{ order_id }}..."
                }
            },
            {
                "name": "shipping_info",
                "priority": 40,
                "conditional": "{% if issue_type == 'shipping' %}",
                "template": {
                    "zh": """📦 配送说明：
• 标准配送：3-5个工作日（江浙沪）
• 偏远地区：5-7个工作日
• 配送费：满99元包邮
• 查询热线：400-888-8888""",
                    "en": """📦 Shipping Information:
• Standard delivery: 3-5 business days (major cities)
• Remote areas: 5-7 business days  
• Free shipping over {{ _currency }}99
• Hotline: 400-888-8888"""
                }
            },
            {
                "name": "refund_policy",
                "priority": 40,
                "conditional": "{% if issue_type == 'refund' %}",
                "template": {
                    "zh": """💰 退款政策：
• 7天无理由退换货
• 收到商品后15天内可申请售后
• 退款将在3-5个工作日内到账
• 请保持商品包装完好""",
                    "en": """💰 Refund Policy:
• 7-day no-questions-asked returns
• 15-day after-sales service window
• Refunds processed in 3-5 business days
• Items must be in original packaging"""
                }
            },
            {
                "name": "support_hours",
                "priority": 80,
                "template": {
                    "zh": "我们的客服时间是每天{{ _support_hours }}。",
                    "en": "Our support hours are {{ _support_hours }} daily."
                }
            },
            {
                "name": "closing",
                "priority": 90,
                "template": {
                    "zh": "还有什么可以帮助您的吗？祝您购物愉快！",
                    "en": "Is there anything else I can help you with? Happy shopping!"
                }
            }
        ],
        
        # Tools for dynamic knowledge
        "tools": [
            "check_order_status",
            "calculate_shipping_fee",
            "check_refund_eligibility"
        ]
    }
    
    return MultilingualTemplate(template_data)


def demonstrate_language_detection():
    """Show how language detection works in practice."""
    print("🌐 Language Detection Demo")
    print("=" * 50)
    
    selector = LanguageSelector(
        default_language="zh",
        supported_languages=["zh", "en"]
    )
    
    # Simulate different customer scenarios
    scenarios = [
        {
            "name": "Chinese customer with browser in Chinese",
            "params": {
                "accept_language": "zh-CN,zh;q=0.9,en;q=0.8"
            },
            "expected": "zh"
        },
        {
            "name": "International customer preferring English", 
            "params": {
                "user_preference": "en"
            },
            "expected": "en"
        },
        {
            "name": "Customer explicitly selects English",
            "params": {
                "explicit_lang": "en",
                "accept_language": "zh-CN"  # Even if browser is Chinese
            },
            "expected": "en"
        }
    ]
    
    for scenario in scenarios:
        lang = selector.select_language(**scenario["params"])
        print(f"\n{scenario['name']}:")
        print(f"  Selected language: {lang}")
        print(f"  ✓ Correct" if lang == scenario["expected"] else "  ✗ Incorrect")


def demonstrate_cultural_adaptation():
    """Show cultural differences in customer service."""
    print("\n\n🎭 Cultural Adaptation Demo")
    print("=" * 50)
    
    template = create_chinese_ecommerce_template()
    
    # Same issue, different cultural approaches
    test_data = {
        "customer_name": "张先生",
        "order_id": "2024012345",
        "issue_type": "refund"
    }
    
    print("\n📋 Chinese Version (Professional/Formal):")
    print("-" * 40)
    zh_response = template.render(language="zh", **test_data)
    print(zh_response)
    
    # Switch customer name for English
    test_data["customer_name"] = "Mr. Zhang"
    
    print("\n📋 English Version (Friendly/Direct):")
    print("-" * 40)
    en_response = template.render(language="en", **test_data)
    print(en_response)
    
    print("\n💡 Key Differences:")
    print("• Chinese: More formal greeting, detailed policy info")
    print("• English: Friendlier tone, concise information")
    print("• Both: Same core information, culturally adapted")


def demonstrate_dynamic_language_switching():
    """Show runtime language switching."""
    print("\n\n🔄 Dynamic Language Switching Demo")
    print("=" * 50)
    
    template = create_chinese_ecommerce_template()
    
    # Simulate a conversation that switches languages
    print("\n💬 Customer starts in Chinese:")
    response1 = template.render(
        language="zh",
        customer_name="李女士",
        issue_type="shipping"
    )
    print(response1)
    
    print("\n💬 Customer switches to English mid-conversation:")
    response2 = template.render(
        language="en",
        customer_name="Ms. Li",
        issue_type="shipping"
    )
    print(response2)
    
    print("\n✅ System seamlessly adapts to language preference!")


def demonstrate_integration_with_tools():
    """Show how i18n works with dynamic knowledge tools."""
    print("\n\n🔧 Integration with Dynamic Tools Demo")
    print("=" * 50)
    
    template = create_chinese_ecommerce_template()
    
    # The template references tools that work regardless of language
    print("\n📋 Template tools (language-agnostic):")
    for tool in template.tools:
        print(f"  • {tool}")
    
    print("\n💡 Tools provide data, templates handle language:")
    print("• check_order_status → Returns order data")
    print("• Template → Formats in user's language")
    print("• Result → Culturally appropriate response")
    
    # Example tool integration
    print("\n📊 Example Flow:")
    print("1. Tool returns: {'status': 'shipped', 'eta': '2024-01-20'}")
    print("2. Chinese template: '您的订单已发货，预计2024年01月20日送达'")
    print("3. English template: 'Your order has shipped, ETA: 01/20/2024'")


def main():
    """Run the Chinese e-commerce i18n demo."""
    print("🛍️ Chinese E-commerce i18n Demo")
    print("=" * 70)
    print()
    
    try:
        # Run all demonstrations
        demonstrate_language_detection()
        demonstrate_cultural_adaptation()
        demonstrate_dynamic_language_switching()
        demonstrate_integration_with_tools()
        
        print("\n\n✅ Demo completed successfully!")
        print("\n🎯 Key Takeaways:")
        print("• i18n is more than translation - it's cultural adaptation")
        print("• Language detection should be smart with fallbacks")
        print("• Templates remain modular while supporting multiple languages")
        print("• Tools stay language-agnostic, templates handle presentation")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()