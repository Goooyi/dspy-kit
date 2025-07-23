#!/usr/bin/env python3
"""
Demo: LLM-as-Judge Template Analysis

This demo shows how to use LLM-as-Judge features to:
1. Analyze template quality
2. Get optimization suggestions
3. Generate improved templates
4. Compare template versions
"""

import sys
from pathlib import Path

# Add dspy_kit to path
sys.path.insert(0, str(Path(__file__).parent / "dspy_kit"))

import os

os.chdir(str(Path(__file__).parent))

import dspy
from dspy_kit.templates import InheritablePromptTemplate
from dspy_kit.templates.analysis.template_judge import TemplateJudge, TemplateQualityGrader, create_template_judge


def demo_template_analysis():
    """Demo template quality analysis."""
    print("🔍 Template Quality Analysis Demo")
    print("=" * 50)
    print()

    # Load a template to analyze
    print("📋 Loading template for analysis...")
    template = InheritablePromptTemplate.from_file("templates/shops/example_shop_customer_support.yaml")

    # Create template judge (using mock for demo)
    print("🧑‍⚖️ Creating LLM-as-Judge analyzer...")
    judge = create_template_judge()

    # Analyze template
    print("\n🔬 Analyzing template quality...")

    # Simulate analysis results (in real use, this would call the LLM)
    print("\n📊 Analysis Results:")
    print(f"   Template: {template.name}")
    print(f"   Version: {template.version}")
    print()

    print("📈 Quality Scores:")
    print("   • Clarity:       ⭐⭐⭐⭐☆ (0.85/1.0)")
    print("     Clear module structure, good naming conventions")
    print()
    print("   • Structure:     ⭐⭐⭐⭐⭐ (0.92/1.0)")
    print("     Excellent use of inheritance, logical module ordering")
    print()
    print("   • Efficiency:    ⭐⭐⭐☆☆ (0.72/1.0)")
    print("     Some redundancy in greetings, could be more concise")
    print()
    print("   • Completeness:  ⭐⭐⭐⭐☆ (0.88/1.0)")
    print("     Covers main scenarios well, could add error handling")
    print()
    print("   Overall Score: 0.84/1.0 ✅")

    print("\n💪 Strengths:")
    print("   • Well-organized module structure with clear priorities")
    print("   • Good use of inheritance from base templates")
    print("   • Appropriate use of conditional rendering")
    print("   • Clear separation of concerns")

    print("\n🔧 Areas for Improvement:")
    print("   • Reduce redundancy in greeting modules")
    print("   • Add more specific error handling modules")
    print("   • Consider consolidating similar modules")
    print("   • Optimize for token efficiency in product descriptions")


def demo_optimization_suggestions():
    """Demo template optimization suggestions."""
    print("\n\n💡 Template Optimization Suggestions")
    print("=" * 50)
    print()

    print("🎯 Optimization Goals:")
    print("   • Reduce token usage by 20%")
    print("   • Improve response clarity")
    print("   • Enhance error handling")
    print()

    print("📝 Module Recommendations:")
    print()
    print("1. Merge modules 'example_shop_branding' and 'greeting':")
    print("   ```yaml")
    print("   - name: 'welcome'")
    print("     priority: 5")
    print("     template: |")
    print("       🛏️ 示例商店官方旗舰店 - 让每个人都拥有好睡眠")
    print("       您好，欢迎光临！我是您的专属睡眠顾问。")
    print("   ```")
    print("   💡 Saves ~15 tokens per interaction")
    print()

    print("2. Add error handling module:")
    print("   ```yaml")
    print("   - name: 'error_handler'")
    print("     priority: 95")
    print("     conditional: '{% if error_state %}'")
    print("     template: |")
    print("       抱歉，我暂时无法处理您的请求。")
    print("       请稍后再试或联系人工客服。")
    print("   ```")
    print("   💡 Improves robustness")
    print()

    print("3. Optimize product description module:")
    print("   ```yaml")
    print("   - name: 'product_series_context'")
    print("     priority: 17")
    print("     template: '您咨询的{{ product_series }}系列'")
    print("   ```")
    print("   💡 Reduces from 3 lines to 1, saves ~20 tokens")

    print("\n🔄 Priority Adjustments:")
    print("   • 'vip_recognition': 12 → 8 (move earlier)")
    print("   • 'quality_assurance': 80 → 70 (group with product info)")
    print("   • 'closing': 90 → 100 (ensure it's always last)")


def demo_template_improvement():
    """Demo automatic template improvement."""
    print("\n\n🚀 Automatic Template Improvement")
    print("=" * 50)
    print()

    print("🎯 Improvement Goals:")
    print("   • Reduce tokens by 25%")
    print("   • Improve clarity for VIP customers")
    print("   • Add multilingual support readiness")
    print()

    print("🔄 Generating improved template...")
    print()
    print("📝 Improved Template Preview:")
    print("```yaml")
    print("name: 'example_shop_customer_support_v2'")
    print("version: '2.0'")
    print("extends: 'chinese_ecommerce_support'")
    print()
    print("modules:")
    print("  - name: 'welcome_vip'")
    print("    priority: 5")
    print("    template: |")
    print("      🛏️ 示例商店官方旗舰店")
    print("      {% if vip_level %}尊贵的{{ vip_level }}会员，{% endif %}欢迎您！")
    print()
    print("  - name: 'product_info_compact'")
    print("    priority: 20")
    print("    conditional: '{% if product_series %}'")
    print("    template: '{{ product_series }}系列 - {{ product_features }}'")
    print()
    print("  - name: 'quick_benefits'")
    print("    priority: 30")
    print("    template: '✅ 15天退换 | 10年保修 | 全国联保'")
    print("```")
    print()
    print("✅ Changes Summary:")
    print("   • Merged greeting modules (saved 25 tokens)")
    print("   • Condensed product info (saved 40 tokens)")
    print("   • Simplified benefits display (saved 30 tokens)")
    print("   • Total reduction: ~35% fewer tokens")
    print("   • Maintained all functionality")
    print("   • Improved VIP recognition")


def demo_template_comparison():
    """Demo template version comparison."""
    print("\n\n📊 Template Version Comparison")
    print("=" * 50)
    print()

    print("Comparing: example_shop_customer_support v1.0 vs v2.0")
    print()

    print("📈 Comparison Results:")
    print()
    print("┌─────────────────┬──────────┬──────────┐")
    print("│ Metric          │ v1.0     │ v2.0     │")
    print("├─────────────────┼──────────┼──────────┤")
    print("│ Clarity         │ 0.85     │ 0.88 ↑   │")
    print("│ Structure       │ 0.92     │ 0.90 ↓   │")
    print("│ Efficiency      │ 0.72     │ 0.95 ↑   │")
    print("│ Completeness    │ 0.88     │ 0.86 ↓   │")
    print("│ Overall         │ 0.84     │ 0.90 ↑   │")
    print("└─────────────────┴──────────┴──────────┘")
    print()
    print("🏆 Recommendation: v2.0 is recommended")
    print("   • Significantly more token-efficient (23% reduction)")
    print("   • Slightly better clarity with VIP handling")
    print("   • Minor trade-off in structure complexity")
    print("   • Suitable for high-volume customer support")


def demo_model_specific_optimization():
    """Demo LLM-specific optimizations."""
    print("\n\n🤖 Model-Specific Optimizations")
    print("=" * 50)
    print()

    print("🎯 Optimizing for different LLMs:")
    print()

    print("📌 GPT-4 Optimizations:")
    print("   • Use detailed module descriptions")
    print("   • Leverage complex conditional logic")
    print("   • Can handle longer context windows")
    print()

    print("📌 Claude Optimizations:")
    print("   • Structure with clear XML-like tags")
    print("   • Use explicit section markers")
    print("   • Excellent for Chinese language support")
    print()

    print("📌 Llama Optimizations:")
    print("   • Keep modules concise and focused")
    print("   • Use simple, direct language")
    print("   • Minimize complex conditionals")
    print()

    print("📌 Qwen Optimizations:")
    print("   • Native Chinese language patterns")
    print("   • Cultural context awareness")
    print("   • Optimized for e-commerce scenarios")


def main():
    """Run the LLM-as-Judge demo."""
    print("🧑‍⚖️ LLM-as-Judge Template Analysis System")
    print("=" * 70)
    print()

    try:
        # Run all demos
        demo_template_analysis()
        demo_optimization_suggestions()
        demo_template_improvement()
        demo_template_comparison()
        demo_model_specific_optimization()

        print("\n\n✅ LLM-as-Judge demo completed!")
        print("\n🎯 Key Features Demonstrated:")
        print("   • Comprehensive template quality analysis")
        print("   • Actionable optimization suggestions")
        print("   • Automatic template improvement")
        print("   • Version comparison and A/B testing")
        print("   • Model-specific optimizations")
        print()
        print("💡 Benefits:")
        print("   • Data-driven template optimization")
        print("   • Continuous improvement process")
        print("   • Reduced token usage while maintaining quality")
        print("   • Better performance across different LLMs")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
