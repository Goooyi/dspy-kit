#!/usr/bin/env python3
"""
🤝 DEMO: DSPy Harmony - Template Validation + DSPy Schema Integration

Shows how our template validation works harmoniously with DSPy's built-in:
1. Type checking and schema validation
2. ChatAdapter and other adapters
3. JSON mode fallback
4. Signature validation

Our validation complements DSPy by catching issues BEFORE runtime.
"""

from dspy_kit.templates import PromptTemplate, create_dspy_signature, create_dspy_module

try:
    import dspy
    from dotenv import load_dotenv

    load_dotenv()
    dspy_available = True
except ImportError:
    dspy_available = False


def demo_dspy_schema_system():
    """Demo 1: Understanding DSPy's Built-in Schema System"""
    print("🤝 Demo 1: DSPy's Built-in Schema System")
    print("=" * 50)

    if not dspy_available:
        print("❌ DSPy not available - showing conceptual overview")

    print("📋 DSPy's Schema Features:")
    dspy_features = [
        "🔍 Runtime type checking via dspy.InputField/OutputField",
        "🎯 ChatAdapter for chat-based input/output formatting",
        "📦 JSON mode fallback when structured output fails",
        "🔄 Automatic retry with format corrections",
        "✅ Signature validation at module creation",
        "🛡️ Pydantic integration for complex types",
    ]

    for feature in dspy_features:
        print(f"   {feature}")

    print("\n📝 Example DSPy Signature:")
    dspy_example = '''
class CustomerSupportSignature(dspy.Signature):
    """Customer support response generation."""

    customer_query: str = dspy.InputField(desc="Customer question")
    shop_name: str = dspy.InputField(desc="Name of the shop")

    response: str = dspy.OutputField(desc="Helpful response")
    confidence: float = dspy.OutputField(desc="Confidence score 0-1")

# DSPy handles:
# - Type validation (str, float)
# - Field descriptions
# - Runtime checking
# - JSON fallback if structured output fails
'''
    print(dspy_example)


def demo_validation_layers():
    """Demo 2: Two-Layer Validation Architecture"""
    print("\n🤝 Demo 2: Two-Layer Validation Architecture")
    print("=" * 50)

    print("🏗️ Validation happens at TWO levels:")
    print()

    print("🔍 LAYER 1: Template Development Time (Our System)")
    template_validation = [
        "📄 YAML structure validation",
        "🎨 Jinja2 template syntax checking",
        "🔗 Variable cross-references (template ↔ schema)",
        "🔧 Tool dependency validation",
        "🏪 Domain-specific rules (Chinese e-commerce)",
        "📋 Best practices recommendations",
    ]

    for item in template_validation:
        print(f"   {item}")

    print("\n⚡ LAYER 2: Runtime Execution (DSPy System)")
    dspy_validation = [
        "🎯 Type checking during module execution",
        "📦 JSON mode fallback for LLM failures",
        "🔄 Automatic format correction and retry",
        "✅ Output structure validation",
        "🛡️ Pydantic model validation",
        "📊 Response format enforcement",
    ]

    for item in dspy_validation:
        print(f"   {item}")

    print("\n💡 Key Insight: Our validation catches issues BEFORE they become runtime problems!")


def demo_harmonious_integration():
    """Demo 3: Harmonious Integration Example"""
    print("\n🤝 Demo 3: Harmonious Integration")
    print("=" * 50)

    # Create template with our validation
    template_yaml = """---
name: "validated_customer_support"
version: "1.0"
domain: "customer_support"
language: "zh-CN"

input_schema:
  customer_query:
    type: "string"
    required: true
    description: "Customer question or request"
  shop_name:
    type: "string"
    required: true
    description: "Name of the shop"
  urgency_level:
    type: "string"
    required: false
    description: "Urgency level: low, medium, high"
    enum: ["low", "medium", "high"]

output_schema:
  response:
    type: "string"
    description: "Customer service response in Chinese"
  confidence:
    type: "number"
    description: "Confidence score between 0 and 1"
  category:
    type: "string"
    description: "Response category"

tools:
  - "get_product_info"
  - "get_shop_activities"

metadata:
  description: "Validated template for Chinese customer support"
  author: "Template Validation System"
---

您好，欢迎来到{{ shop_name }}！

客户询问：{{ customer_query }}

{% if urgency_level %}
紧急程度：{{ urgency_level }}
{% endif %}

{% if has_get_product_info %}
💡 我可以为您查询商品详细信息
{% endif %}

{% if has_get_shop_activities %}
🎉 我可以为您查询最新优惠活动
{% endif %}

请稍等，我来为您提供准确的帮助..."""

    # Step 1: Template-level validation
    print("🔍 Step 1: Template-Level Validation")
    template = PromptTemplate.from_string(template_yaml)
    validation_result = template.validate()

    print(f"   📊 Our Validation: {validation_result.summary()}")

    if validation_result.errors:
        print("   ❌ Template has errors - fix before creating DSPy module!")
        return

    if validation_result.warnings:
        print("   ⚠️ Template has warnings - consider improvements")

    print("   ✅ Template passes validation - safe to create DSPy module")

    # Step 2: DSPy integration
    print("\n⚡ Step 2: DSPy Integration")

    if dspy_available:
        try:
            # Create DSPy signature from validated template
            signature_class = create_dspy_signature(template)
            print(f"   ✅ DSPy Signature created: {signature_class.__name__}")

            # Show the signature fields
            annotations = getattr(signature_class, "__annotations__", {})
            print(f"   📋 Fields: {list(annotations.keys())}")

            # Create DSPy module
            module_class = create_dspy_module(template)
            print(f"   ✅ DSPy Module created: {module_class.__name__}")

            print("   🤝 Both systems working together harmoniously!")

        except Exception as e:
            print(f"   ❌ DSPy integration failed: {e}")
            print("   💡 This is where our pre-validation helps catch issues early!")
    else:
        print("   ⚠️ DSPy not available - showing conceptual integration")
        print("   ✅ Template validated ✓ → DSPy Signature ✓ → DSPy Module ✓")


def demo_error_prevention():
    """Demo 4: Error Prevention - Template vs Runtime"""
    print("\n🤝 Demo 4: Error Prevention Comparison")
    print("=" * 50)

    print("🚨 Without Template Validation:")
    without_validation = [
        "❌ Runtime errors when template has undefined variables",
        "❌ DSPy signature creation fails with cryptic errors",
        "❌ Type mismatches discovered only during execution",
        "❌ Missing tool dependencies cause runtime failures",
        "❌ Debugging requires running the full LLM pipeline",
        "💸 Wasted API calls and time during development",
    ]

    for issue in without_validation:
        print(f"   {issue}")

    print("\n✅ With Template Validation:")
    with_validation = [
        "✅ Undefined variables caught at development time",
        "✅ Schema mismatches detected before DSPy creation",
        "✅ Type consistency validated in template definition",
        "✅ Tool dependencies verified during validation",
        "✅ Fast feedback loop without LLM calls",
        "💰 No wasted API costs during template development",
    ]

    for benefit in with_validation:
        print(f"   {benefit}")


def demo_dspy_fallback_harmony():
    """Demo 5: DSPy JSON Fallback Harmony"""
    print("\n🤝 Demo 5: DSPy JSON Fallback Integration")
    print("=" * 50)

    print("🔄 How Our Validation Improves DSPy's JSON Fallback:")

    fallback_benefits = [
        "📋 Well-defined output_schema → Better JSON structure hints",
        "🏷️ Clear field descriptions → Better LLM understanding",
        "🎯 Type specifications → Cleaner JSON parsing",
        "✅ Pre-validated structure → Higher success rate",
        "🔧 Tool context → Better structured responses",
        "📊 Consistent templates → More predictable outputs",
    ]

    for benefit in fallback_benefits:
        print(f"   {benefit}")

    print("\n💡 Result: When DSPy falls back to JSON mode, it has:")
    print("   • Cleaner templates with validated structure")
    print("   • Better type hints from our schema validation")
    print("   • More consistent variable naming")
    print("   • Validated tool dependencies")

    # Show example of improved JSON structure
    print("\n📝 Example Improved JSON Structure:")
    json_example = """
{
  "response": "您好！根据您的询问，这款智能床垫...",
  "confidence": 0.95,
  "category": "product_inquiry"
}

# Our validation ensures:
# ✅ All fields defined in output_schema
# ✅ Correct types (string, number)
# ✅ Meaningful field names
# ✅ Consistent structure across templates
"""
    print(json_example)


def demo_integration_workflow():
    """Demo 6: Recommended Integration Workflow"""
    print("\n🤝 Demo 6: Recommended Development Workflow")
    print("=" * 50)

    print("🔄 Recommended Development Workflow:")

    workflow_steps = [
        "1️⃣ Create template with YAML frontmatter + Jinja2 content",
        "2️⃣ Run template.validate() to catch early issues",
        "3️⃣ Fix validation errors and warnings",
        "4️⃣ Create DSPy signature from validated template",
        "5️⃣ Create DSPy module with tool integration",
        "6️⃣ Test with real LLM (DSPy handles runtime validation)",
        "7️⃣ Iterate: Update template → Validate → Test → Deploy",
    ]

    for step in workflow_steps:
        print(f"   {step}")

    print("\n🎯 Benefits of This Workflow:")
    benefits = [
        "⚡ Fast development cycle with early error detection",
        "🔒 High confidence in template quality before testing",
        "💰 Reduced API costs during development",
        "🤝 Seamless integration with DSPy's runtime features",
        "📊 Consistent template structure across projects",
        "🏪 Domain-specific validation (e.g., Chinese e-commerce)",
    ]

    for benefit in benefits:
        print(f"   {benefit}")


def main():
    print("🤝 DSPY HARMONY: Template Validation + DSPy Integration")
    print("=" * 60)

    # Demo the harmonious integration
    demo_dspy_schema_system()
    demo_validation_layers()
    demo_harmonious_integration()
    demo_error_prevention()
    demo_dspy_fallback_harmony()
    demo_integration_workflow()

    print("\n" + "=" * 60)
    print("🎉 HARMONY ACHIEVED: Best of Both Worlds")
    print("=" * 60)

    harmony_summary = [
        "🔍 Template Validation: Early error detection & structure validation",
        "⚡ DSPy Runtime: Type checking, JSON fallback, format correction",
        "🤝 Perfect Complement: Pre-validation + Runtime robustness",
        "🏪 Domain Awareness: Chinese e-commerce + DSPy's LLM integration",
        "🔧 Tool Integration: Template tools + DSPy execution",
        "💰 Cost Effective: Catch errors before expensive LLM calls",
        "🚀 Developer Experience: Fast feedback + Robust execution",
    ]

    for point in harmony_summary:
        print(f"   {point}")

    print("\n💡 Key Insight:")
    print("   Our validation doesn't replace DSPy's schema system -")
    print("   it makes DSPy's system work BETTER by providing")
    print("   cleaner, pre-validated templates with consistent structure!")


if __name__ == "__main__":
    main()
