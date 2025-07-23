#!/usr/bin/env python3
"""
🤝 Perfect Harmony in Action - Minimal Code Example

This is the exact code example showing how template validation 
and DSPy work together harmoniously with minimal setup.
"""

from dspy_kit.templates import PromptTemplate, create_dspy_signature, create_dspy_module

def perfect_harmony_example():
    """The exact 'Perfect Harmony in Action' code."""
    
    # Chinese e-commerce template with validation
    template_yaml = '''---
name: "harmony_ecommerce_demo"
version: "1.0" 
domain: "customer_support"
language: "zh-CN"

input_schema:
  customer_query:
    type: "string"
    required: true
    description: "客户询问内容"
  shop_name:
    type: "string" 
    required: true
    description: "店铺名称"

output_schema:
  response:
    type: "string"
    description: "客服回复"
  confidence:
    type: "number"
    description: "置信度 0-1"

tools:
  - "get_product_info"
  - "get_shop_activities"
---

您好，欢迎来到{{ shop_name }}！

客户询问：{{ customer_query }}

{% if has_get_product_info %}
我可以为您查询商品信息。
{% endif %}

{% if has_get_shop_activities %}  
我可以为您查询活动优惠。
{% endif %}

请稍等，我来帮您处理...'''

    print("🤝 PERFECT HARMONY IN ACTION")
    print("=" * 50)
    
    # 1. Create template with our validation
    print("🔍 Step 1: Create Template + Validate")
    template = PromptTemplate.from_string(template_yaml)
    result = template.validate()  # Catches issues early ✅
    
    print(f"   Validation: {result.summary()}")
    
    # 2. Only if validation passes, create DSPy components  
    if result.is_valid:
        print("\n⚡ Step 2: Create DSPy Components")
        
        # Clean structure guaranteed by validation
        signature = create_dspy_signature(template)  
        print(f"   Signature: {signature.__name__} ✅")
        
        # Tool integration seamlessly handled
        module = create_dspy_module(template)        
        print(f"   Module: {module.__name__} ✅")
        
        print("\n🚀 Step 3: Ready for DSPy Execution")
        print("   # DSPy handles runtime robustness:")
        print("   # - Type checking")
        print("   # - JSON fallback if needed") 
        print("   # - Auto retry with format corrections")
        print("   # - Tool execution")
        
        print("\n   # Example usage:")
        print('   # response = module(')
        print('   #     customer_query="床垫有优惠吗？",')
        print('   #     shop_name="示例商店官方旗舰店"')
        print('   # )')
        
        return True
    else:
        print("❌ Validation failed - fix issues before DSPy creation")
        for error in result.errors:
            print(f"   Error: {error.message}")
        return False

def show_harmony_benefits():
    """Show the key benefits of this harmonious approach."""
    print("\n💡 HARMONY BENEFITS")
    print("=" * 50)
    
    benefits = [
        "💰 Cost Effective: Catch errors before expensive LLM calls",
        "⚡ Fast Development: No need to run full LLM pipeline for validation", 
        "🔒 Higher Success Rate: Pre-validated templates work better with DSPy",
        "🏪 Domain Aware: Chinese e-commerce + DSPy LLM integration",
        "🔧 Tool Ready: Validated dependencies work seamlessly",
        "🤝 Complementary: Two systems enhance each other"
    ]
    
    for benefit in benefits:
        print(f"   {benefit}")

def show_two_layer_architecture():
    """Explain the two-layer validation architecture."""
    print("\n🏗️ TWO-LAYER ARCHITECTURE")
    print("=" * 50)
    
    print("🔍 LAYER 1: Template Development Time (Our System)")
    layer1 = [
        "📄 YAML structure validation",
        "🎨 Jinja2 template syntax checking", 
        "🔗 Variable cross-references (template ↔ schema)",
        "🔧 Tool dependency validation",
        "🏪 Domain-specific rules (Chinese e-commerce)",
        "📋 Best practices recommendations"
    ]
    for item in layer1:
        print(f"   {item}")
    
    print("\n⚡ LAYER 2: Runtime Execution (DSPy System)")
    layer2 = [
        "🎯 Type checking during module execution",
        "📦 JSON mode fallback for LLM failures",
        "🔄 Automatic format correction and retry", 
        "✅ Output structure validation",
        "🛡️ Pydantic model validation",
        "📊 Response format enforcement"
    ]
    for item in layer2:
        print(f"   {item}")
    
    print("\n💡 Key Insight: Our validation catches issues BEFORE runtime!")

def main():
    """Run the perfect harmony demonstration."""
    success = perfect_harmony_example()
    
    if success:
        show_harmony_benefits()
        show_two_layer_architecture()
        
        print("\n🎉 PERFECT HARMONY ACHIEVED!")
        print("Template validation + DSPy = Better, faster, cheaper LLM apps")

if __name__ == "__main__":
    main()