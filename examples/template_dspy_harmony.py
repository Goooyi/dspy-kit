#!/usr/bin/env python3
"""
🤝 Template + DSPy Perfect Harmony Example

This example demonstrates how dspy-kit's template validation system works 
harmoniously with DSPy's built-in schema validation and runtime features.

Key Concepts:
- Two-layer validation: Template-time + Runtime  
- Complementary systems that enhance each other
- Cost-effective development with early error detection
- Seamless integration from template to DSPy execution

Requirements:
- dspy-ai
- jsonschema
- jinja2
"""

import os
from typing import Optional
from pathlib import Path

# Import dspy-kit components
from dspy_kit.templates import PromptTemplate, create_dspy_signature, create_dspy_module
from dspy_kit.templates.validation.validator import TemplateValidator

# Import DSPy (handle gracefully if not available)
try:
    import dspy
    from dotenv import load_dotenv
    
    # Load environment if available
    load_dotenv()
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    print("⚠️  DSPy not available - install with: pip install dspy-ai")


class HarmonyExample:
    """Demonstrates the perfect harmony between template validation and DSPy."""
    
    def __init__(self):
        """Initialize the harmony example."""
        self.validator = TemplateValidator(strict_mode=False)
        self.templates = {}
        self.dspy_modules = {}
        
        # Configure DSPy if available
        if DSPY_AVAILABLE and self._has_api_config():
            self._configure_dspy()
    
    def _has_api_config(self) -> bool:
        """Check if API configuration is available."""
        return bool(os.environ.get("OPENAI_API_KEY"))
    
    def _configure_dspy(self):
        """Configure DSPy with available credentials."""
        try:
            # Set proxy for yuzhoumao endpoint if needed
            if "OPENAI_API_BASE" in os.environ:
                os.environ["http_proxy"] = "http://localhost:8888"
                os.environ["https_proxy"] = "http://localhost:8888"
            
            lm = dspy.LM(
                model="openai/qwen-max",
                api_base=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
                api_key=os.environ["OPENAI_API_KEY"],
                temperature=0.1,
            )
            dspy.configure(lm=lm)
            print("✅ DSPy configured successfully")
            
        except Exception as e:
            print(f"⚠️  DSPy configuration failed: {e}")
    
    def demo_layer_1_validation(self):
        """Demo Layer 1: Template Development-Time Validation."""
        print("🔍 LAYER 1: Template Development-Time Validation")
        print("=" * 60)
        
        # Example 1: Valid Chinese E-commerce Template
        print("\n📋 Example 1: Valid Chinese E-commerce Template")
        print("-" * 40)
        
        valid_template_yaml = '''---
name: "chinese_ecommerce_support"
version: "1.0"
domain: "e_commerce"
language: "zh-CN"

input_schema:
  customer_query:
    type: "string"
    required: true
    description: "客户的问题或请求"
  shop_name:
    type: "string"
    required: true
    description: "店铺名称"
  product_id:
    type: "string"
    required: false
    description: "商品ID（可选）"

output_schema:
  response:
    type: "string"
    description: "客服回复内容"
  confidence:
    type: "number"
    description: "回复置信度 (0-1)"
  requires_human:
    type: "boolean"
    description: "是否需要人工介入"

tools:
  - "get_product_info"
  - "get_shop_activities"
  - "check_inventory"

metadata:
  description: "中文电商客服模板，支持商品查询和活动咨询"
  author: "DSPy-Kit Template System"
  tags: ["ecommerce", "chinese", "customer_support"]
---

您好，欢迎来到{{ shop_name }}！

客户询问：{{ customer_query }}

{% if product_id %}
涉及商品：{{ product_id }}
{% endif %}

当前可用工具：
{{ available_tools }}

{% if has_get_product_info %}
💡 我可以为您查询商品详细信息
{% endif %}

{% if has_get_shop_activities %}
🎉 我可以为您查询最新优惠活动
{% endif %}

{% if has_check_inventory %}
📦 我可以为您查询库存状态
{% endif %}

请稍等，我来为您提供准确的帮助...'''

        # Create and validate template
        template = PromptTemplate.from_string(valid_template_yaml)
        result = template.validate()
        
        print(f"✅ Validation Result: {result.summary()}")
        
        if result.info:
            print("\nℹ️  Template Information:")
            for info in result.info[:3]:
                print(f"   • {info.message}")
        
        if result.warnings:
            print("\n⚠️  Warnings (Optional Improvements):")
            for warning in result.warnings:
                print(f"   • {warning.message}")
                if warning.suggestion:
                    print(f"     💡 {warning.suggestion}")
        
        # Store valid template for Layer 2
        self.templates["ecommerce"] = template
        
        return result.is_valid
    
    def demo_layer_2_dspy_integration(self):
        """Demo Layer 2: DSPy Runtime Integration."""
        print("\n⚡ LAYER 2: DSPy Runtime Integration")
        print("=" * 60)
        
        if "ecommerce" not in self.templates:
            print("❌ No validated template available - run Layer 1 first")
            return
        
        template = self.templates["ecommerce"]
        
        print("\n📋 Step 1: Create DSPy Signature from Validated Template")
        print("-" * 50)
        
        try:
            # Create DSPy signature from validated template
            signature_class = create_dspy_signature(template)
            
            print(f"✅ DSPy Signature Created: {signature_class.__name__}")
            print(f"   📝 Docstring: {signature_class.__doc__}")
            
            # Show signature fields
            annotations = getattr(signature_class, '__annotations__', {})
            print(f"   📋 Input Fields: {list(template.input_schema.keys())}")
            print(f"   📤 Output Fields: {list(template.output_schema.keys())}")
            
            # Show DSPy field details
            for field_name in annotations:
                field_obj = getattr(signature_class, field_name, None)
                if hasattr(field_obj, 'desc'):
                    print(f"      • {field_name}: {field_obj.desc}")
            
        except Exception as e:
            print(f"❌ Signature creation failed: {e}")
            return
        
        print("\n📋 Step 2: Create DSPy Module with Tool Support")
        print("-" * 50)
        
        try:
            # Create DSPy module from validated template
            module_class = create_dspy_module(template)
            module = module_class()
            
            print(f"✅ DSPy Module Created: {module_class.__name__}")
            print(f"   🔧 Tools Available: {bool(template.tools)}")
            print(f"   📊 Tool Count: {len(template.tools)}")
            print(f"   🎯 Module Type: {type(module).__name__}")
            
            # Store module for testing
            self.dspy_modules["ecommerce"] = module
            
        except Exception as e:
            print(f"❌ Module creation failed: {e}")
            return
        
        print("\n📋 Step 3: DSPy Runtime Features")
        print("-" * 50)
        
        dspy_features = [
            "🔍 Input/Output type checking via dspy.InputField/OutputField",
            "🎯 Automatic prompt formatting for Chinese content",
            "📦 JSON mode fallback if structured output fails",
            "🔄 Automatic retry with format corrections",
            "✅ Response structure validation",
            "🛡️ Pydantic integration for complex types",
            "🔧 Tool execution integration (when configured)"
        ]
        
        for feature in dspy_features:
            print(f"   {feature}")
    
    def demo_harmony_benefits(self):
        """Demo the specific benefits of the harmonious integration."""
        print("\n🤝 HARMONY BENEFITS: Template + DSPy Integration")
        print("=" * 60)
        
        print("\n💰 Cost-Effective Development")
        print("-" * 30)
        cost_benefits = [
            "🚫 No LLM calls needed for template validation",
            "⚡ Fast feedback loop during development",
            "🔧 Catch configuration errors before API costs",
            "📊 Validate complex templates without execution"
        ]
        for benefit in cost_benefits:
            print(f"   {benefit}")
        
        print("\n🔒 Higher Success Rate")
        print("-" * 30)
        quality_benefits = [
            "✅ Pre-validated templates have consistent structure",
            "🎯 Better type hints improve DSPy's JSON fallback",
            "🔗 Variable consistency reduces runtime errors", 
            "🛠️ Tool dependencies verified before execution"
        ]
        for benefit in quality_benefits:
            print(f"   {benefit}")
        
        print("\n🏪 Domain-Specific Intelligence")
        print("-" * 30)
        domain_benefits = [
            "🇨🇳 Chinese e-commerce template validation",
            "🛍️ E-commerce terminology verification",
            "🔧 Business-specific tool validation",
            "📋 Industry best practices enforcement"
        ]
        for benefit in domain_benefits:
            print(f"   {benefit}")
    
    def demo_error_prevention(self):
        """Demo how template validation prevents common DSPy errors."""
        print("\n🚨 ERROR PREVENTION: Before vs After")
        print("=" * 60)
        
        print("\n❌ WITHOUT Template Validation:")
        print("-" * 30)
        without_validation = [
            "💸 Runtime errors discovered during expensive LLM calls",
            "🔍 Cryptic DSPy signature creation failures",
            "⚙️ Tool dependency errors at execution time",
            "🔤 Type mismatches found only during testing",
            "🐛 Debugging requires full pipeline execution",
            "⏰ Slow development cycle with late feedback"
        ]
        for issue in without_validation:
            print(f"   {issue}")
        
        print("\n✅ WITH Template Validation:")
        print("-" * 30)
        with_validation = [
            "⚡ Fast validation without any LLM calls",
            "🔧 Clear error messages with helpful suggestions",
            "🛠️ Tool dependencies verified during development",
            "📊 Type consistency checked before DSPy creation",
            "🎯 Early feedback enables rapid iteration",
            "💰 Zero API costs during template development"
        ]
        for benefit in with_validation:
            print(f"   {benefit}")
    
    def demo_practical_workflow(self):
        """Demo the recommended practical workflow."""
        print("\n🔄 PRACTICAL WORKFLOW: Development to Production")
        print("=" * 60)
        
        workflow_steps = [
            {
                "step": "1️⃣ Create Template",
                "description": "Write YAML frontmatter + Jinja2 content",
                "tools": "Text editor, dspy-kit template format"
            },
            {
                "step": "2️⃣ Validate Template", 
                "description": "Run template.validate() for immediate feedback",
                "tools": "TemplateValidator, jsonschema"
            },
            {
                "step": "3️⃣ Fix Issues",
                "description": "Address errors and warnings with suggestions",
                "tools": "Validation error messages, best practices"
            },
            {
                "step": "4️⃣ Create DSPy Components",
                "description": "Generate signature and module from validated template",
                "tools": "create_dspy_signature, create_dspy_module"
            },
            {
                "step": "5️⃣ Test with DSPy",
                "description": "Execute with real LLM, DSPy handles runtime robustness",
                "tools": "DSPy runtime, LLM APIs, tool execution"
            },
            {
                "step": "6️⃣ Iterate & Deploy",
                "description": "Update template → validate → test → deploy cycle",
                "tools": "Version control, CI/CD pipelines"
            }
        ]
        
        for workflow in workflow_steps:
            print(f"\n{workflow['step']} {workflow['description']}")
            print(f"   🛠️ Tools: {workflow['tools']}")
    
    def demo_real_execution(self):
        """Demo real execution if DSPy is properly configured."""
        print("\n🚀 REAL EXECUTION DEMO")
        print("=" * 60)
        
        if not DSPY_AVAILABLE:
            print("❌ DSPy not available - install with: pip install dspy-ai")
            return
        
        if not self._has_api_config():
            print("⚠️  API configuration not available")
            print("   Set OPENAI_API_KEY environment variable for real execution")
            return
        
        if "ecommerce" not in self.dspy_modules:
            print("❌ No DSPy module available - run previous demos first")
            return
        
        module = self.dspy_modules["ecommerce"]
        
        print("\n📋 Test Case: Chinese E-commerce Customer Inquiry")
        print("-" * 50)
        
        # Test data
        test_data = {
            "customer_query": "这款智能床垫有什么优惠活动吗？价格怎么样？",
            "shop_name": "示例商店官方旗舰店",
            "product_id": "MZ-2024"
        }
        
        print("📝 Input:")
        for key, value in test_data.items():
            print(f"   {key}: {value}")
        
        try:
            print("\n⚡ Executing DSPy Module...")
            print("   (This would call the LLM with validated template)")
            
            # Note: Actual execution would be:
            # result = module(**test_data)
            # print(f"📤 Output: {result}")
            
            print("✅ DSPy would handle:")
            execution_features = [
                "🎯 Type checking for all inputs/outputs",
                "📦 JSON fallback if structured output fails",
                "🔄 Automatic retry with format corrections",
                "🔧 Tool execution for get_product_info, etc.",
                "🛡️ Response validation against output_schema"
            ]
            
            for feature in execution_features:
                print(f"   {feature}")
                
        except Exception as e:
            print(f"❌ Execution failed: {e}")
    
    def run_complete_demo(self):
        """Run the complete harmony demonstration."""
        print("🤝 TEMPLATE + DSPY PERFECT HARMONY")
        print("=" * 80)
        print("Demonstrating how dspy-kit template validation enhances DSPy")
        print("=" * 80)
        
        # Run all demos in sequence
        layer1_valid = self.demo_layer_1_validation()
        
        if layer1_valid:
            self.demo_layer_2_dspy_integration()
        
        self.demo_harmony_benefits()
        self.demo_error_prevention() 
        self.demo_practical_workflow()
        self.demo_real_execution()
        
        print("\n" + "=" * 80)
        print("🎉 HARMONY DEMONSTRATION COMPLETE")
        print("=" * 80)
        
        summary_points = [
            "🔍 Template validation catches errors before expensive LLM calls",
            "⚡ DSPy handles runtime robustness with type checking and fallbacks", 
            "🤝 Two systems complement each other perfectly",
            "💰 Cost-effective development with early feedback",
            "🏪 Domain-specific validation for Chinese e-commerce",
            "🔧 Seamless tool integration from template to execution",
            "🚀 Production-ready workflow from development to deployment"
        ]
        
        for point in summary_points:
            print(f"   {point}")
        
        print("\n💡 Key Takeaway:")
        print("   Template validation + DSPy = Better templates, faster development,")
        print("   lower costs, and more reliable LLM applications!")


def main():
    """Main entry point for the harmony demonstration."""
    # Create and run harmony example
    harmony = HarmonyExample()
    harmony.run_complete_demo()


if __name__ == "__main__":
    main()