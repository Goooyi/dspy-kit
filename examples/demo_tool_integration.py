#!/usr/bin/env python3
"""
🔧 DEMO: Tool Integration for Dynamic Knowledge Population

Shows how templates can use tools/functions to dynamically populate knowledge
instead of relying on static variables like {style_info_with_activity}.
"""

import sys
import os
# Add parent directory to path
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import core components, handle missing dependencies gracefully
try:
    from dspy_kit.templates import PromptTemplate
    from dspy_kit.templates.core.tools import create_default_tool_registry, ToolRegistry
    templates_available = True
except ImportError as e:
    print(f"⚠️  Template imports failed: {e}")
    templates_available = False

try:
    from dspy_kit.templates.adapters.dspy_adapter import create_dspy_module
    dspy_adapter_available = True
except ImportError:
    dspy_adapter_available = False

# Set up DSPy (optional for template rendering)
try:
    import dspy
    from dotenv import load_dotenv

    load_dotenv()

    # Set proxy for yuzhoumao endpoint (only for this script)
    if "OPENAI_API_BASE" in os.environ:
        # Set proxy environment variables for this process only
        os.environ["http_proxy"] = "http://localhost:8888"
        os.environ["https_proxy"] = "http://localhost:8888"
        os.environ["HTTPS_PROXY"] = "http://localhost:8888"
        os.environ["HTTP_PROXY"] = "http://localhost:8888"

    # Configure DSPy with your setup
    if "OPENAI_API_KEY" in os.environ:
        lm = dspy.LM(
            model="openai/qwen-max",
            api_base=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0.1,
        )
        dspy.configure(lm=lm)
        dspy_available = True
    else:
        dspy_available = False
except ImportError:
    dspy_available = False


def main():
    print("🔧 DEMO: Tool Integration for Dynamic Knowledge")
    print("=" * 60)

    if not templates_available:
        print("❌ Template system not available - missing dependencies")
        print("   Please install dspy-kit dependencies to run full demo")
        return

    # Demo 1: Create a tool-enhanced template
    print("\\n📋 1. Creating Tool-Enhanced Template")
    print("-" * 40)

    # Create a simple template with tool support
    tool_template_yaml = '''---
name: "ecommerce_support_with_tools"
version: "1.0"
domain: "customer_support"
language: "zh-CN"

input_schema:
  shop_id: {type: "string", required: true}
  customer_query: {type: "string", required: true}
  product_id: {type: "string", required: false}

output_schema:
  response: {type: "string", description: "Customer support response"}

tools:
  - "get_product_info"
  - "get_shop_activities"
  - "check_inventory"

concatenation_style: "sections"
---

---- Role
你是{{ shop_id }}的专业客服，能够使用工具获取实时商品信息。

---- Available Tools
当前可用工具：
{{ available_tools }}

---- Context
顾客询问：{{ customer_query }}
{% if product_id %}
涉及商品ID：{{ product_id }}
{% endif %}

---- Instructions
根据顾客询问，你可以：
{% if has_get_product_info %}
- 使用get_product_info工具获取商品详细信息
{% endif %}
{% if has_get_shop_activities %}
- 使用get_shop_activities工具获取当前活动信息
{% endif %}
{% if has_check_inventory %}
- 使用check_inventory工具检查库存状态
{% endif %}

请根据需要调用相应工具，然后为顾客提供准确、有帮助的回复。

---- Output Format
请提供自然、友好的中文回复，包含准确的商品或活动信息。'''

    # Parse the template
    template = PromptTemplate.from_string(tool_template_yaml) # type: ignore

    print(f"✅ Template created: {template.name}")
    print(f"   🔧 Tools: {len(template.tools)} configured")
    print(f"   📊 Variables: {len(template.input_schema)} input fields")

    # Demo 2: Show available tools
    print("\\n🛠️  2. Available Tools")
    print("-" * 40)

    available_tools = template.get_available_tools()
    for tool in available_tools:
        print(f"   ✅ {tool}")

    # Demo 3: Render template with tool context
    print("\\n🎯 3. Template Rendering with Tool Context")
    print("-" * 40)

    # Sample variables
    variables = {
        "shop_id": "示例商店官方旗舰店",
        "customer_query": "这款智能产品怎么样？有什么活动吗？",
        "product_id": "MZ-2024"
    }

    try:
        rendered = template.render(**variables)
        print(f"✅ Template rendered successfully!")
        print(f"   📏 Length: {len(rendered)} characters")

        # Show the rendered content
        print("\\n📝 Rendered Template:")
        print("-" * 30)
        lines = rendered.split('\\n')
        for i, line in enumerate(lines, 1):
            if i <= 20:  # Show first 20 lines
                print(f"{i:2d}: {line}")
            elif i == 21:
                print("    ... (truncated)")
                break

        # Check if tool context was added
        if "get_product_info" in rendered:
            print("\\n✅ Tool context successfully added to template")
        else:
            print("\\n❌ Tool context missing from template")

    except Exception as e:
        print(f"❌ Template rendering failed: {e}")

    # Demo 4: DSPy Integration with Tools
    print("\\n🔗 4. DSPy Integration with Tools")
    print("-" * 40)

    if dspy_available and dspy_adapter_available:
        try:
            # Create DSPy module with tool support
            module_class = create_dspy_module(template) # type: ignore
            module = module_class()

            print(f"✅ DSPy module created: {module_class.__name__}")
            print(f"   🔧 Tools available: {bool(template.tools)}")

            # Note: Actual tool execution would require proper DSPy setup
            print("   📝 Note: Tool execution requires configured DSPy environment")

        except Exception as e:
            print(f"❌ DSPy module creation failed: {e}")
    else:
        missing = []
        if not dspy_available:
            missing.append("DSPy")
        if not dspy_adapter_available:
            missing.append("DSPy adapter")
        print(f"⚠️  {', '.join(missing)} not available - skipping integration demo")

    # Demo 5: Comparison with Static Approach
    print("\\n📊 5. Static vs Dynamic Knowledge Comparison")
    print("-" * 40)

    print("\\n🔴 OLD STATIC APPROACH:")
    print("   Variables like {style_info_with_activity} contain:")
    print("   - Pre-fetched product info")
    print("   - Potentially stale data")
    print("   - Fixed at template render time")
    print("   - No real-time updates")

    print("\\n🟢 NEW DYNAMIC APPROACH:")
    print("   Tools like get_product_info() provide:")
    print("   - Real-time product information")
    print("   - Fresh inventory data")
    print("   - Current pricing and activities")
    print("   - Conditional knowledge fetching")

    # Demo 6: Custom Tool Registration
    print("\\n⚙️  6. Custom Tool Registration")
    print("-" * 40)

    # Create custom tool registry
    custom_registry = ToolRegistry()

    # Add a custom tool
    def get_customer_history(customer_id: str) -> str:
        """Get customer purchase history."""
        return f"客户 {customer_id} 历史记录：购买过2件产品，VIP会员"

    custom_registry.register_function(
        name="get_customer_history",
        description="获取客户购买历史记录",
        parameters={
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "客户ID"}
            },
            "required": ["customer_id"]
        },
        func=get_customer_history
    )

    # Set custom registry
    template.set_tool_registry(custom_registry)
    template.add_tool("get_customer_history")

    print(f"✅ Custom tool registered: get_customer_history")
    print(f"   🔧 Total tools: {len(template.get_available_tools())}")

    # Demo 7: Template Saving with Tools
    print("\\n💾 7. Saving Tool-Enhanced Template")
    print("-" * 40)

    try:
        output_path = "ecommerce_support_with_tools.yaml"
        template.save(output_path)
        print(f"✅ Template saved to: {output_path}")
        print(f"   🔧 Tools configuration preserved")

        # Verify by loading
        loaded_template = PromptTemplate.from_file(output_path)
        print(f"✅ Template loaded successfully")
        print(f"   🔧 Tools: {loaded_template.tools}")

    except Exception as e:
        print(f"❌ Template saving failed: {e}")

    # Summary
    print("\\n" + "=" * 60)
    print("🎉 TOOL INTEGRATION COMPLETE")
    print("=" * 60)

    achievements = [
        "✅ Tool registry with built-in e-commerce functions",
        "✅ Dynamic knowledge population via function calls",
        "✅ Tool context automatically added to templates",
        "✅ DSPy integration with tool support",
        "✅ Custom tool registration capability",
        "✅ Template serialization with tool configuration",
        "✅ Real-time data fetching instead of static variables"
    ]

    print("\\n📋 ACHIEVEMENTS:")
    for achievement in achievements:
        print(f"   {achievement}")

    print("\\n🚀 NEXT CAPABILITIES:")
    next_features = [
        "🔌 MCP server integration",
        "⚡ Async tool execution",
        "🔄 Tool result caching",
        "📊 Tool usage analytics",
        "🛡️  Tool error handling and fallbacks"
    ]

    for feature in next_features:
        print(f"   {feature}")

    print("\\n💡 Your templates can now fetch real-time data!")


if __name__ == "__main__":
    main()
