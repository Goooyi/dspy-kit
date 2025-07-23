#!/usr/bin/env python3
"""
Demo: MCP Integration with Template System

This demo shows how to use MCP servers with templates for dynamic knowledge population.
It demonstrates:
1. Setting up an MCP server connection
2. Discovering tools from the MCP server
3. Using MCP tools in templates for real-time data
"""

import asyncio
import sys
from pathlib import Path

# Add dspy_kit to path
sys.path.insert(0, str(Path(__file__).parent / "dspy_kit"))

import os

os.chdir(str(Path(__file__).parent))

from dspy_kit.templates import InheritablePromptTemplate
from dspy_kit.templates.integrations.mcp_client import create_mcp_integration
from dspy_kit.templates.core.tools import create_default_tool_registry


async def demo_mcp_integration():
    """Demo MCP integration with templates."""
    print("🔌 MCP Integration Demo")
    print("=" * 50)
    print()

    # Create tool registry
    tool_registry = create_default_tool_registry()

    # Create MCP integration
    mcp_integration = create_mcp_integration(tool_registry)

    # Add e-commerce MCP server
    print("📡 Adding MCP e-commerce server...")
    mcp_integration.add_mcp_server(
        name="ecommerce",
        command="python",
        args=["examples/mcp_servers/ecommerce_server.py"],
        description="E-commerce data provider",
    )

    # Note: In a real scenario, you would start the MCP server separately
    # For this demo, we'll simulate the tool discovery
    print("🔍 Simulating tool discovery from MCP server...")

    # Simulate discovered tools
    discovered_tools = {
        "ecommerce.get_product_info": "获取商品详细信息",
        "ecommerce.get_shop_activities": "获取店铺促销活动",
        "ecommerce.check_inventory": "检查商品库存",
        "ecommerce.calculate_vip_price": "计算VIP会员价格",
    }

    print("\n✅ Discovered MCP tools:")
    for tool_name, description in discovered_tools.items():
        print(f"   • {tool_name}: {description}")

    # Create a template that uses MCP tools
    print("\n📝 Creating template with MCP tool configuration...")

    template_content = """---
name: "mcp_enabled_customer_support"
extends: "chinese_ecommerce_support"
version: "1.0"

# MCP tools configuration
tools:
  - "ecommerce.get_product_info"
  - "ecommerce.get_shop_activities"
  - "ecommerce.check_inventory"
  - "ecommerce.calculate_vip_price"

# Additional modules that use MCP data
modules:
  - name: "dynamic_product_info"
    priority: 20
    conditional: "{% if product_id and has_ecommerce_get_product_info %}"
    template: |
      📦 商品实时信息：
      [调用 ecommerce.get_product_info 获取商品 {{ product_id }} 的最新信息]
    description: "Dynamic product information from MCP"

  - name: "live_inventory_status"
    priority: 25
    conditional: "{% if product_id and has_ecommerce_check_inventory %}"
    template: |
      📊 库存状态：
      [调用 ecommerce.check_inventory 检查商品 {{ product_id }} 的实时库存]
    description: "Real-time inventory check"

  - name: "current_promotions"
    priority: 30
    conditional: "{% if shop_id and has_ecommerce_get_shop_activities %}"
    template: |
      🎉 当前优惠活动：
      [调用 ecommerce.get_shop_activities 获取 {{ shop_id }} 的最新促销信息]
    description: "Live promotional activities"

  - name: "vip_pricing"
    priority: 35
    conditional: "{% if product_id and vip_level and has_ecommerce_calculate_vip_price %}"
    template: |
      💎 VIP专享价格：
      [调用 ecommerce.calculate_vip_price 计算您的专属价格]
    description: "VIP member pricing calculation"

metadata:
  description: "MCP-enabled customer support template with dynamic data"
  mcp_servers: ["ecommerce"]
---

{% for module in modules %}
{{ module.template }}
{% endfor %}"""

    # Save the template
    template_path = Path("templates/demos/mcp_enabled_support.yaml")
    template_path.parent.mkdir(parents=True, exist_ok=True)

    with open(template_path, "w", encoding="utf-8") as f:
        f.write(template_content)

    print(f"✅ Created MCP-enabled template: {template_path}")

    # Load and render the template
    print("\n🚀 Loading and rendering MCP-enabled template...")

    template = InheritablePromptTemplate.from_file(str(template_path))

    # Render with sample data
    rendered = template.render(
        shop_name="示例商店官方旗舰店",
        shop_id="shop_001",
        customer_query="我想了解PROD-A-001这款产品的价格和库存",
        product_id="PROD-A-001",
        vip_level="gold",
        product_series="智能系列",
        # MCP tool availability flags (would be set automatically by MCP integration)
        has_ecommerce_get_product_info=True,
        has_ecommerce_check_inventory=True,
        has_ecommerce_get_shop_activities=True,
        has_ecommerce_calculate_vip_price=True,
    )

    print("\n📄 Rendered Output:")
    print("-" * 50)
    print(rendered)
    print("-" * 50)

    print("\n🎯 Key Benefits of MCP Integration:")
    print("   • Real-time data access without static injection")
    print("   • Dynamic tool discovery from MCP servers")
    print("   • Seamless integration with template system")
    print("   • Conditional rendering based on tool availability")
    print("   • Scalable to multiple MCP servers and tools")


def demo_mcp_tool_invocation():
    """Demo how MCP tools would be invoked in practice."""
    print("\n\n🔧 MCP Tool Invocation Example")
    print("=" * 50)
    print()

    print("In practice, when the LLM processes the template output:")
    print()
    print("1. LLM sees: '[调用 ecommerce.get_product_info 获取商品 PROD-A-001 的最新信息]'")
    print()
    print("2. LLM invokes the MCP tool:")
    print("   ```")
    print("   result = await mcp_client.invoke_tool(")
    print('       "ecommerce.get_product_info",')
    print('       product_id="PROD-A-001"')
    print("   )")
    print("   ```")
    print()
    print("3. MCP server returns:")
    print("   ```")
    print("   商品信息：")
    print("   名称：高级智能产品A - 豪华版")
    print("   价格：¥3999 (原价：¥4999)")
    print("   描述：采用先进材料技术，提供最佳使用体验")
    print("   特点：智能调节, 舒适支撑, 环保材料, 10年保修")
    print("   库存：50件")
    print("   ```")
    print()
    print("4. LLM incorporates the real-time data into its response")


def main():
    """Run the MCP integration demo."""
    try:
        # Run async demo
        asyncio.run(demo_mcp_integration())

        # Show tool invocation example
        demo_mcp_tool_invocation()

        print("\n✅ MCP integration demo completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
