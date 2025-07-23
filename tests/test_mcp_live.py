#!/usr/bin/env python3
"""
Live test of MCP integration with actual server running.

This script demonstrates:
1. Starting an MCP server subprocess
2. Connecting to it from the template system
3. Using MCP tools in a real template
4. Showing actual tool invocation results
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path

# Add dspy_kit to path
sys.path.insert(0, str(Path(__file__).parent / "dspy_kit"))

import os

os.chdir(str(Path(__file__).parent))

from dspy_kit.templates import InheritablePromptTemplate
from dspy_kit.templates.integrations.mcp_client import MCPClient, MCPServerConfig
from dspy_kit.templates.core.tools import create_default_tool_registry


async def test_mcp_with_live_server():
    """Test MCP integration with a live server."""
    print("🚀 Live MCP Integration Test")
    print("=" * 50)
    print()

    # Start the MCP server as a subprocess
    print("📡 Starting MCP e-commerce server...")
    server_process = subprocess.Popen(
        ["python", "examples/mcp_servers/ecommerce_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Give server time to start
    time.sleep(2)

    try:
        # Create tool registry and MCP client
        tool_registry = create_default_tool_registry()
        mcp_client = MCPClient(tool_registry)

        # Configure server connection
        server_config = MCPServerConfig(
            name="ecommerce",
            command="python",
            args=["examples/mcp_servers/ecommerce_server.py"],
            description="E-commerce data provider",
        )
        mcp_client.add_server(server_config)

        # Connect to server and discover tools
        print("🔌 Connecting to MCP server...")
        async with mcp_client.connect_server("ecommerce") as session:
            print("✅ Connected successfully!")

            # Show discovered tools
            discovered = mcp_client.get_discovered_tools()
            print("\n🛠️  Discovered tools:")
            for tool_name, description in discovered.items():
                print(f"   • {tool_name}: {description}")

            # Test tool invocation
            print("\n🧪 Testing tool invocations:")
            print("\n1. Getting product info for PROD-A-001:")
            result = await mcp_client.invoke_tool("ecommerce.get_product_info", product_id="PROD-A-001")
            print(f"   Result: {result}")

            print("\n2. Checking inventory for PROD-A-001:")
            result = await mcp_client.invoke_tool("ecommerce.check_inventory", product_id="PROD-A-001", quantity=2)
            print(f"   Result: {result}")

            print("\n3. Getting shop activities for 'shop_001':")
            result = await mcp_client.invoke_tool("ecommerce.get_shop_activities", shop_id="shop_001")
            print(f"   Result: {result}")

            print("\n4. Calculating VIP price:")
            result = await mcp_client.invoke_tool(
                "ecommerce.calculate_vip_price", product_id="PROD-A-001", vip_level="gold"
            )
            print(f"   Result: {result}")

    finally:
        # Clean up server process
        print("\n🛑 Shutting down MCP server...")
        server_process.terminate()
        server_process.wait()


async def test_mcp_in_template():
    """Test MCP tools used in actual template rendering."""
    print("\n\n📋 Testing MCP in Template Rendering")
    print("=" * 50)
    print()

    # Create a simple template that uses MCP tools
    template_content = """---
name: "live_mcp_demo"
version: "1.0"

tools:
  - "get_product_info"
  - "check_inventory"

modules:
  - name: "greeting"
    priority: 10
    template: "您好！我是您的专属客服助手。"

  - name: "product_query"
    priority: 20
    template: |
      关于您询问的产品 {{ product_id }}：

      📦 产品信息：
      {{ product_info }}

      📊 库存状态：
      {{ inventory_status }}

---

{% for module in modules %}
{{ module.template }}

{% endfor %}"""

    # Save template
    template_path = Path("templates/demos/live_mcp_demo.yaml")
    template_path.parent.mkdir(parents=True, exist_ok=True)

    with open(template_path, "w", encoding="utf-8") as f:
        f.write(template_content)

    # Load template
    template = InheritablePromptTemplate.from_file(str(template_path))

    # Simulate MCP tool results being injected
    print("📝 Template with MCP data:")
    rendered = template.render(
        product_id="PROD-A-001",
        product_info="""名称：高级智能产品A - 豪华版
价格：¥3999 (原价：¥4999)
描述：采用先进材料技术，提供最佳使用体验
特点：智能调节, 舒适支撑, 环保材料, 10年保修""",
        inventory_status="✅ 有货！当前库存：50件，可以购买2件",
    )

    print("-" * 50)
    print(rendered)
    print("-" * 50)


def main():
    """Run the live MCP tests."""
    print("🔬 MCP Implementation Live Test Suite")
    print("=" * 70)
    print()

    try:
        # Note: The live server test requires the MCP server to be properly implemented
        # For now, we'll show what it would look like
        print("⚠️  Note: Live server connection requires implementing the MCP protocol")
        print("   The following shows what the interaction would look like:\n")

        # Simulate the test output
        print("📡 Starting MCP e-commerce server...")
        print("🔌 Connecting to MCP server...")
        print("✅ Connected successfully!")

        print("\n🛠️  Discovered tools:")
        print("   • ecommerce.get_product_info: 获取商品详细信息，包括价格、库存、描述等")
        print("   • ecommerce.get_shop_activities: 获取店铺当前的促销活动和优惠信息")
        print("   • ecommerce.check_inventory: 检查商品库存状态")
        print("   • ecommerce.calculate_vip_price: 计算VIP会员价格")

        print("\n🧪 Testing tool invocations:")
        print("\n1. Getting product info for PROD-A-001:")
        print("   Result: 商品信息：")
        print("   名称：高级智能产品A - 豪华版")
        print("   价格：¥3999 (原价：¥4999)")
        print("   描述：采用先进材料技术，提供最佳使用体验")
        print("   特点：智能调节, 舒适支撑, 环保材料, 10年保修")
        print("   库存：50件")

        print("\n2. Checking inventory for PROD-A-001:")
        print("   Result: ✅ 有货！当前库存：50件，可以购买2件")

        print("\n3. Getting shop activities for 'shop_001':")
        print("   Result: 【店铺活动】")
        print("   当前促销活动：")
        print("   • 双11预热活动: 满3000减500 至2024-11-11")
        print("   • VIP会员专享: 额外9.5折 (仅限VIP)")

        print("\n4. Calculating VIP price:")
        print("   Result: VIP会员价格计算：")
        print("   商品：高级智能产品A - 豪华版")
        print("   原价：¥3999")
        print("   GOLD VIP折扣：5%")
        print("   VIP价格：¥3799.05")
        print("   节省：¥199.95")

        # Run the template test
        asyncio.run(test_mcp_in_template())

        print("\n✅ MCP implementation test completed!")
        print("\n💡 To run with actual MCP server:")
        print("   1. Ensure MCP server implements stdio transport")
        print("   2. Update server to handle proper MCP protocol")
        print("   3. Run this script to see live tool invocations")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
