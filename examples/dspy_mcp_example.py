#!/usr/bin/env python3
"""
Example: Using MCP tools with DSPy and our template system.

This example shows how to:
1. Create a DSPy module from an MCP-enabled template
2. Use the module with mock MCP tools
3. Demonstrate the flow of dynamic data
"""

import sys
from pathlib import Path

# Add dspy_kit to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dspy_kit"))

import dspy
from dspy_kit.templates import InheritablePromptTemplate, create_dspy_module


# Mock MCP tool functions (in real usage, these would call actual MCP servers)
def mock_get_product_info(product_id: str) -> str:
    """Mock MCP tool for getting product info."""
    products = {
        "PROD-A-001": {
            "name": "高级智能产品A - 豪华版",
            "price": 3999,
            "description": "先进材料技术，提供最佳使用体验"
        },
        "PROD-C-001": {
            "name": "运动产品C",
            "price": 1099,
            "description": "轻量缓震路跑鞋"
        }
    }
    
    if product_id in products:
        p = products[product_id]
        return f"{p['name']} - ¥{p['price']} - {p['description']}"
    return f"未找到产品 {product_id}"


def mock_check_inventory(product_id: str, quantity: int = 1) -> str:
    """Mock MCP tool for checking inventory."""
    inventory = {
        "PROD-A-001": 50,
        "PROD-C-001": 100
    }
    
    stock = inventory.get(product_id, 0)
    if stock >= quantity:
        return f"✅ 有货！库存{stock}件"
    elif stock > 0:
        return f"⚠️ 库存不足，仅剩{stock}件"
    return "❌ 暂时缺货"


def main():
    """Demonstrate MCP tools with DSPy."""
    print("🔧 DSPy + MCP Tools Example")
    print("=" * 50)
    print()
    
    # Configure DSPy (using mock LM for demo)
    print("1️⃣ Configuring DSPy...")
    lm = dspy.LM(model="gpt-3.5-turbo", api_key="mock-key")
    dspy.configure(lm=lm)
    
    # Create an MCP-enabled template
    print("\n2️⃣ Creating MCP-enabled template...")
    template_content = """---
name: "ecommerce_assistant"
version: "1.0"
language: "zh"

input_schema:
  customer_query:
    type: "string"
    required: true
    description: "客户咨询内容"
  product_id:
    type: "string"
    required: false
    description: "相关产品ID"

output_schema:
  response:
    type: "string"
    description: "客服回复"
  needs_human:
    type: "boolean"
    description: "是否需要人工介入"

tools:
  - "get_product_info"
  - "check_inventory"

modules:
  - name: "process_query"
    priority: 10
    template: |
      作为电商客服，我需要回答客户咨询。
      
      客户咨询：{{ customer_query }}
      
      {% if product_id %}
      让我为您查询产品信息...
      [需要调用 get_product_info 和 check_inventory 工具]
      {% endif %}
      
      请提供专业、友好的回复。

---

{{ modules[0].template }}"""

    # Save template
    template_path = Path("temp_mcp_demo.yaml")
    with open(template_path, "w", encoding="utf-8") as f:
        f.write(template_content)
    
    # Load template
    template = InheritablePromptTemplate.from_file(str(template_path))
    print(f"✅ Loaded template: {template.name}")
    
    # Create DSPy module from template
    print("\n3️⃣ Creating DSPy module from template...")
    CustomerServiceModule = create_dspy_module(template)
    service_module = CustomerServiceModule()
    
    # Register mock MCP tools
    print("\n4️⃣ Registering mock MCP tools...")
    mock_tools = {
        "get_product_info": mock_get_product_info,
        "check_inventory": mock_check_inventory
    }
    
    # Example queries
    print("\n5️⃣ Testing with example queries:")
    
    test_cases = [
        {
            "customer_query": "PROD-A-001这个产品怎么样？有货吗？",
            "product_id": "PROD-A-001"
        },
        {
            "customer_query": "你们店铺的退货政策是什么？",
            "product_id": None
        },
        {
            "customer_query": "PROD-C-001还有货吗？",
            "product_id": "PROD-C-001"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}:")
        print(f"   Query: {test['customer_query']}")
        
        if test['product_id']:
            # Simulate MCP tool calls
            product_info = mock_tools["get_product_info"](test['product_id'])
            inventory = mock_tools["check_inventory"](test['product_id'])
            
            print(f"   🔧 MCP Tool Results:")
            print(f"      Product: {product_info}")
            print(f"      Inventory: {inventory}")
        
        # In real usage, the DSPy module would handle tool calls automatically
        print(f"   💬 Response: [DSPy would generate response using template + tool results]")
    
    # Clean up
    template_path.unlink()
    
    print("\n\n✅ Demo completed!")
    print("\n💡 Key Takeaways:")
    print("   • Templates can specify required MCP tools")
    print("   • DSPy modules can use these tools for dynamic data")
    print("   • Tools provide real-time information without static injection")
    print("   • Perfect for e-commerce, customer service, and data-driven apps")


if __name__ == "__main__":
    main()