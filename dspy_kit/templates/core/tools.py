"""
Tool integration for dynamic knowledge population in templates.
"""

import dspy
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass


@dataclass
class ToolConfig:
    """Configuration for a tool in the template."""

    name: str
    description: str
    parameters: Dict[str, Any]
    provider: str = "function"  # "function", "mcp", "builtin"
    implementation: Optional[Union[Callable, str]] = None


class ToolRegistry:
    """Registry for managing tools available to templates."""

    def __init__(self):
        self._tools: Dict[str, ToolConfig] = {}
        self._dspy_tools: Dict[str, dspy.Tool] = {}

    def register_function(self, name: str, description: str, parameters: Dict[str, Any], func: Callable):
        """Register a Python function as a tool."""
        config = ToolConfig(
            name=name, description=description, parameters=parameters, provider="function", implementation=func
        )
        self._tools[name] = config

        # Create DSPy tool
        self._dspy_tools[name] = dspy.Tool(func=func, name=name, desc=description)

    def register_mcp_tool(self, name: str, mcp_tool):
        """Register an MCP tool."""
        # For demo/mock tools, register as function
        # In real implementation, this would use dspy.Tool.from_mcp_tool()
        if hasattr(mcp_tool, "__call__"):
            # Mock MCP tool - register as function
            config = ToolConfig(
                name=name,
                description=mcp_tool.description,
                parameters=mcp_tool.inputSchema,
                provider="mcp",
                implementation=mcp_tool,
            )
            self._tools[name] = config

            # Create DSPy tool from function
            self._dspy_tools[name] = dspy.Tool(func=mcp_tool, name=name, desc=mcp_tool.description)
        else:
            # Real MCP tool - would use: dspy.Tool.from_mcp_tool(mcp_tool)
            config = ToolConfig(
                name=name,
                description=getattr(mcp_tool, "description", ""),
                parameters=getattr(mcp_tool, "inputSchema", {}),
                provider="mcp",
                implementation=mcp_tool,
            )
            self._tools[name] = config
            # Note: Real implementation would be:
            # self._dspy_tools[name] = dspy.Tool.from_mcp_tool(mcp_tool)

    def has_mcp_integration(self) -> bool:
        """Check if MCP integration is available."""
        try:
            from ..integrations.mcp_client import MCPTemplateIntegration

            return True
        except ImportError:
            return False

    def get_tool(self, name: str) -> Optional[dspy.Tool]:
        """Get a DSPy tool by name."""
        return self._dspy_tools.get(name)

    def get_config(self, name: str) -> Optional[ToolConfig]:
        """Get tool configuration by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all available tool names."""
        return list(self._tools.keys())

    def get_tools_for_template(self, tool_names: List[str]) -> List[dspy.Tool]:
        """Get DSPy tools for a template."""
        tools = []
        for name in tool_names:
            tool = self.get_tool(name)
            if tool:
                tools.append(tool)
        return tools


class ToolAwareTemplateEngine:
    """Template engine that can work with tools for dynamic knowledge."""

    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry

    def render_with_tools(self, template, variables: Dict[str, Any], tools: List[str]) -> str:
        """Render template with access to tools for dynamic knowledge."""
        # Get tool instances
        available_tools = self.tool_registry.get_tools_for_template(tools)

        # Create a tool-aware context
        tool_context = self._create_tool_context(available_tools, variables)

        # Merge with variables (variables already includes modules from template)
        render_variables = {**variables, **tool_context}

        # Render template directly with Jinja2 to avoid circular calls
        jinja_template = template._jinja_env.from_string(template.content_template)
        return jinja_template.render(**render_variables)

    def _create_tool_context(self, tools: List[dspy.Tool], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Create context variables for tool usage."""
        context = {}

        # Add tool availability flags
        for tool in tools:
            context[f"has_{tool.name}"] = True

        # Add tool descriptions for prompt inclusion
        if tools:
            tool_descriptions = []
            for tool in tools:
                desc = f"- {tool.name}: {tool.desc}"
                tool_descriptions.append(desc)
            context["available_tools"] = "\\n".join(tool_descriptions)
        else:
            context["available_tools"] = "No tools available"

        return context


# Built-in tools for e-commerce scenarios
def get_product_info(product_id: str, shop_id: str) -> str:
    """
    Fetch product information including price, stock, and details.

    Args:
        product_id: ID of the product to fetch
        shop_id: ID of the shop

    Returns:
        Product information as formatted string
    """
    # This would typically call an API or database
    # For demo purposes, return mock data
    return f"""
    产品ID: {product_id}
    店铺: {shop_id}
    名称: 示例智能产品A
    价格: ¥8999
    库存: 50件
    活动: 买就送配件
    描述: 具有智能监测功能，可自动调节参数
    """


def get_shop_activities(shop_id: str, category: str = "all") -> str:
    """
    Fetch current shop activities and promotions.

    Args:
        shop_id: ID of the shop
        category: Category filter (optional)

    Returns:
        Current activities as formatted string
    """
    return f"""
    店铺活动 - {shop_id}:
    1. 新会员优惠 - 入会享满1000元使用100元优惠券
    2. 买就送活动 - 购买产品A送配件
    3. 季节优惠 - 符合条件商品享受15%折扣
    活动时间: 2024年1月1日 - 2024年12月31日
    """


def check_inventory(product_id: str) -> str:
    """
    Check product inventory status.

    Args:
        product_id: ID of the product

    Returns:
        Inventory status
    """
    return f"产品 {product_id} 库存充足，现货50件"


# Create default tool registry with built-in tools
def create_default_tool_registry() -> ToolRegistry:
    """Create a tool registry with built-in e-commerce tools."""
    registry = ToolRegistry()

    # Register built-in tools
    registry.register_function(
        name="get_product_info",
        description="获取商品详细信息包括价格、库存、活动等",
        parameters={
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "商品ID"},
                "shop_id": {"type": "string", "description": "店铺ID"},
            },
            "required": ["product_id", "shop_id"],
        },
        func=get_product_info,
    )

    registry.register_function(
        name="get_shop_activities",
        description="获取店铺当前活动和优惠信息",
        parameters={
            "type": "object",
            "properties": {
                "shop_id": {"type": "string", "description": "店铺ID"},
                "category": {"type": "string", "description": "商品类别筛选"},
            },
            "required": ["shop_id"],
        },
        func=get_shop_activities,
    )

    registry.register_function(
        name="check_inventory",
        description="检查商品库存状态",
        parameters={
            "type": "object",
            "properties": {"product_id": {"type": "string", "description": "商品ID"}},
            "required": ["product_id"],
        },
        func=check_inventory,
    )

    return registry
