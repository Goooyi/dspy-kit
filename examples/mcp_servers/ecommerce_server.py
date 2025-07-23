#!/usr/bin/env python3
"""
Example MCP server for e-commerce data.

This server provides dynamic product information, inventory status,
and shop activities for Chinese e-commerce templates.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ToolCallResponse


# Mock e-commerce data
PRODUCTS_DB = {
    "PROD-A-001": {
        "name": "高级智能产品A - 豪华版",
        "price": 3999,
        "original_price": 4999,
        "description": "采用先进材料技术，提供最佳使用体验",
        "features": ["智能调节", "舒适支撑", "环保材料", "10年保修"],
        "inventory": 50,
        "category": "智能系列",
    },
    "PROD-B-002": {
        "name": "舒适型产品B",
        "price": 5999,
        "original_price": 6999,
        "description": "采用进口材料，透气舒适",
        "features": ["天然环保", "高弹支撑", "透气排湿", "防菌处理"],
        "inventory": 30,
        "category": "舒适系列",
    },
    "PROD-C-001": {
        "name": "运动产品C",
        "price": 1099,
        "original_price": 1299,
        "description": "轻量设计，日常使用首选",
        "features": ["轻量设计", "缓冲技术", "耐用材料", "多种尺寸"],
        "inventory": 100,
        "sizes": ["S", "M", "L", "XL", "XXL"],
    },
}

SHOP_ACTIVITIES = {
    "shop_001": {
        "current_promotions": [
            {"name": "双11预热活动", "discount": "满3000减500", "end_date": "2024-11-11"},
            {"name": "VIP会员专享", "discount": "额外9.5折", "vip_only": True},
        ],
        "flash_sales": [],
    },
    "shop_002": {
        "current_promotions": [{"name": "新品上市", "discount": "首单8.5折", "end_date": "2024-12-31"}],
        "flash_sales": [
            {"product_id": "PROD-C-001", "discount": "限时7折", "stock": 20, "end_time": "2024-12-31 23:59:59"}
        ],
    },
}

VIP_BENEFITS = {
    "gold": {
        "discount": 0.95,
        "points_multiplier": 1.5,
        "free_shipping": True,
        "priority_service": True,
        "exclusive_products": True,
    },
    "silver": {
        "discount": 0.97,
        "points_multiplier": 1.2,
        "free_shipping": True,
        "priority_service": False,
        "exclusive_products": False,
    },
}


class EcommerceServer:
    """E-commerce MCP server providing product and shop information."""

    def __init__(self):
        self.server = Server("ecommerce-server")
        self._setup_tools()

    def _setup_tools(self):
        """Register available tools."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available e-commerce tools."""
            return [
                Tool(
                    name="get_product_info",
                    description="获取商品详细信息，包括价格、库存、描述等",
                    inputSchema={
                        "type": "object",
                        "properties": {"product_id": {"type": "string", "description": "商品ID"}},
                        "required": ["product_id"],
                    },
                ),
                Tool(
                    name="get_shop_activities",
                    description="获取店铺当前的促销活动和优惠信息",
                    inputSchema={
                        "type": "object",
                        "properties": {"shop_id": {"type": "string", "description": "店铺ID (如: shop_001, shop_002)"}},
                        "required": ["shop_id"],
                    },
                ),
                Tool(
                    name="check_inventory",
                    description="检查商品库存状态",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "product_id": {"type": "string", "description": "商品ID"},
                            "quantity": {"type": "integer", "description": "需要的数量", "default": 1},
                        },
                        "required": ["product_id"],
                    },
                ),
                Tool(
                    name="calculate_vip_price",
                    description="计算VIP会员价格",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "product_id": {"type": "string", "description": "商品ID"},
                            "vip_level": {
                                "type": "string",
                                "description": "VIP等级",
                                "enum": ["gold", "silver", "bronze"],
                            },
                        },
                        "required": ["product_id", "vip_level"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> ToolCallResponse:
            """Handle tool calls."""

            if name == "get_product_info":
                product_id = arguments.get("product_id")

                if product_id in PRODUCTS_DB:
                    product = PRODUCTS_DB[product_id]
                    result = f"""商品信息：
名称：{product["name"]}
价格：¥{product["price"]} (原价：¥{product["original_price"]})
描述：{product["description"]}
特点：{", ".join(product["features"])}
库存：{product["inventory"]}件"""
                else:
                    result = f"未找到商品 {product_id}"

                return ToolCallResponse(content=[TextContent(type="text", text=result)])

            elif name == "get_shop_activities":
                shop_id = arguments.get("shop_id")

                if shop_id in SHOP_ACTIVITIES:
                    activities = SHOP_ACTIVITIES[shop_id]
                    promotions = activities["current_promotions"]

                    result = f"【{shop_id.upper()}店铺活动】\n"

                    if promotions:
                        result += "\n当前促销活动：\n"
                        for promo in promotions:
                            result += f"• {promo['name']}: {promo['discount']}"
                            if promo.get("vip_only"):
                                result += " (仅限VIP)"
                            if promo.get("end_date"):
                                result += f" 至{promo['end_date']}"
                            result += "\n"

                    if activities.get("flash_sales"):
                        result += "\n限时抢购：\n"
                        for sale in activities["flash_sales"]:
                            result += f"• 商品{sale['product_id']}: {sale['discount']} (剩余{sale['stock']}件)\n"
                else:
                    result = f"未找到店铺 {shop_id} 的活动信息"

                return ToolCallResponse(content=[TextContent(type="text", text=result)])

            elif name == "check_inventory":
                product_id = arguments.get("product_id")
                quantity = arguments.get("quantity", 1)

                if product_id in PRODUCTS_DB:
                    inventory = PRODUCTS_DB[product_id]["inventory"]

                    if inventory >= quantity:
                        result = f"✅ 有货！当前库存：{inventory}件，可以购买{quantity}件"
                    elif inventory > 0:
                        result = f"⚠️ 库存不足！当前仅剩{inventory}件，需要{quantity}件"
                    else:
                        result = f"❌ 暂时缺货"
                else:
                    result = f"未找到商品 {product_id}"

                return ToolCallResponse(content=[TextContent(type="text", text=result)])

            elif name == "calculate_vip_price":
                product_id = arguments.get("product_id")
                vip_level = arguments.get("vip_level")

                if product_id in PRODUCTS_DB and vip_level in VIP_BENEFITS:
                    product = PRODUCTS_DB[product_id]
                    vip_info = VIP_BENEFITS[vip_level]

                    original_price = product["price"]
                    vip_price = original_price * vip_info["discount"]
                    saved = original_price - vip_price

                    result = f"""VIP会员价格计算：
商品：{product["name"]}
原价：¥{original_price}
{vip_level.upper()} VIP折扣：{(1 - vip_info["discount"]) * 100:.0f}%
VIP价格：¥{vip_price:.2f}
节省：¥{saved:.2f}

其他VIP权益：
• 积分倍数：{vip_info["points_multiplier"]}x
• 免费配送：{"是" if vip_info["free_shipping"] else "否"}
• 优先客服：{"是" if vip_info["priority_service"] else "否"}"""
                else:
                    result = "无效的商品ID或VIP等级"

                return ToolCallResponse(content=[TextContent(type="text", text=result)])

            else:
                return ToolCallResponse(content=[TextContent(type="text", text=f"未知工具: {name}")])

    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream, write_stream, InitializationOptions(server_name="ecommerce-server", server_version="1.0.0")
            )


def main():
    """Run the e-commerce MCP server."""
    server = EcommerceServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
