"""
MCP (Model Context Protocol) client integration for templates.

This module provides MCP client capabilities to templates, allowing them
to discover and use tools from MCP servers for dynamic knowledge population.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

import dspy
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as MCPTool

from ..core.tools import ToolRegistry, ToolConfig

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""

    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Optional[Dict[str, str]] = None
    description: str = ""

    def to_stdio_params(self) -> StdioServerParameters:
        """Convert to MCP StdioServerParameters."""
        return StdioServerParameters(command=self.command, args=self.args, env=self.env)


class MCPClient:
    """
    MCP client for template system integration.

    Handles connection to MCP servers, tool discovery, and registration
    with the template tool registry.
    """

    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        """
        Initialize MCP client.

        Args:
            tool_registry: Template tool registry to register discovered tools
        """
        self.tool_registry = tool_registry
        self._servers: Dict[str, MCPServerConfig] = {}
        self._sessions: Dict[str, ClientSession] = {}
        self._discovered_tools: Dict[str, MCPTool] = {}

    def add_server(self, config: MCPServerConfig):
        """Add an MCP server configuration."""
        self._servers[config.name] = config
        logger.info(f"Added MCP server: {config.name}")

    @asynccontextmanager
    async def connect_server(self, server_name: str):
        """
        Connect to an MCP server.

        Args:
            server_name: Name of the server to connect to

        Yields:
            ClientSession: Connected MCP client session
        """
        if server_name not in self._servers:
            raise ValueError(f"Server {server_name} not configured")

        config = self._servers[server_name]

        async with stdio_client(config.to_stdio_params()) as (read, write):
            async with ClientSession(read, write) as session:
                logger.info(f"Connected to MCP server: {server_name}")

                # Store session for tool invocation
                self._sessions[server_name] = session

                try:
                    # Initialize the session
                    await session.initialize()

                    # Discover and register tools
                    await self._discover_tools(server_name, session)

                    yield session
                finally:
                    # Clean up
                    if server_name in self._sessions:
                        del self._sessions[server_name]

    async def _discover_tools(self, server_name: str, session: ClientSession):
        """
        Discover tools from an MCP server and register them.

        Args:
            server_name: Name of the server
            session: Connected MCP session
        """
        # List available tools
        tools_result = await session.list_tools()

        for tool in tools_result.tools:
            # Create unique tool name with server prefix
            full_tool_name = f"{server_name}.{tool.name}"

            # Store the MCP tool
            self._discovered_tools[full_tool_name] = tool

            logger.info(f"Discovered tool: {full_tool_name} - {tool.description}")

            # Register with tool registry if available
            if self.tool_registry:
                await self._register_tool(server_name, tool)

    async def _register_tool(self, server_name: str, mcp_tool: MCPTool):
        """
        Register an MCP tool with the template tool registry.

        Args:
            server_name: Name of the server providing the tool
            mcp_tool: MCP tool to register
        """
        full_tool_name = f"{server_name}.{mcp_tool.name}"

        # Create tool config
        tool_config = ToolConfig(
            name=full_tool_name,
            description=mcp_tool.description or f"MCP tool from {server_name}",
            parameters=mcp_tool.inputSchema.properties if mcp_tool.inputSchema else {},
            provider="mcp",
            implementation=f"mcp:{server_name}:{mcp_tool.name}",
        )

        # Register with the tool registry
        self.tool_registry._tools[full_tool_name] = tool_config

        # Create DSPy tool wrapper
        async def mcp_tool_wrapper(**kwargs):
            """Wrapper to invoke MCP tool through DSPy."""
            if server_name not in self._sessions:
                raise RuntimeError(f"Server {server_name} not connected")

            session = self._sessions[server_name]
            result = await session.call_tool(mcp_tool.name, kwargs)

            # Handle different result types
            if hasattr(result, "content"):
                return result.content
            elif hasattr(result, "text"):
                return result.text
            else:
                return str(result)

        # Create DSPy tool
        dspy_tool = dspy.Tool(
            func=mcp_tool_wrapper, name=full_tool_name, desc=mcp_tool.description or f"MCP tool from {server_name}"
        )

        self.tool_registry._dspy_tools[full_tool_name] = dspy_tool

    async def invoke_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Invoke an MCP tool.

        Args:
            tool_name: Full tool name (server.tool_name)
            **kwargs: Tool arguments

        Returns:
            Tool result
        """
        if tool_name not in self._discovered_tools:
            raise ValueError(f"Tool {tool_name} not found")

        # Parse server and tool names
        parts = tool_name.split(".", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid tool name format: {tool_name}")

        server_name, actual_tool_name = parts

        if server_name not in self._sessions:
            raise RuntimeError(f"Server {server_name} not connected")

        session = self._sessions[server_name]
        result = await session.call_tool(actual_tool_name, kwargs)

        return result

    def get_discovered_tools(self) -> Dict[str, str]:
        """
        Get all discovered tools with their descriptions.

        Returns:
            Dict mapping tool names to descriptions
        """
        return {name: tool.description or "No description" for name, tool in self._discovered_tools.items()}


class MCPTemplateIntegration:
    """
    Integration layer between MCP and the template system.

    Provides high-level methods for templates to use MCP capabilities.
    """

    def __init__(self, tool_registry: ToolRegistry):
        """
        Initialize MCP template integration.

        Args:
            tool_registry: Template tool registry
        """
        self.tool_registry = tool_registry
        self.mcp_client = MCPClient(tool_registry)
        self._event_loop = None

    def add_mcp_server(
        self, name: str, command: str, args: List[str] = None, env: Dict[str, str] = None, description: str = ""
    ):
        """
        Add an MCP server configuration.

        Args:
            name: Server name
            command: Command to start the server
            args: Command arguments
            env: Environment variables
            description: Server description
        """
        config = MCPServerConfig(name=name, command=command, args=args or [], env=env, description=description)
        self.mcp_client.add_server(config)

    def connect_and_discover(self, server_names: Optional[List[str]] = None):
        """
        Connect to MCP servers and discover tools.

        Args:
            server_names: List of server names to connect to (None for all)
        """
        # Create or get event loop
        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)

        # Connect to servers
        servers_to_connect = server_names or list(self.mcp_client._servers.keys())

        for server_name in servers_to_connect:
            self._event_loop.run_until_complete(self._connect_single_server(server_name))

    async def _connect_single_server(self, server_name: str):
        """Connect to a single MCP server."""
        async with self.mcp_client.connect_server(server_name) as session:
            # Tools are discovered automatically in connect_server
            logger.info(f"Successfully connected to {server_name}")

            # Keep session alive for tool invocation
            # In a real implementation, you'd want to manage this better
            await asyncio.sleep(0.1)

    def get_available_mcp_tools(self) -> Dict[str, str]:
        """
        Get all available MCP tools.

        Returns:
            Dict mapping tool names to descriptions
        """
        return self.mcp_client.get_discovered_tools()


def create_mcp_integration(tool_registry: ToolRegistry) -> MCPTemplateIntegration:
    """
    Create an MCP integration instance.

    Args:
        tool_registry: Template tool registry

    Returns:
        MCPTemplateIntegration instance
    """
    return MCPTemplateIntegration(tool_registry)
