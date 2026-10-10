"""Object-oriented adapter for discovering and invoking remote MCP tools."""
from contextlib import AsyncExitStack
from typing import Any

from fastmcp import Client as MCPClient

from .serialization import JsonValueSerializer


class MCPToolGateway:
    """Manage remote MCP client lifetimes and expose Chat Completions tools."""

    def __init__(self, urls: tuple[str, ...], serializer: JsonValueSerializer | None = None):
        self._urls = urls
        self._serializer = serializer or JsonValueSerializer()
        self._stack: AsyncExitStack | None = None
        self._clients_by_tool: dict[str, MCPClient] = {}
        self.tools: list[dict[str, Any]] = []

    async def __aenter__(self) -> "MCPToolGateway":
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        try:
            for url in self._urls:
                client = MCPClient(url)
                await self._stack.enter_async_context(client)
                await self._register_client_tools(client)
        except Exception:
            await self._stack.aclose()
            raise
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self._stack:
            await self._stack.__aexit__(exc_type, exc, traceback)

    async def _register_client_tools(self, client: MCPClient) -> None:
        for tool in await client.list_tools():
            tool_data = self._serializer.encode(tool)
            name = tool_data["name"]
            if name in self._clients_by_tool:
                raise RuntimeError(f"Duplicate MCP tool name: {name}")
            self._clients_by_tool[name] = client
            self.tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool_data.get("description", ""),
                    "parameters": tool_data.get(
                        "inputSchema",
                        tool_data.get("input_schema", {"type": "object", "properties": {}}),
                    ),
                },
            })

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        client = self._clients_by_tool.get(name)
        if client is None:
            raise KeyError(f"Unknown MCP tool: {name}")
        return self._serializer.encode(await client.call_tool(name, arguments))
