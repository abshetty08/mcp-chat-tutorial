"""MCP transport adapter for the web-search service."""
import os

from fastmcp import FastMCP

from search_service import DuckDuckGoSearchService


class SearchMCPApplication:
    def __init__(self, search_service: DuckDuckGoSearchService):
        self.search_service = search_service
        self.server = FastMCP("web-search")
        self.server.tool()(self.web_search)

    async def web_search(self, query: str, max_results: int = 5) -> dict:
        """Search the public web for current information and return short source snippets."""
        return await self.search_service.search(query, max_results)

    def run(self) -> None:
        self.server.run(
            transport="streamable-http",
            host="0.0.0.0",
            port=int(os.getenv("PORT", "8001")),
            path="/mcp",
        )


application = SearchMCPApplication(DuckDuckGoSearchService())


if __name__ == "__main__":
    application.run()
