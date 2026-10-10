"""An MCP server providing a small, keyless web-search tool for the tutorial."""
import os
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup
from fastmcp import FastMCP

mcp = FastMCP("web-search")


@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> dict:
    """Search the public web for current information and return short source snippets."""
    if not query.strip():
        return {"results": [], "error": "A non-empty search query is required."}
    max_results = max(1, min(max_results, 10))
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {"User-Agent": "MCPChatTutorial/1.0 (+https://example.invalid)"}
    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        return {"results": [], "error": f"Search provider unavailable: {exc}"}

    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for node in soup.select(".result"):
        link = node.select_one(".result__a")
        snippet = node.select_one(".result__snippet")
        if link:
            results.append({
                "title": link.get_text(" ", strip=True),
                "url": link.get("href", ""),
                "snippet": snippet.get_text(" ", strip=True) if snippet else "",
            })
        if len(results) >= max_results:
            break
    return {"query": query, "results": results}


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=int(os.getenv("PORT", "8001")), path="/mcp")
