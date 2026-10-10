"""Search-domain service independent from the MCP transport."""
import httpx
from bs4 import BeautifulSoup


class DuckDuckGoSearchService:
    """Query and parse DuckDuckGo's public HTML search endpoint."""

    ENDPOINT = "https://html.duckduckgo.com/html/"

    def __init__(self, timeout: float = 12):
        self._timeout = timeout
        self._headers = {"User-Agent": "MCPChatTutorial/1.0 (+https://example.invalid)"}

    async def search(self, query: str, max_results: int = 5) -> dict:
        query = query.strip()
        if not query:
            return {"results": [], "error": "A non-empty search query is required."}
        result_limit = max(1, min(max_results, 10))
        try:
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                response = await client.get(
                    self.ENDPOINT,
                    params={"q": query},
                    headers=self._headers,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            return {"results": [], "error": f"Search provider unavailable: {exc}"}
        return {"query": query, "results": self._parse_results(response.text, result_limit)}

    @staticmethod
    def _parse_results(html: str, result_limit: int) -> list[dict[str, str]]:
        soup = BeautifulSoup(html, "html.parser")
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
            if len(results) >= result_limit:
                break
        return results
