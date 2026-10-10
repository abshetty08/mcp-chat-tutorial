import pytest

from search_service import DuckDuckGoSearchService


@pytest.mark.asyncio
async def test_empty_query_is_rejected_without_network_access():
    result = await DuckDuckGoSearchService().search("   ")
    assert result == {"results": [], "error": "A non-empty search query is required."}


def test_result_parser_limits_and_normalizes_entries():
    html = """
    <div class="result"><a class="result__a" href="https://one.example"> One </a>
      <span class="result__snippet"> First result </span></div>
    <div class="result"><a class="result__a" href="https://two.example"> Two </a></div>
    """
    results = DuckDuckGoSearchService._parse_results(html, 1)
    assert results == [{
        "title": "One",
        "url": "https://one.example",
        "snippet": "First result",
    }]
