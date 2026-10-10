"""MCP service for weather, live scores, and market snapshots."""
import os
from datetime import datetime, timezone

import httpx
from fastmcp import FastMCP

mcp = FastMCP("live-data")
HEADERS = {"User-Agent": "MCPChatTutorial/1.0"}


@mcp.tool()
async def weather_lookup(location: str) -> dict:
    """Get the current weather for a city, region, postal code, or place name."""
    async with httpx.AsyncClient(timeout=12, headers=HEADERS) as client:
        try:
            geo = await client.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": location, "count": 1, "language": "en", "format": "json"})
            geo.raise_for_status()
            places = geo.json().get("results", [])
            if not places:
                return {"error": f"No place found for {location!r}."}
            place = places[0]
            weather = await client.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": place["latitude"], "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
                "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            })
            weather.raise_for_status()
        except httpx.HTTPError as exc:
            return {"error": f"Weather provider unavailable: {exc}"}
    current = weather.json().get("current", {})
    return {"location": {key: place.get(key) for key in ("name", "admin1", "country", "latitude", "longitude")}, "current": current, "units": weather.json().get("current_units", {})}


@mcp.tool()
async def stock_quote(symbol: str) -> dict:
    """Get the latest available price and daily change for a stock or ETF ticker."""
    symbol = symbol.upper().strip()
    if not symbol.replace("-", "").replace(".", "").isalnum():
        return {"error": "Ticker may only contain letters, numbers, dots, and hyphens."}
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    async with httpx.AsyncClient(timeout=12, headers=HEADERS) as client:
        try:
            response = await client.get(url, params={"range": "5d", "interval": "1d"})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return {"error": f"Market data provider unavailable: {exc}"}
    result = response.json().get("chart", {}).get("result")
    if not result:
        return {"error": f"No market data found for {symbol}."}
    data = result[0]
    closes = [value for value in data.get("indicators", {}).get("quote", [{}])[0].get("close", []) if value is not None]
    if not closes:
        return {"error": f"No closing price found for {symbol}."}
    last, previous = closes[-1], closes[-2] if len(closes) > 1 else closes[-1]
    meta = data.get("meta", {})
    return {"symbol": meta.get("symbol", symbol), "price": round(last, 4), "currency": meta.get("currency"), "previous_close": round(previous, 4), "change": round(last - previous, 4), "change_percent": round((last - previous) / previous * 100, 3) if previous else None, "exchange": meta.get("fullExchangeName"), "note": "Latest available chart close; it may not be a real-time quote."}


@mcp.tool()
async def sports_scores(sport: str = "basketball", league: str = "nba", date: str | None = None) -> dict:
    """Get current or scheduled scores. Examples: nba basketball, nfl football, mlb baseball, nhl hockey."""
    sport, league = sport.lower().strip(), league.lower().strip()
    endpoint = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard"
    params = {"dates": date.replace("-", "")} if date else {}
    async with httpx.AsyncClient(timeout=12, headers=HEADERS) as client:
        try:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return {"error": f"Scores provider unavailable: {exc}"}
    payload = response.json()
    events = []
    for event in payload.get("events", []):
        competition = event.get("competitions", [{}])[0]
        competitors = competition.get("competitors", [])
        events.append({
            "name": event.get("name"), "date": event.get("date"), "status": competition.get("status", {}).get("type", {}).get("description"),
            "competitors": [{"team": c.get("team", {}).get("displayName"), "score": c.get("score"), "home_away": c.get("homeAway")} for c in competitors],
        })
    return {"league": payload.get("leagues", [{}])[0].get("name", league.upper()), "retrieved_at": datetime.now(timezone.utc).isoformat(), "events": events}


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=int(os.getenv("PORT", "8002")), path="/mcp")
