"""Domain services for weather, market, and sports data."""
from datetime import datetime, timezone
from typing import Any

import httpx


class JsonHttpClient:
    """Small reusable HTTP adapter for JSON APIs."""

    def __init__(self, timeout: float = 12):
        self._timeout = timeout
        self._headers = {"User-Agent": "MCPChatTutorial/1.0"}

    async def get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout, headers=self._headers) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()


class WeatherService:
    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, http_client: JsonHttpClient):
        self._http = http_client

    async def current(self, location: str) -> dict:
        try:
            geocoding = await self._http.get(self.GEOCODING_URL, {
                "name": location,
                "count": 1,
                "language": "en",
                "format": "json",
            })
            places = geocoding.get("results", [])
            if not places:
                return {"error": f"No place found for {location!r}."}
            place = places[0]
            forecast = await self._http.get(self.FORECAST_URL, {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": (
                    "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "weather_code,wind_speed_10m"
                ),
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
            })
        except httpx.HTTPError as exc:
            return {"error": f"Weather provider unavailable: {exc}"}
        return {
            "location": {
                key: place.get(key)
                for key in ("name", "admin1", "country", "latitude", "longitude")
            },
            "current": forecast.get("current", {}),
            "units": forecast.get("current_units", {}),
        }


class MarketQuoteService:
    ENDPOINT = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

    def __init__(self, http_client: JsonHttpClient):
        self._http = http_client

    async def quote(self, symbol: str) -> dict:
        symbol = symbol.upper().strip()
        if not symbol.replace("-", "").replace(".", "").isalnum():
            return {"error": "Ticker may only contain letters, numbers, dots, and hyphens."}
        try:
            payload = await self._http.get(
                self.ENDPOINT.format(symbol=symbol),
                {"range": "5d", "interval": "1d"},
            )
        except httpx.HTTPError as exc:
            return {"error": f"Market data provider unavailable: {exc}"}
        result = payload.get("chart", {}).get("result")
        if not result:
            return {"error": f"No market data found for {symbol}."}
        data = result[0]
        closes = [
            value
            for value in data.get("indicators", {}).get("quote", [{}])[0].get("close", [])
            if value is not None
        ]
        if not closes:
            return {"error": f"No closing price found for {symbol}."}
        last = closes[-1]
        previous = closes[-2] if len(closes) > 1 else last
        meta = data.get("meta", {})
        return {
            "symbol": meta.get("symbol", symbol),
            "price": round(last, 4),
            "currency": meta.get("currency"),
            "previous_close": round(previous, 4),
            "change": round(last - previous, 4),
            "change_percent": round((last - previous) / previous * 100, 3) if previous else None,
            "exchange": meta.get("fullExchangeName"),
            "note": "Latest available chart close; it may not be a real-time quote.",
        }


class SportsScoreService:
    ENDPOINT = "https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard"

    def __init__(self, http_client: JsonHttpClient):
        self._http = http_client

    async def scores(
        self,
        sport: str = "basketball",
        league: str = "nba",
        date: str | None = None,
    ) -> dict:
        sport, league = sport.lower().strip(), league.lower().strip()
        params = {"dates": date.replace("-", "")} if date else None
        try:
            payload = await self._http.get(
                self.ENDPOINT.format(sport=sport, league=league),
                params,
            )
        except httpx.HTTPError as exc:
            return {"error": f"Scores provider unavailable: {exc}"}
        events = []
        for event in payload.get("events", []):
            competition = event.get("competitions", [{}])[0]
            events.append({
                "name": event.get("name"),
                "date": event.get("date"),
                "status": competition.get("status", {}).get("type", {}).get("description"),
                "competitors": [
                    {
                        "team": competitor.get("team", {}).get("displayName"),
                        "score": competitor.get("score"),
                        "home_away": competitor.get("homeAway"),
                    }
                    for competitor in competition.get("competitors", [])
                ],
            })
        return {
            "league": payload.get("leagues", [{}])[0].get("name", league.upper()),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "events": events,
        }
