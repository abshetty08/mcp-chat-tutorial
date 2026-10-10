"""MCP transport adapter for live-data domain services."""
import os

from fastmcp import FastMCP

from live_data_services import JsonHttpClient, MarketQuoteService, SportsScoreService, WeatherService


class LiveDataMCPApplication:
    def __init__(
        self,
        weather: WeatherService,
        markets: MarketQuoteService,
        sports: SportsScoreService,
    ):
        self.weather = weather
        self.markets = markets
        self.sports = sports
        self.server = FastMCP("live-data")
        self.server.tool()(self.weather_lookup)
        self.server.tool()(self.stock_quote)
        self.server.tool()(self.sports_scores)

    async def weather_lookup(self, location: str) -> dict:
        """Get the current weather for a city, region, postal code, or place name."""
        return await self.weather.current(location)

    async def stock_quote(self, symbol: str) -> dict:
        """Get the latest available price and daily change for a stock or ETF ticker."""
        return await self.markets.quote(symbol)

    async def sports_scores(
        self,
        sport: str = "basketball",
        league: str = "nba",
        date: str | None = None,
    ) -> dict:
        """Get current or scheduled scores for common professional leagues."""
        return await self.sports.scores(sport, league, date)

    def run(self) -> None:
        self.server.run(
            transport="streamable-http",
            host="0.0.0.0",
            port=int(os.getenv("PORT", "8002")),
            path="/mcp",
        )


http_client = JsonHttpClient()
application = LiveDataMCPApplication(
    WeatherService(http_client),
    MarketQuoteService(http_client),
    SportsScoreService(http_client),
)


if __name__ == "__main__":
    application.run()
