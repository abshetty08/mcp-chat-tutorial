import pytest

from live_data_services import MarketQuoteService, SportsScoreService, WeatherService


class FakeHttpClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    async def get(self, url, params=None):
        self.requests.append((url, params))
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_weather_service_combines_geocoding_and_forecast():
    http = FakeHttpClient(
        {"results": [{"name": "Chicago", "country": "United States", "latitude": 41.8, "longitude": -87.6}]},
        {"current": {"temperature_2m": 72}, "current_units": {"temperature_2m": "°F"}},
    )
    result = await WeatherService(http).current("Chicago")
    assert result["location"]["name"] == "Chicago"
    assert result["current"]["temperature_2m"] == 72
    assert len(http.requests) == 2


@pytest.mark.asyncio
async def test_market_service_calculates_daily_change():
    http = FakeHttpClient({
        "chart": {
            "result": [{
                "meta": {"symbol": "TEST", "currency": "USD"},
                "indicators": {"quote": [{"close": [10.0, 12.5]}]},
            }],
        },
    })
    result = await MarketQuoteService(http).quote("test")
    assert result["symbol"] == "TEST"
    assert result["change"] == 2.5
    assert result["change_percent"] == 25.0


@pytest.mark.asyncio
async def test_sports_service_maps_scoreboard_events():
    http = FakeHttpClient({
        "leagues": [{"name": "NBA"}],
        "events": [{
            "name": "Away at Home",
            "date": "2026-10-10T00:00Z",
            "competitions": [{
                "status": {"type": {"description": "Final"}},
                "competitors": [{"team": {"displayName": "Home"}, "score": "101", "homeAway": "home"}],
            }],
        }],
    })
    result = await SportsScoreService(http).scores()
    assert result["league"] == "NBA"
    assert result["events"][0]["status"] == "Final"
    assert result["events"][0]["competitors"][0]["team"] == "Home"
