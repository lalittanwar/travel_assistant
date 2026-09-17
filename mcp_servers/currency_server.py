"""
MCP server exposing a currency-conversion tool, backed by the free
Frankfurter API (ECB rates, no API key required).

Uses the v1 `FastMCP` API — paired with `pip install "mcp<2"` for
compatibility with langchain-mcp-adapters.

Run standalone for testing:
    mcp dev currency_server.py
"""

import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("currency-converter", dependencies=["requests"])


@mcp.tool()
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """
    Convert an amount from one currency to another using current exchange rates.

    Args:
        amount: The amount to convert (e.g. 50000).
        from_currency: 3-letter ISO currency code to convert from (e.g. "INR").
        to_currency: 3-letter ISO currency code to convert to (e.g. "SGD").

    Returns:
        A dict with the original amount, converted amount, exchange rate used,
        and the date the rate applies to. Returns an "error" key if the
        conversion fails, instead of fabricating a rate.
    """
    from_currency = from_currency.strip().upper()
    to_currency = to_currency.strip().upper()

    try:
        resp = requests.get(
            "https://api.frankfurter.dev/v1/latest",
            params={"base": from_currency, "symbols": to_currency},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return {"error": f"Currency service unavailable: {e}"}

    rates = data.get("rates", {})
    if to_currency not in rates:
        return {
            "error": (
                f"Could not get a rate for {from_currency} -> {to_currency}. "
                f"Check the currency codes are valid ISO 4217 codes."
            )
        }

    rate = rates[to_currency]
    converted = round(amount * rate, 2)

    return {
        "amount": amount,
        "from_currency": from_currency,
        "to_currency": to_currency,
        "rate": rate,
        "converted_amount": converted,
        "rate_date": data.get("date"),
        "source": "Frankfurter API (ECB reference rates)",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
