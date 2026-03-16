import yfinance as yf

from tools.base import BaseTool, ToolResult
from tools.registry import register_tool


@register_tool(backends=["ollama", "claude"])
class StockMarketTool(BaseTool):
    name = "stock_market"
    description = (
        "ALWAYS use this tool instead of web_search when the user asks about stocks, "
        "share prices, tickers, market data, or financial quotes. "
        "Accepts a ticker symbol (e.g. AAPL, GOOGL, TSLA) and returns real-time price, "
        "daily change, volume, market cap, 52-week range, P/E ratio, and company details. "
        "Faster and more accurate than a web search for any stock-related query."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol, e.g. 'AAPL' for Apple, 'GOOGL' for Google",
            },
            "action": {
                "type": "string",
                "description": (
                    "What information to retrieve. Options: "
                    "'quote' for current price and key stats (default), "
                    "'history' for recent price history (last 5 trading days), "
                    "'summary' for a full company overview including sector, description, and financials"
                ),
                "enum": ["quote", "history", "summary"],
            },
        },
        "required": ["ticker"],
    }

    def execute(self, ticker: str, action: str = "quote", **kwargs) -> ToolResult:
        try:
            stock = yf.Ticker(ticker.upper().strip())
            info = stock.info

            if not info or info.get("trailingPegRatio") is None and info.get("regularMarketPrice") is None:
                # Try to check if we got any useful data at all
                if not info.get("shortName") and not info.get("symbol"):
                    return ToolResult(
                        success=False, output="",
                        error=f"Could not find data for ticker '{ticker}'. Check the symbol and try again.",
                    )

            if action == "quote":
                return self._get_quote(info)
            elif action == "history":
                return self._get_history(stock, info)
            elif action == "summary":
                return self._get_summary(info)
            else:
                return self._get_quote(info)

        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _get_quote(self, info: dict) -> ToolResult:
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose")
        change = None
        change_pct = None
        if price and prev_close:
            change = round(price - prev_close, 2)
            change_pct = round((change / prev_close) * 100, 2)

        lines = [
            f"{info.get('shortName', info.get('symbol', 'Unknown'))} ({info.get('symbol', '')})",
            f"Price: ${price}" if price else "Price: N/A",
        ]
        if change is not None:
            direction = "+" if change >= 0 else ""
            lines.append(f"Change: {direction}{change} ({direction}{change_pct}%)")
        lines.extend([
            f"Day Range: {info.get('dayLow', 'N/A')} - {info.get('dayHigh', 'N/A')}",
            f"Volume: {self._fmt_number(info.get('volume'))}",
            f"Market Cap: {self._fmt_number(info.get('marketCap'))}",
            f"52-Week Range: {info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}",
            f"P/E Ratio: {info.get('trailingPE', 'N/A')}",
        ])
        return ToolResult(success=True, output="\n".join(lines))

    def _get_history(self, stock, info: dict) -> ToolResult:
        hist = stock.history(period="5d")
        if hist.empty:
            return ToolResult(success=False, output="", error="No recent price history available.")

        lines = [f"Recent price history for {info.get('shortName', info.get('symbol', 'Unknown'))}:"]
        for date, row in hist.iterrows():
            date_str = date.strftime("%Y-%m-%d")
            lines.append(
                f"  {date_str}: Open ${row['Open']:.2f}, Close ${row['Close']:.2f}, "
                f"High ${row['High']:.2f}, Low ${row['Low']:.2f}, Vol {self._fmt_number(int(row['Volume']))}"
            )
        return ToolResult(success=True, output="\n".join(lines))

    def _get_summary(self, info: dict) -> ToolResult:
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        lines = [
            f"{info.get('shortName', 'Unknown')} ({info.get('symbol', '')})",
            f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}",
            f"Price: ${price}" if price else "Price: N/A",
            f"Market Cap: {self._fmt_number(info.get('marketCap'))}",
            f"P/E Ratio: {info.get('trailingPE', 'N/A')}",
            f"EPS: {info.get('trailingEps', 'N/A')}",
            f"Dividend Yield: {self._fmt_pct(info.get('dividendYield'))}",
            f"52-Week Range: {info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}",
            f"Avg Volume: {self._fmt_number(info.get('averageVolume'))}",
            "",
            info.get("longBusinessSummary", "No description available.")[:500],
        ]
        return ToolResult(success=True, output="\n".join(lines))

    @staticmethod
    def _fmt_number(n) -> str:
        if n is None:
            return "N/A"
        if n >= 1_000_000_000_000:
            return f"${n / 1_000_000_000_000:.2f}T"
        if n >= 1_000_000_000:
            return f"${n / 1_000_000_000:.2f}B"
        if n >= 1_000_000:
            return f"${n / 1_000_000:.2f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)

    @staticmethod
    def _fmt_pct(val) -> str:
        if val is None:
            return "N/A"
        return f"{val * 100:.2f}%"
