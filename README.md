# Simple Multi-Agent Trading Prototype

This is a small local prototype inspired by TradingAgents. The dashboard defaults to
the KOSPI Composite Index using the Yahoo Finance ticker `^KS11`.

It uses three rule-based agents:

- `ResearcherAgent`: downloads Yahoo Finance price data and builds a technical research report.
- `TraderAgent`: converts the research report into a simple BUY / SELL / HOLD plan.
- `RiskManagerAgent`: reviews volatility, drawdown, confidence, and caps the position size.

This project is educational only. It is not financial advice and it is not an automated trading system.

## Setup

```powershell
C:\Users\hana\anaconda3\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
C:\Users\hana\anaconda3\python.exe simple_trading_agents.py --ticker ^KS11
```

Save a JSON result:

```powershell
C:\Users\hana\anaconda3\python.exe simple_trading_agents.py --ticker AAPL --json-out outputs\aapl.json
```

## Dashboard

Run the local dashboard:

```powershell
C:\Users\hana\anaconda3\python.exe -m streamlit run dashboard.py
```

The dashboard polls Yahoo Finance on a timer and re-runs the three agents on each refresh.
Yahoo Finance is convenient for prototyping, but it should be treated as delayed/best-effort market data rather than guaranteed exchange-grade real-time data.
