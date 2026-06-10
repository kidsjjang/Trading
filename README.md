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

## Local LLM Agents

The dashboard can optionally add a local LLM second-opinion layer using Ollama.
This works only on the machine where Ollama is running, not on Streamlit Community Cloud.
On this PC, `qwen3:4b-instruct` is the recommended default for all three agents because it is light enough for CPU inference while still following JSON instructions reliably.

Install Ollama and pull the recommended local model:

```powershell
irm https://ollama.com/install.ps1 | iex
C:\Users\hana\AppData\Local\Programs\Ollama\ollama.exe pull qwen3:4b-instruct
```

Then run the dashboard locally and turn on `Use local LLM agents` in the sidebar.

## Hermes Agent Lite

Hermes Agent is installed separately under `C:\Users\hana\AppData\Local\hermes`.
The full Hermes CLI tool stack expects a large local context window and is too slow on this CPU-only PC, so this project uses a lighter Hermes Python Agent runner for experiments.

Create the lightweight Ollama model used by the runner:

```powershell
C:\Users\hana\AppData\Local\Programs\Ollama\ollama.exe create trading-hermes-qwen-lite -f .\ollama-hermes-lite.Modelfile
```

Run a one-off Hermes Agent market review:

```powershell
C:\Users\hana\anaconda3\python.exe .\hermes_trading_agents.py --ticker ^KS11 --period 6mo --json-out outputs\hermes_ks11.json
```

In the dashboard, turn on `Use local LLM agents`, choose `Hermes Agent Lite`, and click `Run Hermes Agent Lite review`.
Hermes Agent Lite is intentionally button-triggered because local CPU inference can take 1-3 minutes and should not run on every Streamlit rerun.

## Deploy Online

The easiest hosted option is Streamlit Community Cloud.

1. Go to `https://share.streamlit.io/` and sign in with GitHub.
2. Choose the repository `kidsjjang/Trading`.
3. Choose branch `main`.
4. Choose app file `streamlit_app.py`.
5. In advanced settings, choose Python 3.12 if available.
6. Click deploy.

After deployment, make the app public from the Streamlit app's Share settings if you want anyone with the link to view it.
