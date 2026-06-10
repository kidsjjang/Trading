from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


@dataclass
class ResearchReport:
    ticker: str
    as_of: str
    last_price: float
    sma_20: float
    sma_50: float
    rsi_14: float
    return_20d_pct: float
    volatility_20d_annual_pct: float
    max_drawdown_90d_pct: float
    score: int
    thesis: list[str]
    risks: list[str]


@dataclass
class TradePlan:
    action: str
    confidence: float
    proposed_position_pct: float
    rationale: list[str]


@dataclass
class RiskReview:
    final_action: str
    approved_position_pct: float
    risk_level: str
    controls: list[str]
    notes: list[str]


@dataclass
class AgentRunResult:
    researcher: ResearchReport
    trader: TradePlan
    risk_manager: RiskReview
    disclaimer: str


def _round(value: Any, digits: int = 2) -> float:
    if pd.isna(value):
        return float("nan")
    return round(float(value), digits)


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def load_yahoo_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if data.empty:
        raise RuntimeError(f"No Yahoo Finance data returned for ticker '{ticker}'.")

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(data.columns)
    if missing:
        raise RuntimeError(f"Yahoo Finance data is missing columns: {sorted(missing)}")

    return data.dropna(subset=["Close"]).copy()


class ResearcherAgent:
    name = "Researcher"

    def analyze(self, ticker: str, history: pd.DataFrame) -> ResearchReport:
        close = history["Close"]
        returns = close.pct_change()

        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        rsi_14 = calculate_rsi(close, 14)
        drawdown = close / close.cummax() - 1

        last_price = close.iloc[-1]
        last_sma_20 = sma_20.iloc[-1]
        last_sma_50 = sma_50.iloc[-1]
        last_rsi = rsi_14.iloc[-1]
        return_20d = close.pct_change(20).iloc[-1]
        vol_20d = returns.tail(20).std() * (252**0.5)
        max_drawdown_90d = drawdown.tail(90).min()

        score = 0
        thesis: list[str] = []
        risks: list[str] = []

        if last_price > last_sma_20:
            score += 1
            thesis.append("Price is above the 20-day moving average.")
        else:
            score -= 1
            risks.append("Price is below the 20-day moving average.")

        if last_price > last_sma_50:
            score += 1
            thesis.append("Price is above the 50-day moving average.")
        else:
            score -= 1
            risks.append("Price is below the 50-day moving average.")

        if last_sma_20 > last_sma_50:
            score += 1
            thesis.append("Short-term trend is stronger than the medium-term trend.")
        else:
            score -= 1
            risks.append("Short-term trend is weaker than the medium-term trend.")

        if return_20d > 0:
            score += 1
            thesis.append("20-day momentum is positive.")
        else:
            score -= 1
            risks.append("20-day momentum is negative.")

        if last_rsi >= 70:
            score -= 1
            risks.append("RSI is in an overbought zone.")
        elif last_rsi <= 30:
            score += 1
            thesis.append("RSI is in an oversold rebound zone.")
        elif 45 <= last_rsi <= 65:
            score += 1
            thesis.append("RSI is in a balanced trend-confirming range.")

        if vol_20d > 0.35:
            risks.append("Recent annualized volatility is elevated.")
        if max_drawdown_90d < -0.15:
            risks.append("The 90-day drawdown is significant.")

        as_of = history.index[-1].date().isoformat()
        return ResearchReport(
            ticker=ticker.upper(),
            as_of=as_of,
            last_price=_round(last_price),
            sma_20=_round(last_sma_20),
            sma_50=_round(last_sma_50),
            rsi_14=_round(last_rsi),
            return_20d_pct=_round(return_20d * 100),
            volatility_20d_annual_pct=_round(vol_20d * 100),
            max_drawdown_90d_pct=_round(max_drawdown_90d * 100),
            score=score,
            thesis=thesis,
            risks=risks,
        )


class TraderAgent:
    name = "Trader"

    def decide(self, report: ResearchReport) -> TradePlan:
        if report.score >= 3:
            action = "BUY"
        elif report.score <= -3:
            action = "SELL"
        else:
            action = "HOLD"

        confidence = min(0.9, 0.45 + abs(report.score) * 0.1)
        proposed_position = 0.0 if action == "HOLD" else min(15.0, 5.0 + abs(report.score) * 2.0)

        rationale = [
            f"Research score is {report.score}.",
            f"20-day return is {report.return_20d_pct}%.",
            f"RSI(14) is {report.rsi_14}.",
        ]

        return TradePlan(
            action=action,
            confidence=_round(confidence, 2),
            proposed_position_pct=_round(proposed_position),
            rationale=rationale,
        )


class RiskManagerAgent:
    name = "RiskManager"

    def review(self, report: ResearchReport, plan: TradePlan, max_position_pct: float) -> RiskReview:
        notes: list[str] = []
        controls = [
            "Do not treat this as investment advice.",
            "Re-check the signal with fresh data before any real decision.",
            "Use a predefined stop/loss or invalidation rule in any live experiment.",
        ]

        risk_level = "LOW"
        volatility_cap = max_position_pct

        if report.volatility_20d_annual_pct >= 45:
            risk_level = "HIGH"
            volatility_cap = min(volatility_cap, 3.0)
            notes.append("Position capped because recent volatility is very high.")
        elif report.volatility_20d_annual_pct >= 30:
            risk_level = "MEDIUM"
            volatility_cap = min(volatility_cap, 5.0)
            notes.append("Position capped because recent volatility is elevated.")

        if report.max_drawdown_90d_pct <= -20:
            risk_level = "HIGH"
            volatility_cap = min(volatility_cap, 3.0)
            notes.append("Position capped because recent drawdown is severe.")

        if plan.proposed_position_pct > 0 and plan.confidence < 0.6:
            volatility_cap = min(volatility_cap, 5.0)
            notes.append("Position capped because trader confidence is modest.")

        approved_position = min(plan.proposed_position_pct, volatility_cap)
        final_action = plan.action

        if plan.action == "BUY" and approved_position < 1:
            final_action = "HOLD"
            notes.append("Buy was downgraded to hold because approved size is too small.")

        if plan.action == "SELL":
            controls.append("For existing holdings, consider staged reduction rather than a single market order.")

        if not notes:
            notes.append("No additional risk override was applied.")

        return RiskReview(
            final_action=final_action,
            approved_position_pct=_round(approved_position),
            risk_level=risk_level,
            controls=controls,
            notes=notes,
        )


def run_agents(ticker: str, period: str, interval: str, max_position_pct: float) -> AgentRunResult:
    history = load_yahoo_history(ticker, period, interval)
    return run_agents_from_history(ticker, history, max_position_pct)


def run_agents_from_history(
    ticker: str, history: pd.DataFrame, max_position_pct: float
) -> AgentRunResult:
    researcher = ResearcherAgent()
    trader = TraderAgent()
    risk_manager = RiskManagerAgent()

    research_report = researcher.analyze(ticker, history)
    trade_plan = trader.decide(research_report)
    risk_review = risk_manager.review(research_report, trade_plan, max_position_pct)

    return AgentRunResult(
        researcher=research_report,
        trader=trade_plan,
        risk_manager=risk_review,
        disclaimer=(
            "Educational multi-agent prototype only. It is not financial advice, "
            "not a recommendation, and not an automated trading system."
        ),
    )


def print_result(result: AgentRunResult) -> None:
    print("\n=== Simple Multi-Agent Trading Prototype ===")
    print(f"Ticker: {result.researcher.ticker} | As of: {result.researcher.as_of}")
    print("\n[Researcher]")
    print(f"Last price: {result.researcher.last_price}")
    print(f"SMA20 / SMA50: {result.researcher.sma_20} / {result.researcher.sma_50}")
    print(f"RSI14: {result.researcher.rsi_14}")
    print(f"20D return: {result.researcher.return_20d_pct}%")
    print(f"20D annualized volatility: {result.researcher.volatility_20d_annual_pct}%")
    print(f"90D max drawdown: {result.researcher.max_drawdown_90d_pct}%")
    print(f"Score: {result.researcher.score}")
    print("Thesis:")
    for item in result.researcher.thesis:
        print(f"  - {item}")
    print("Risks:")
    for item in result.researcher.risks:
        print(f"  - {item}")

    print("\n[Trader]")
    print(f"Action: {result.trader.action}")
    print(f"Confidence: {result.trader.confidence}")
    print(f"Proposed position: {result.trader.proposed_position_pct}%")
    for item in result.trader.rationale:
        print(f"  - {item}")

    print("\n[Risk Manager]")
    print(f"Final action: {result.risk_manager.final_action}")
    print(f"Approved position: {result.risk_manager.approved_position_pct}%")
    print(f"Risk level: {result.risk_manager.risk_level}")
    print("Controls:")
    for item in result.risk_manager.controls:
        print(f"  - {item}")
    print("Notes:")
    for item in result.risk_manager.notes:
        print(f"  - {item}")

    print(f"\nDisclaimer: {result.disclaimer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny 3-agent trading analysis prototype.")
    parser.add_argument("--ticker", default="SPY", help="Yahoo Finance ticker, e.g. SPY, AAPL, NVDA.")
    parser.add_argument("--period", default="6mo", help="Yahoo Finance history period, e.g. 3mo, 6mo, 1y.")
    parser.add_argument("--interval", default="1d", help="Yahoo Finance interval, e.g. 1d, 1wk.")
    parser.add_argument("--max-position-pct", type=float, default=10.0, help="Risk manager max position cap.")
    parser.add_argument("--json-out", type=Path, help="Optional path to save the full run result as JSON.")
    args = parser.parse_args()

    result = run_agents(args.ticker, args.period, args.interval, args.max_position_pct)
    print_result(result)

    if args.json_out:
        payload = asdict(result)
        payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved JSON result to {args.json_out}")


if __name__ == "__main__":
    main()
