from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / "AppData" / "Local" / "hermes"))
DEFAULT_HERMES_AGENT_DIR = DEFAULT_HERMES_HOME / "hermes-agent"
DEFAULT_HERMES_PYTHON = DEFAULT_HERMES_AGENT_DIR / "venv" / "Scripts" / "python.exe"
DEFAULT_MODEL = "trading-hermes-qwen-lite:latest"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_TIMEOUT_SECONDS = 480


class HermesTradingError(RuntimeError):
    pass


def run_hermes_trading_agents(
    ticker: str,
    period: str = "6mo",
    max_position_pct: float = 10.0,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    hermes_agent_dir: Path = DEFAULT_HERMES_AGENT_DIR,
    hermes_python: Path = DEFAULT_HERMES_PYTHON,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    from dataclasses import asdict

    from simple_trading_agents import load_yahoo_history, run_agents_from_history

    history = load_yahoo_history(ticker, period=period, interval="1d")
    rule_result = run_agents_from_history(ticker, history, max_position_pct=max_position_pct)
    payload = {
        "ticker": ticker,
        "period": period,
        "max_position_pct": max_position_pct,
        "rule_based_agents": {
            "researcher": asdict(rule_result.researcher),
            "trader": asdict(rule_result.trader),
            "risk_manager": asdict(rule_result.risk_manager),
        },
    }

    hermes_result = run_hermes_review_payload(
        payload=payload,
        model=model,
        base_url=base_url,
        hermes_agent_dir=hermes_agent_dir,
        hermes_python=hermes_python,
        session_id=f"trading_hermes_{ticker.replace('^', '').lower()}",
        timeout_seconds=timeout_seconds,
    )
    return {
        "ticker": ticker,
        "period": period,
        "model": model,
        "base_url": base_url,
        "rule_based_agents": payload["rule_based_agents"],
        **hermes_result,
    }


def run_hermes_review_payload(
    payload: dict[str, Any],
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    hermes_agent_dir: Path = DEFAULT_HERMES_AGENT_DIR,
    hermes_python: Path = DEFAULT_HERMES_PYTHON,
    session_id: str = "trading_hermes_review",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    prompt = _build_prompt(payload)
    result = _call_hermes_worker(
        prompt=prompt,
        model=model,
        base_url=base_url,
        hermes_agent_dir=hermes_agent_dir,
        hermes_python=hermes_python,
        session_id=session_id,
        timeout_seconds=timeout_seconds,
    )
    raw_response = str(result.get("final_response", ""))
    parsed = _parse_json_response(raw_response)
    return {
        "model": model,
        "base_url": base_url,
        "hermes_agents": parsed,
        "hermes_raw_response": raw_response,
        "hermes_runtime": {
            key: result.get(key)
            for key in (
                "session_id",
                "completed",
                "failed",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "turn_exit_reason",
            )
            if isinstance(result, dict) and key in result
        },
    }


def _call_hermes_worker(
    prompt: str,
    model: str,
    base_url: str,
    hermes_agent_dir: Path,
    hermes_python: Path,
    session_id: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    if not hermes_python.exists():
        raise HermesTradingError(f"Hermes Python was not found at {hermes_python}.")
    request = {
        "prompt": prompt,
        "model": model,
        "base_url": base_url,
        "hermes_agent_dir": str(hermes_agent_dir),
        "session_id": session_id,
    }
    completed = subprocess.run(
        [str(hermes_python), str(Path(__file__).resolve()), "--worker"],
        input=json.dumps(request, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise HermesTradingError(
            "Hermes worker failed.\n"
            f"stdout:\n{completed.stdout[-1000:]}\n"
            f"stderr:\n{completed.stderr[-2000:]}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise HermesTradingError(
            "Hermes worker returned non-JSON output.\n"
            f"stdout:\n{completed.stdout[-2000:]}\n"
            f"stderr:\n{completed.stderr[-2000:]}"
        ) from exc


def _worker_main() -> None:
    request = json.loads(sys.stdin.read())
    hermes_agent_dir = Path(request["hermes_agent_dir"])
    _prepare_hermes_imports(hermes_agent_dir)
    from run_agent import AIAgent  # type: ignore

    with contextlib.redirect_stdout(sys.stderr):
        agent = AIAgent(
            model=request["model"],
            provider="custom",
            base_url=request["base_url"],
            api_key="ollama-local",
            quiet_mode=True,
            enabled_toolsets=[],
            skip_memory=True,
            session_id=request["session_id"],
        )
        result = agent.run_conversation(user_message=request["prompt"])
    print(json.dumps(result if isinstance(result, dict) else {"final_response": str(result)}, ensure_ascii=False))


def _prepare_hermes_imports(hermes_agent_dir: Path) -> None:
    if not hermes_agent_dir.exists():
        raise HermesTradingError(
            f"Hermes Agent code was not found at {hermes_agent_dir}. Install Hermes first."
        )
    sys.path.insert(0, str(hermes_agent_dir))


def _build_prompt(payload: dict[str, Any]) -> str:
    return (
        "You are running a compact Hermes multi-agent market review. "
        "Use only the supplied Yahoo Finance and rule-based agent data. "
        "This is educational and not financial advice.\n\n"
        "Return strict JSON only, with exactly this shape:\n"
        "{\n"
        '  "researcher": {"decision": "BULLISH|NEUTRAL|BEARISH", "confidence": 0.0, '
        '"summary": "one Korean sentence", "reasons": ["two short Korean strings"], '
        '"cautions": ["one short Korean risk string"]},\n'
        '  "trader": {"decision": "BUY|HOLD|SELL", "confidence": 0.0, '
        '"summary": "one Korean sentence", "reasons": ["two short Korean strings"], '
        '"cautions": ["one short Korean risk string"]},\n'
        '  "risk_manager": {"decision": "APPROVE|REDUCE|REJECT", "confidence": 0.0, '
        '"summary": "one Korean sentence", "reasons": ["two short Korean strings"], '
        '"cautions": ["one short Korean risk string"]}\n'
        "}\n\n"
        "Agent responsibilities:\n"
        "- Researcher: interpret trend, momentum, volatility, and limitations.\n"
        "- Trader: convert the evidence into a tentative trade stance.\n"
        "- Risk Manager: be conservative about drawdown, volatility, and position size.\n\n"
        "Payload:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _parse_json_response(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise HermesTradingError(f"Hermes did not return JSON: {text[:500]}")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise HermesTradingError(f"Hermes returned malformed JSON: {text[:500]}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a compact Hermes Agent trading review.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ticker", default="^KS11")
    parser.add_argument("--period", default="6mo")
    parser.add_argument("--max-position-pct", type=float, default=10.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.worker:
        _worker_main()
        return

    result = run_hermes_trading_agents(
        ticker=args.ticker,
        period=args.period,
        max_position_pct=args.max_position_pct,
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout,
    )
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
