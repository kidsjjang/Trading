from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_OLLAMA_URL = "http://localhost:11434"


class LocalLlmError(RuntimeError):
    pass


@dataclass
class LocalLlmAgentResult:
    role: str
    model: str
    decision: str
    confidence: float
    summary: str
    reasons: list[str]
    cautions: list[str]
    raw_response: str


def list_ollama_models(base_url: str = DEFAULT_OLLAMA_URL) -> list[str]:
    data = _ollama_request(f"{base_url}/api/tags", payload=None)
    return [model["name"] for model in data.get("models", []) if model.get("name")]


def run_local_llm_agents(
    model: str,
    analysis_payload: dict[str, Any],
    base_url: str = DEFAULT_OLLAMA_URL,
) -> dict[str, LocalLlmAgentResult]:
    return {
        "researcher": _run_single_agent(
            model=model,
            role="Researcher",
            role_instruction=(
                "Interpret the technical evidence. Do not make a final trade. "
                "Focus on trend, momentum, volatility, and data limitations."
            ),
            analysis_payload=analysis_payload,
            base_url=base_url,
        ),
        "trader": _run_single_agent(
            model=model,
            role="Trader",
            role_instruction=(
                "Convert the research evidence into a tentative BUY, HOLD, or SELL view. "
                "Respect the rule-based trader output, but you may disagree if you explain why."
            ),
            analysis_payload=analysis_payload,
            base_url=base_url,
        ),
        "risk_manager": _run_single_agent(
            model=model,
            role="Risk Manager",
            role_instruction=(
                "Review the trader view conservatively. Focus on position sizing, volatility, "
                "drawdown, and what could go wrong. You may downgrade risk exposure."
            ),
            analysis_payload=analysis_payload,
            base_url=base_url,
        ),
    }


def _run_single_agent(
    model: str,
    role: str,
    role_instruction: str,
    analysis_payload: dict[str, Any],
    base_url: str,
) -> LocalLlmAgentResult:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a local LLM agent inside an educational multi-agent market "
                "analysis dashboard. This is not financial advice. Use only the supplied "
                "data. Be concise. Return strict JSON only. Write summary, reasons, and "
                "cautions in Korean. Do not include markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Role: {role}\n"
                f"Role instruction: {role_instruction}\n\n"
                "Return JSON with exactly these keys:\n"
                "{\n"
                '  "decision": "BUY | HOLD | SELL | BULLISH | NEUTRAL | BEARISH",\n'
                '  "confidence": 0.0,\n'
                '  "summary": "one short Korean sentence",\n'
                '  "reasons": ["two short Korean strings, max 60 chars each"],\n'
                '  "cautions": ["one short Korean risk string, max 60 chars"]\n'
                "}\n\n"
                "Market and rule-based agent payload:\n"
                f"{json.dumps(analysis_payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]
    raw_response = _ollama_chat(model=model, messages=messages, base_url=base_url)
    parsed = _parse_json_response(raw_response)
    return LocalLlmAgentResult(
        role=role,
        model=model,
        decision=str(parsed.get("decision", "NEUTRAL")).upper(),
        confidence=_coerce_confidence(parsed.get("confidence", 0.0)),
        summary=str(parsed.get("summary", "")).strip(),
        reasons=_coerce_string_list(parsed.get("reasons", [])),
        cautions=_coerce_string_list(parsed.get("cautions", [])),
        raw_response=raw_response,
    )


def _ollama_chat(model: str, messages: list[dict[str, str]], base_url: str) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 500,
        },
    }
    data = _ollama_request(f"{base_url}/api/chat", payload=payload)
    message = data.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise LocalLlmError("Ollama returned no assistant content.")
    return content.strip()


def _ollama_request(url: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise LocalLlmError(
            "Could not reach Ollama at localhost:11434. Start Ollama and try again."
        ) from exc
    except json.JSONDecodeError as exc:
        raise LocalLlmError("Ollama returned invalid JSON.") from exc


def _parse_json_response(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LocalLlmError(f"Local LLM did not return JSON: {text[:300]}")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LocalLlmError(f"Local LLM returned malformed JSON: {text[:300]}") from exc


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if confidence > 1:
        confidence = confidence / 100
    return max(0.0, min(1.0, confidence))


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []
