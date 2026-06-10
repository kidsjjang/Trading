from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from local_llm_agents import LocalLlmAgentResult, LocalLlmError, list_ollama_models, run_local_llm_agents
from simple_trading_agents import load_yahoo_history, run_agents_from_history


DISCLAIMER = (
    "Educational prototype only. Yahoo Finance polling is not guaranteed tick-level real-time data, "
    "and these agent outputs are not financial advice."
)


st.set_page_config(
    page_title="Multi-Agent Trading Dashboard",
    page_icon="M",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(26, 115, 232, 0.12), transparent 32rem),
            linear-gradient(180deg, #f7faf7 0%, #eef3ec 100%);
        color: #193225;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] * {
        color: #193225;
    }
    [data-testid="stHeader"] {
        background: rgba(247, 250, 247, 0.78);
    }
    section[data-testid="stSidebar"] {
        background: #17231c;
    }
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] * {
        color: #f7faf7 !important;
    }
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #193225 !important;
    }
    .block-container {
        padding-top: 2rem;
    }
    [data-testid="stMetric"] {
        border: 1px solid rgba(31, 41, 55, 0.10);
        border-radius: 16px;
        padding: 0.85rem 1rem;
        background: rgba(255, 255, 255, 0.72);
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }
    [data-testid="stMetric"] * {
        color: #193225 !important;
    }
    .agent-card {
        border: 1px solid rgba(31, 41, 55, 0.12);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        background: rgba(255, 255, 255, 0.78);
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
        min-height: 210px;
        color: #193225;
    }
    .agent-card * {
        color: #193225 !important;
    }
    .agent-card h3 {
        margin: 0 0 0.55rem 0;
        color: #193225;
    }
    .small-muted {
        color: #647067;
        font-size: 0.9rem;
    }
    .decision-pill {
        display: inline-block;
        padding: 0.24rem 0.65rem;
        border-radius: 999px;
        background: #173f2a;
        color: #ffffff;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .agent-card .decision-pill {
        color: #ffffff !important;
    }
    .llm-note {
        border-left: 4px solid #174f35;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        background: rgba(255, 255, 255, 0.62);
        color: #193225;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def auto_refresh(enabled: bool, seconds: int) -> None:
    if not enabled:
        return
    components.html(
        f"""
        <script>
        const refreshMs = {max(seconds, 15) * 1000};
        setTimeout(() => window.parent.location.reload(), refreshMs);
        </script>
        """,
        height=0,
    )


@st.cache_data(show_spinner=False, ttl=30)
def get_history(ticker: str, period: str, interval: str):
    return load_yahoo_history(ticker, period=period, interval=interval)


@st.cache_data(show_spinner=False, ttl=120)
def get_llm_review(model: str, payload_json: str):
    return run_local_llm_agents(model=model, analysis_payload=json.loads(payload_json))


def make_price_chart(history, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["Close"],
            mode="lines",
            name="Close",
            line={"color": "#174f35", "width": 2.5},
        )
    )

    if len(history) >= 20:
        fig.add_trace(
            go.Scatter(
                x=history.index,
                y=history["Close"].rolling(20).mean(),
                mode="lines",
                name="SMA 20",
                line={"color": "#d08c2f", "width": 1.7},
            )
        )

    if len(history) >= 50:
        fig.add_trace(
            go.Scatter(
                x=history.index,
                y=history["Close"].rolling(50).mean(),
                mode="lines",
                name="SMA 50",
                line={"color": "#3867d6", "width": 1.7},
            )
        )

    fig.update_layout(
        title=title,
        height=430,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        font={"color": "#193225"},
        title_font={"color": "#193225", "size": 20},
        legend_font={"color": "#193225"},
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.72)",
        xaxis_title=None,
        yaxis_title="Price",
        xaxis={
            "color": "#193225",
            "gridcolor": "rgba(25, 50, 37, 0.12)",
            "linecolor": "rgba(25, 50, 37, 0.35)",
            "tickfont": {"color": "#193225"},
            "zerolinecolor": "rgba(25, 50, 37, 0.18)",
        },
        yaxis={
            "color": "#193225",
            "gridcolor": "rgba(25, 50, 37, 0.12)",
            "linecolor": "rgba(25, 50, 37, 0.35)",
            "tickfont": {"color": "#193225"},
            "title_font": {"color": "#193225"},
            "zerolinecolor": "rgba(25, 50, 37, 0.18)",
        },
    )
    return fig


def render_agent_card(title: str, lines: list[str], footer: str | None = None) -> None:
    body = "".join(f"<li>{line}</li>" for line in lines)
    footer_html = f"<p class='small-muted'>{footer}</p>" if footer else ""
    st.markdown(
        f"""
        <div class="agent-card">
            <h3>{title}</h3>
            <ul>{body}</ul>
            {footer_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_llm_result(result: LocalLlmAgentResult) -> None:
    st.markdown(f"#### {result.role}")
    st.metric("LLM View", result.decision, f"{result.confidence:.0%} confidence")
    st.write(result.summary or "No summary returned.")
    if result.reasons:
        st.write("Reasons")
        for item in result.reasons:
            st.write(f"- {item}")
    if result.cautions:
        st.write("Cautions")
        for item in result.cautions:
            st.write(f"- {item}")


with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Ticker", value="^KS11", help="KOSPI Composite Index uses ^KS11 on Yahoo Finance.").strip().upper() or "^KS11"
    analysis_period = st.selectbox("Analysis window", ["3mo", "6mo", "1y", "2y"], index=1)
    max_position_pct = st.slider("Max position cap", min_value=1.0, max_value=25.0, value=10.0, step=1.0)
    refresh_seconds = st.slider("Refresh seconds", min_value=15, max_value=300, value=60, step=15)
    enable_auto_refresh = st.toggle("Auto refresh", value=True)
    st.divider()
    st.subheader("Local LLM")
    enable_local_llm = st.toggle(
        "Use local LLM agents",
        value=False,
        help="Requires Ollama running on this computer. This will not work from Streamlit Cloud.",
    )
    selected_llm_model = "qwen3:4b-instruct"
    if enable_local_llm:
        try:
            available_models = list_ollama_models()
            if available_models:
                default_index = available_models.index("qwen3:4b-instruct") if "qwen3:4b-instruct" in available_models else 0
                selected_llm_model = st.selectbox("Ollama model", available_models, index=default_index)
            else:
                st.warning("Ollama is running, but no local models are installed.")
        except LocalLlmError as exc:
            st.warning(str(exc))
            enable_local_llm = False
    if enable_local_llm and enable_auto_refresh:
        st.caption("Auto refresh is paused while local LLM agents are enabled.")
    if st.button("Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

auto_refresh(enable_auto_refresh and not enable_local_llm, refresh_seconds)

st.title("Multi-Agent Trading Dashboard")
st.caption(DISCLAIMER)

try:
    with st.spinner(f"Loading Yahoo Finance data for {ticker}..."):
        daily_history = get_history(ticker, analysis_period, "1d")
        try:
            intraday_history = get_history(ticker, "1d", "1m")
        except Exception:
            intraday_history = daily_history.tail(60)

    result = run_agents_from_history(ticker, daily_history, max_position_pct=max_position_pct)
    latest_price = float(intraday_history["Close"].iloc[-1])
    previous_close = float(daily_history["Close"].iloc[-2]) if len(daily_history) > 1 else latest_price
    change_pct = ((latest_price / previous_close) - 1) * 100 if previous_close else 0
    last_source_time = intraday_history.index[-1]
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Latest Price", f"{latest_price:,.2f}", f"{change_pct:.2f}% vs prev close")
    kpi2.metric("Final Action", result.risk_manager.final_action, f"{result.risk_manager.approved_position_pct:.1f}% size")
    kpi3.metric("Research Score", result.researcher.score, f"RSI {result.researcher.rsi_14}")
    kpi4.metric("Risk Level", result.risk_manager.risk_level, f"Vol {result.researcher.volatility_20d_annual_pct}%")

    st.markdown(
        f"<span class='small-muted'>Yahoo source timestamp: {last_source_time} | Dashboard fetched: {fetched_at}</span>",
        unsafe_allow_html=True,
    )

    st.plotly_chart(make_price_chart(daily_history, f"{ticker} Daily Price With Moving Averages"), use_container_width=True)

    researcher_lines = [
        f"Last daily close: {result.researcher.last_price}",
        f"SMA20 / SMA50: {result.researcher.sma_20} / {result.researcher.sma_50}",
        f"20D return: {result.researcher.return_20d_pct}%",
        f"90D max drawdown: {result.researcher.max_drawdown_90d_pct}%",
    ]
    trader_lines = [
        f"Action: <span class='decision-pill'>{result.trader.action}</span>",
        f"Confidence: {result.trader.confidence}",
        f"Proposed position: {result.trader.proposed_position_pct}%",
        *result.trader.rationale,
    ]
    risk_lines = [
        f"Final action: <span class='decision-pill'>{result.risk_manager.final_action}</span>",
        f"Approved position: {result.risk_manager.approved_position_pct}%",
        *result.risk_manager.notes,
    ]

    left, middle, right = st.columns(3)
    with left:
        render_agent_card("Researcher", researcher_lines, "Technical view from daily Yahoo Finance bars.")
    with middle:
        render_agent_card("Trader", trader_lines, "Converts the research score into BUY / SELL / HOLD.")
    with right:
        render_agent_card("Risk Manager", risk_lines, "Caps exposure based on volatility, drawdown, and confidence.")

    if enable_local_llm:
        st.subheader("Local LLM Agent Overlay")
        st.markdown(
            "<div class='llm-note'>Local LLM output is a second opinion generated by Ollama on this computer. "
            "It is not financial advice and may disagree with the rule-based agents.</div>",
            unsafe_allow_html=True,
        )
        payload_json = json.dumps(
            {
                "researcher": asdict(result.researcher),
                "trader": asdict(result.trader),
                "risk_manager": asdict(result.risk_manager),
            },
            sort_keys=True,
        )
        try:
            with st.spinner(f"Running local LLM agents with {selected_llm_model}..."):
                llm_review = get_llm_review(selected_llm_model, payload_json)
            llm_left, llm_middle, llm_right = st.columns(3)
            with llm_left:
                with st.container(border=True):
                    render_llm_result(llm_review["researcher"])
            with llm_middle:
                with st.container(border=True):
                    render_llm_result(llm_review["trader"])
            with llm_right:
                with st.container(border=True):
                    render_llm_result(llm_review["risk_manager"])
        except LocalLlmError as exc:
            st.error(str(exc))

    with st.expander("Research thesis and risks", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Thesis")
            st.write(result.researcher.thesis or ["No positive thesis items."])
        with c2:
            st.subheader("Risks")
            st.write(result.researcher.risks or ["No risk items."])

    with st.expander("Recent data sample", expanded=False):
        st.dataframe(daily_history.tail(20), use_container_width=True)

except Exception as exc:
    st.error(f"Dashboard could not load data for {ticker}.")
    st.exception(exc)
