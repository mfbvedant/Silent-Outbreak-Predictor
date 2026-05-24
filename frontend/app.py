"""
app.py — Streamlit frontend for the Silent Outbreak Predictor.

Premium dark-themed surveillance dashboard connecting to the FastAPI
backend for CrewAI-powered outbreak analysis.
"""

import base64
import time
from pathlib import Path

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------
BACKEND_BASE = "http://localhost:8000"
API_HEALTH = f"{BACKEND_BASE}/api/health"
API_ANALYZE = f"{BACKEND_BASE}/api/analyze"
API_STATUS = f"{BACKEND_BASE}/api/status/{{run_id}}"
API_HEATMAP = f"{BACKEND_BASE}/api/heatmap/{{run_id}}"

# ---------------------------------------------------------------------------
# Hero banner (base64 encoded for inline display)
# ---------------------------------------------------------------------------
HERO_PATH = Path(__file__).parent / "hero_banner.png"


def _get_hero_base64() -> str:
    if HERO_PATH.exists():
        return base64.b64encode(HERO_PATH.read_bytes()).decode()
    return ""


# ---------------------------------------------------------------------------
# Premium Dark Theme CSS
# ---------------------------------------------------------------------------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global ────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
section.main {
    background: linear-gradient(180deg, #020617 0%, #0f172a 40%, #020617 100%);
    color: #e2e8f0;
}
div.block-container {
    padding-top: 0.5rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.stApp > header { background: transparent; }

/* ── Hero banner ───────────────────────────────────────────────────── */
.hero-container {
    position: relative;
    border-radius: 20px;
    overflow: hidden;
    margin-bottom: 1.5rem;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
}
.hero-container img {
    width: 100%;
    height: 220px;
    object-fit: cover;
    display: block;
    filter: brightness(0.6);
}
.hero-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(2,6,23,0.85) 0%, rgba(15,23,42,0.6) 50%, rgba(2,6,23,0.85) 100%);
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 2rem 2.5rem;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1rem;
    color: #94a3b8;
    font-weight: 400;
    margin: 0;
    max-width: 600px;
}

/* ── Metric cards ──────────────────────────────────────────────────── */
[data-testid='metric-container'] {
    background: linear-gradient(135deg, rgba(30,41,59,0.8) 0%, rgba(15,23,42,0.9) 100%);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid='metric-container']:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.08);
}
[data-testid='metric-container'] label {
    color: #64748b !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    font-size: 0.7rem !important;
    letter-spacing: 1px;
}
[data-testid='metric-container'] [data-testid='stMetricValue'] {
    color: #f1f5f9 !important;
    font-weight: 700 !important;
}

/* ── Glassmorphism cards ───────────────────────────────────────────── */
.glass-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.6) 0%, rgba(15,23,42,0.8) 100%);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(148,163,184,0.1);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}
.glass-card h4 {
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    margin-bottom: 0.8rem;
}

/* ── Buttons ───────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
    color: white !important;
    font-size: 1rem;
    font-weight: 700;
    border-radius: 14px;
    padding: 0.9rem 1.6rem;
    border: none !important;
    box-shadow: 0 8px 30px rgba(59,130,246,0.35);
    transition: all 0.3s ease;
    letter-spacing: 0.3px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(59,130,246,0.5);
}
.stButton > button:active {
    transform: translateY(0px);
}

/* ── Section dividers ──────────────────────────────────────────────── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(100,116,139,0.3) 50%, transparent 100%);
    margin: 1.5rem 0;
    border: none;
}

/* ── Status badges ─────────────────────────────────────────────────── */
.status-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.status-online {
    background: rgba(34,197,94,0.15);
    color: #4ade80;
    border: 1px solid rgba(34,197,94,0.3);
}
.status-offline {
    background: rgba(239,68,68,0.15);
    color: #f87171;
    border: 1px solid rgba(239,68,68,0.3);
}
.status-processing {
    background: rgba(251,191,36,0.15);
    color: #fbbf24;
    border: 1px solid rgba(251,191,36,0.3);
    animation: pulse-glow 2s infinite;
}
@keyframes pulse-glow {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

/* ── Sidebar ───────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #020617 100%) !important;
    border-right: 1px solid rgba(148,163,184,0.08);
}
[data-testid="stSidebar"] [data-testid='metric-container'] {
    background: rgba(30,41,59,0.5);
    border: 1px solid rgba(148,163,184,0.08);
    padding: 12px 16px;
}

/* ── Charts ────────────────────────────────────────────────────────── */
[data-testid="stVegaLiteChart"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Expander ──────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: rgba(30,41,59,0.5) !important;
    border-radius: 12px !important;
}

/* ── Hide Streamlit chrome ─────────────────────────────────────────── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def check_backend_health() -> dict:
    try:
        resp = requests.get(API_HEALTH, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return {"status": "offline"}


def call_analyze_api() -> dict:
    try:
        resp = requests.post(API_ANALYZE, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as err:
        return {"error": str(err)}


def fetch_status(run_id: str) -> dict:
    try:
        resp = requests.get(API_STATUS.format(run_id=run_id), timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as err:
        return {"error": str(err)}


def fetch_heatmap(run_id: str) -> dict:
    try:
        resp = requests.get(API_HEATMAP.format(run_id=run_id), timeout=15)
        resp.raise_for_status()
        return {"image_bytes": resp.content}
    except requests.RequestException as err:
        return {"error": str(err)}


def poll_run_status(
    run_id: str, max_attempts: int = 60, interval_seconds: int = 5
) -> dict:
    """Poll with animated progress bar — 60 × 5s = 5 min max."""
    progress_bar = st.progress(0, text="")
    status_text = st.empty()
    last_status = {}

    phases = [
        ("🔍 Gatherer agent scanning health bulletins...", 0.0, 0.3),
        ("🧬 Analyst agent assessing outbreak risk...", 0.3, 0.7),
        ("📊 Visualizer agent generating risk plot...", 0.7, 0.95),
    ]

    for attempt in range(1, max_attempts + 1):
        status_response = fetch_status(run_id)
        if "error" in status_response:
            progress_bar.empty()
            status_text.empty()
            return status_response

        last_status = status_response
        current_status = status_response.get("status", "unknown")

        # Estimate which phase we're in
        progress = min(attempt / max_attempts, 0.99)
        phase_idx = min(int(progress / 0.33), 2)
        phase_msg = phases[phase_idx][0]

        progress_bar.progress(progress, text=phase_msg)
        status_text.markdown(
            f'<span class="status-badge status-processing">'
            f'PROCESSING — Attempt {attempt}/{max_attempts}</span>',
            unsafe_allow_html=True,
        )

        if current_status.lower() == "completed":
            progress_bar.progress(1.0, text="✅ Analysis complete!")
            status_text.markdown(
                '<span class="status-badge status-online">COMPLETED</span>',
                unsafe_allow_html=True,
            )
            time.sleep(0.5)
            status_text.empty()
            return status_response

        if attempt < max_attempts:
            time.sleep(interval_seconds)

    progress_bar.progress(1.0, text="⏱️ Polling timed out — pipeline may still be running")
    status_text.empty()
    return last_status


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------
def render_header() -> None:
    st.set_page_config(
        page_title="Silent Outbreak Predictor",
        page_icon="🦠",
        layout="wide",
    )
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    # Hero banner
    hero_b64 = _get_hero_base64()
    if hero_b64:
        st.markdown(
            f"""
            <div class="hero-container">
                <img src="data:image/png;base64,{hero_b64}" alt="banner" />
                <div class="hero-overlay">
                    <h1 class="hero-title">Silent Outbreak Predictor</h1>
                    <p class="hero-subtitle">
                        AI-powered multi-agent surveillance system for real-time
                        epidemic detection and risk analysis across Maharashtra.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown("# 🦠 Silent Outbreak Predictor")
        st.markdown(
            "### AI-powered surveillance for outbreak detection and risk analysis."
        )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### ⚙️ Control Panel")
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Backend connectivity
        health = check_backend_health()
        if health.get("status") == "ok":
            pipeline_mode = "Real AI Pipeline" if health.get("pipeline_available") else "Simulation"
            st.markdown(
                f'<span class="status-badge status-online">● CONNECTED</span> '
                f'<span style="color:#64748b;font-size:0.8rem;">v{health.get("version", "?")}</span>',
                unsafe_allow_html=True,
            )
            st.metric(label="Pipeline", value=pipeline_mode)
        else:
            st.markdown(
                '<span class="status-badge status-offline">● OFFLINE</span>',
                unsafe_allow_html=True,
            )
            st.error("Start backend: `python -m uvicorn api:app`")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown("#### 📡 System Info")
        c1, c2 = st.columns(2)
        c1.metric(label="Agents", value="3")
        c2.metric(label="Region", value="MH")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown("#### 🔄 Pipeline Flow")
        st.markdown(
            """
            ```
            ┌─────────────┐
            │  🔍 Gatherer │ ← OSINT scraping
            └──────┬──────┘
                   ▼
            ┌─────────────┐
            │  🧬 Analyst  │ ← Risk scoring
            └──────┬──────┘
                   ▼
            ┌─────────────┐
            │ 📊 Visualizer│ ← Plot generation
            └─────────────┘
            ```
            """
        )


def render_dashboard() -> None:
    # ── KPI strip ──
    st.markdown(
        '<div class="glass-card"><h4>📈 Live Metrics</h4></div>',
        unsafe_allow_html=True,
    )
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    k1.metric(label="Active Agents", value="3", delta="Online")
    k2.metric(label="Target Region", value="Pune / MH")
    k3.metric(label="Data Sources", value="OSINT + News")
    k4.metric(label="Last Run", value=st.session_state.get("last_run_status", "—"))

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Main content area ──
    col_left, col_right = st.columns([2, 1.2], gap="large")

    with col_left:
        st.markdown(
            '<div class="glass-card"><h4>📈 Signal Trend (7-day)</h4></div>',
            unsafe_allow_html=True,
        )
        import pandas as pd
        trend_df = pd.DataFrame({
            "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "Signals": [32, 45, 37, 50, 66, 71, 82],
            "Alerts": [5, 7, 9, 12, 10, 14, 18],
        }).set_index("Day")
        st.line_chart(trend_df, color=["#3b82f6", "#f472b6"])

    with col_right:
        st.markdown(
            '<div class="glass-card"><h4>🚀 Launch Analysis</h4></div>',
            unsafe_allow_html=True,
        )
        st.write(
            "Deploy the 3-agent CrewAI pipeline to scan for outbreak "
            "signals in Pune & Maharashtra."
        )

        if st.button("▶  Run Surveillance", width="stretch", type="primary"):
            result = call_analyze_api()
            if "error" in result:
                st.error(f"❌ Failed: {result['error']}")
            else:
                run_id = result.get("run_id")
                if not run_id:
                    st.error("No run_id returned.")
                    return

                st.caption(f"🆔 `{run_id}`")
                status_result = poll_run_status(run_id)

                if "error" in status_result:
                    st.error(f"Polling failed: {status_result['error']}")
                    return

                if status_result.get("status") != "completed":
                    st.warning("Pipeline still running — try again shortly.")
                    return

                # Fetch heatmap
                heatmap_response = fetch_heatmap(run_id)

                # Store in session
                st.session_state["surveillance_ran"] = True
                st.session_state["run_id"] = run_id
                st.session_state["status_result"] = status_result
                st.session_state["heatmap_image"] = heatmap_response.get("image_bytes")
                st.session_state["heatmap_error"] = heatmap_response.get("error")
                st.session_state["last_run_status"] = "✅ Complete"

                st.success("✅ Surveillance complete!")
                st.balloons()

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Quick stats
        st.markdown("**Agent Architecture**")
        st.markdown(
            "- 🔍 **Gatherer** — `gpt-4o-mini`\n"
            "- 🧬 **Analyst** — `gpt-4o`\n"
            "- 📊 **Visualizer** — `gpt-4o-mini`"
        )

    # ── Results section ──
    if st.session_state.get("surveillance_ran"):
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        status_result = st.session_state.get("status_result", {})
        heatmap_image = st.session_state.get("heatmap_image")
        heatmap_error = st.session_state.get("heatmap_error")
        run_id = st.session_state.get("run_id")

        confidence = status_result.get("confidence_score", "N/A")
        disease = status_result.get("disease", "Unknown")
        region = status_result.get("region", "Unknown")

        # Results header
        st.markdown(
            '<div class="glass-card"><h4>📋 Analysis Results</h4></div>',
            unsafe_allow_html=True,
        )

        # KPI row
        r1, r2, r3 = st.columns(3, gap="medium")
        r1.metric(label="🦠 Disease Detected", value=str(disease))
        r2.metric(label="📍 Region", value=str(region))

        # Color-code the confidence score
        conf_val = confidence if isinstance(confidence, (int, float)) else 0
        r3.metric(label="⚡ Confidence Score", value=f"{confidence}")

        if run_id:
            st.caption(f"Run ID: `{run_id}`")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Heatmap + Reasoning side by side
        res_left, res_right = st.columns([1.4, 1], gap="large")

        with res_left:
            st.markdown(
                '<div class="glass-card"><h4>🗺️ Outbreak Risk Visualization</h4></div>',
                unsafe_allow_html=True,
            )
            if heatmap_image:
                st.image(
                    heatmap_image,
                    caption="Generated by AI Visualization Agent",
                    width="stretch",
                )
            elif heatmap_error:
                st.warning(f"Plot unavailable: {heatmap_error}")
            else:
                st.info("No visualization generated for this run.")

        with res_right:
            st.markdown(
                '<div class="glass-card"><h4>🧠 Explainable Reasoning</h4></div>',
                unsafe_allow_html=True,
            )
            reasoning = status_result.get(
                "explainable_reasoning", "No explanation provided."
            )
            st.markdown(
                f'<div style="color:#cbd5e1;line-height:1.7;font-size:0.9rem;">'
                f'{reasoning}</div>',
                unsafe_allow_html=True,
            )

        # Full JSON in expander
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        with st.expander("📄 Full API Response (JSON)", expanded=False):
            st.json(status_result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    for key in ["surveillance_ran", "run_id", "status_result",
                "heatmap_image", "heatmap_error", "last_run_status"]:
        if key not in st.session_state:
            st.session_state[key] = None if key != "surveillance_ran" else False

    render_header()
    render_sidebar()
    render_dashboard()


if __name__ == "__main__":
    main()
