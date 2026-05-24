import time

import requests
import streamlit as st

API_URL = "http://localhost:8000/api/analyze"
STATUS_URL_TEMPLATE = "http://localhost:8000/api/status/{run_id}"
HEATMAP_URL_TEMPLATE = "http://localhost:8000/api/heatmap/{run_id}"

DARK_CSS = """
<style>
section.main {background-color: #020617; color: #e2e8f0;}
[data-testid='metric-container'] {background: #111827; border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 24px; padding: 18px;}
.stButton>button {background-color: #2563eb; color: white; font-size: 1.05rem; font-weight: 700; border-radius: 18px; padding: 1rem 1.4rem; border: none; box-shadow: 0 16px 40px rgba(37, 99, 235, 0.2);}
.stButton>button:hover {background-color: #1d4ed8;}
.stButton>button:focus {outline: 3px solid rgba(59, 130, 246, 0.4);}
div.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.css-1d391kg {background-color: #020617 !important;}
.css-10trblm {background-color: #020617 !important;}
.css-1avcm0n, .css-1y1w2u2 {color: #94a3b8 !important;}
</style>
"""


def call_analyze_api() -> dict:
    try:
        response = requests.post(API_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as err:
        return {"error": str(err)}


def fetch_status(run_id: str) -> dict:
    try:
        response = requests.get(STATUS_URL_TEMPLATE.format(run_id=run_id), timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as err:
        return {"error": str(err)}


def fetch_heatmap(run_id: str) -> dict:
    try:
        response = requests.get(HEATMAP_URL_TEMPLATE.format(run_id=run_id), timeout=10)
        response.raise_for_status()
        return {"image_bytes": response.content}
    except requests.RequestException as err:
        return {"error": str(err)}


def poll_run_status(run_id: str, max_attempts: int = 12, interval_seconds: int = 5) -> dict:
    last_status = {}
    with st.spinner("Agents are gathering and analyzing data..."):
        for attempt in range(1, max_attempts + 1):
            status_response = fetch_status(run_id)
            if "error" in status_response:
                return status_response

            last_status = status_response
            status = status_response.get("status", "unknown")
            st.write(f"Checking status (attempt {attempt}/{max_attempts}): {status}")

            if status.lower() == "completed":
                return status_response

            if attempt < max_attempts:
                time.sleep(interval_seconds)

    return last_status


def render_header() -> None:
    st.set_page_config(page_title="Silent Outbreak Predictor", layout="wide")
    st.markdown(DARK_CSS, unsafe_allow_html=True)
    st.markdown("# Silent Outbreak Predictor")
    st.markdown("### AI-powered regional surveillance analytics for outbreak detection and mission-critical operations.")
    st.markdown("---")


def render_sidebar() -> None:
    st.sidebar.header("Dashboard Controls")
    st.sidebar.write("Launch intelligent surveillance analysis, monitor agent health, and review live system metrics.")
    st.sidebar.markdown("---")
    st.sidebar.metric(label="Connected Agents", value="7")
    st.sidebar.metric(label="Live Regions", value="16")
    st.sidebar.metric(label="Data Streams", value="24")
    st.sidebar.markdown("---")
    st.sidebar.write("**Guidance:**")
    st.sidebar.write("- Run surveillance to trigger regional analysis.")
    st.sidebar.write("- Polling updates appear during status checks.")
    st.sidebar.write("- Heatmap and summary metrics render once complete.")


def render_dashboard() -> None:
    with st.container():
        header_left, header_right = st.columns([3, 1], gap="large")
        with header_left:
            st.markdown("#### Operational Intelligence")
            st.write(
                "A modern analytics dashboard designed for visibility into outbreak risk, model confidence, and live response readiness."
            )
        with header_right:
            st.metric(label="System Uptime", value="99.96%")
            st.metric(label="Latest Run", value="Ready")

    with st.container():
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3, gap="large")
        kpi_col1.metric(label="High-risk regions", value="4")
        kpi_col2.metric(label="Active alerts", value="12")
        kpi_col3.metric(label="Average signal score", value="78%")

    with st.container():
        main_left, main_right = st.columns([2, 1.05], gap="large")
        with main_left:
            st.markdown("#### Trend analytics")
            trend_data = {
                "Signals": [32, 45, 37, 50, 66, 71, 82],
                "Alerts": [5, 7, 9, 12, 10, 14, 18],
            }
            st.line_chart(trend_data)

            st.markdown("#### Health overview")
            health_data = {
                "Region": ["North", "South", "East", "West"],
                "Risk Score": [82, 74, 68, 90],
            }
            st.bar_chart(health_data)

        with main_right:
            st.markdown("#### Run Regional Surveillance")
            if st.button("Run Regional Surveillance"):
                result = call_analyze_api()
                if "error" in result:
                    st.error(f"Failed to start surveillance: {result['error']}")
                else:
                    run_id = result.get("run_id")
                    if not run_id:
                        st.error("No run_id returned from the analyze API.")
                        return

                    status_result = poll_run_status(run_id)
                    if "error" in status_result:
                        st.error(f"Status polling failed: {status_result['error']}")
                        return

                    heatmap_response = fetch_heatmap(run_id)
                    if "error" in heatmap_response:
                        st.error(f"Failed to fetch heatmap: {heatmap_response['error']}")
                        return

                    st.success("Regional surveillance executed successfully.")
                    st.balloons()
                    st.session_state["surveillance_ran"] = True
                    st.session_state["run_id"] = run_id
                    st.session_state["status_result"] = status_result
                    st.session_state["heatmap_image"] = heatmap_response.get("image_bytes")

            st.markdown("---")
            st.markdown("#### Action summary")
            st.write("Use this action panel to start analysis and then scroll down for heatmap, metrics, and JSON insights.")

    if st.session_state.get("surveillance_ran"):
        st.markdown("---")
        run_id = st.session_state.get("run_id")
        status_result = st.session_state.get("status_result")
        heatmap_image = st.session_state.get("heatmap_image")

        if heatmap_image:
            st.markdown("#### Generated heatmap")
            st.image(heatmap_image, caption="Heatmap generated from surveillance run", use_column_width=True)

        if status_result:
            confidence = status_result.get("confidence_score", "N/A")
            reasoning = status_result.get("explainable_reasoning", "No explanation provided")
            if run_id:
                st.info(f"Surveillance run_id: {run_id}")

            final_cols = st.columns(2, gap="large")
            final_cols[0].metric(label="Confidence Score", value=f"{confidence}")
            final_cols[1].metric(label="Explainable Reasoning", value=str(reasoning)[:60] + ("..." if len(str(reasoning)) > 60 else ""))

            st.markdown("#### Full analysis output")
            st.json(status_result)

        st.markdown("#### Region alert breakdown")
        chart_data = {
            "Region": ["North", "South", "East", "West"],
            "Risk Score": [82, 74, 68, 90],
        }
        st.bar_chart(chart_data)


def main() -> None:
    if "surveillance_ran" not in st.session_state:
        st.session_state["surveillance_ran"] = False
    if "run_id" not in st.session_state:
        st.session_state["run_id"] = None
    if "status_result" not in st.session_state:
        st.session_state["status_result"] = None
    if "heatmap_image" not in st.session_state:
        st.session_state["heatmap_image"] = None

    render_header()
    render_sidebar()
    render_dashboard()


if __name__ == "__main__":
    main()
