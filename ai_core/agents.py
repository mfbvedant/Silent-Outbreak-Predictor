"""
agents.py — CrewAI Agent definitions for the Silent Outbreak Predictor pipeline.

Defines three specialized agents:
  1. gatherer_agent  — OSINT web scraping for health bulletins
  2. analyst_agent   — Epidemiological analysis and confidence scoring
  3. visualizer_agent — Matplotlib visualization and file output
"""

import io
import contextlib

from crewai import Agent
from crewai.tools import tool
from crewai_tools import ScrapeWebsiteTool

from config import fast_llm, smart_llm
from schemas import EpidemicPrediction


# ---------------------------------------------------------------------------
# Tool instances
# ---------------------------------------------------------------------------
scrape_tool = ScrapeWebsiteTool()          # OSINT web scraping capability


@tool("python_repl")
def python_repl_tool(code: str) -> str:
    """Execute a Python code snippet and return its stdout output.

    Use this tool to run self-contained Python scripts — for example,
    generating matplotlib plots and saving them to disk.  The code runs
    in the current process so it has access to all installed packages
    (matplotlib, pandas, os, etc.).

    Args:
        code: A complete, self-contained Python script as a string.
    """
    stdout_capture = io.StringIO()
    try:
        # Pydantic monkey-patches warnings.warn with a filtered_warn() that
        # does NOT accept Python 3.13's new `skip_file_prefixes` kwarg.
        # This crashes matplotlib (and other libs) when they call
        # warnings.warn(..., skip_file_prefixes=...).
        # Fix: temporarily restore the real C-level warn during execution.
        import warnings
        import _warnings

        patched_warn = warnings.warn          # save pydantic's version
        warnings.warn = _warnings.warn        # restore the real one

        exec_globals = {"__builtins__": __builtins__}

        with contextlib.redirect_stdout(stdout_capture):
            exec(code, exec_globals)  # noqa: S102

        warnings.warn = patched_warn          # restore pydantic's version
        output = stdout_capture.getvalue()
        return output if output else "Code executed successfully (no stdout output)."
    except Exception as e:
        # Make sure we restore even on failure
        try:
            warnings.warn = patched_warn  # type: ignore[possibly-undefined]
        except NameError:
            pass
        return f"Error during execution: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 1. OSINT Medical Intelligence Gatherer
# ---------------------------------------------------------------------------
gatherer_agent = Agent(
    role="OSINT Medical Intelligence Gatherer",
    goal=(
        "Systematically monitor official health bulletins (e.g. Pune Municipal Corporation, "
        "IDSP Maharashtra) and reputable news aggregators to extract disease names, affected "
        "locations, and incident dates for respiratory and infectious illnesses across the "
        "Pune and broader Maharashtra region. Strictly ignore advertisements, opinion pieces, "
        "and irrelevant news articles."
    ),
    backstory=(
        "You are a seasoned open-source intelligence analyst embedded in the WHO South-East "
        "Asia Regional Office. You have spent over a decade perfecting the craft of sifting "
        "through noisy public health data feeds to isolate actionable disease signals before "
        "they escalate into full-blown outbreaks. Your extraction precision has directly "
        "prevented two regional epidemics from going unnoticed."
    ),
    tools=[scrape_tool],
    llm=fast_llm,
    verbose=True,
    allow_delegation=False,
)


# ---------------------------------------------------------------------------
# 2. Senior Epidemiological Data Scientist
# ---------------------------------------------------------------------------
analyst_agent = Agent(
    role="Senior Epidemiological Data Scientist",
    goal=(
        "Cross-reference the raw OSINT data provided by the Gatherer agent against known "
        "epidemic indicators. Assess keyword severity (e.g. 'severe', 'outbreak', 'acute', "
        "'H1N1', 'epidemic', 'mortality'), calculate a mathematically grounded confidence_score "
        "on a 0.0–100.0 scale, and produce explainable reasoning that traces each score "
        "component back to its source evidence. Output a fully validated EpidemicPrediction object."
    ),
    backstory=(
        "You are a Senior Epidemiologist and Data Scientist with dual PhDs in Computational "
        "Epidemiology and Biostatistics from Johns Hopkins Bloomberg School of Public Health. "
        "You have built outbreak early-warning systems for the CDC and ECDC, and your scoring "
        "methodology has been peer-reviewed in The Lancet Infectious Diseases. You never guess — "
        "every number you produce is traceable to a quantitative justification."
    ),
    llm=smart_llm,
    verbose=True,
    allow_delegation=False,
)


# ---------------------------------------------------------------------------
# 3. Backend Python Visualization Engineer
# ---------------------------------------------------------------------------
visualizer_agent = Agent(
    role="Backend Python Visualization Engineer",
    goal=(
        "Write and execute a self-contained Python script using matplotlib to create a "
        "publication-quality bar plot of the outbreak confidence scores. The generated image "
        "MUST be saved precisely to the path '../data_outputs/outbreak_risk_{run_id}.png' "
        "relative to the ai_core directory. Include proper titles, axis labels, color coding "
        "by severity tier, and a clean visual style suitable for stakeholder briefings."
    ),
    backstory=(
        "You are a backend Python engineer at a global health analytics firm. You specialize "
        "in automated data visualization pipelines — transforming structured Pydantic model "
        "outputs into clear, actionable charts that are consumed by epidemiologists and "
        "government health officials. Your plots have been featured in WHO Situation Reports."
    ),
    tools=[python_repl_tool],
    llm=fast_llm,
    verbose=True,
    allow_delegation=False,
)
