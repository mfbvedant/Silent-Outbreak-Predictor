"""
main.py — Silent Outbreak Predictor pipeline orchestrator.

Composes the CrewAI sequential pipeline:
  gather_task  →  analyze_task  →  visualize_task

Usage:
    python main.py              # runs with default run_id "test_001"
    python main.py <run_id>     # runs with a custom run_id
"""

import sys

from crewai import Crew, Task, Process

from agents import gatherer_agent, analyst_agent, visualizer_agent
from schemas import EpidemicPrediction


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

gather_task = Task(
    description=(
        "Scan official municipal health bulletins (Pune Municipal Corporation, IDSP "
        "Maharashtra), WHO SEARO feeds, and reputable news aggregators for any respiratory "
        "or infectious disease events reported in the Pune and broader Maharashtra region "
        "over the last 7 days.\n\n"
        "For every event you find, extract:\n"
        "  • Disease / pathogen name\n"
        "  • Affected geographic location (city, district, or state)\n"
        "  • Date or date range of the reported incident\n"
        "  • A brief summary of the report\n\n"
        "Ignore advertisements, opinion editorials, and unrelated news. Focus strictly on "
        "verified health data sources."
    ),
    expected_output=(
        "A structured raw-text summary listing each extracted disease event with its "
        "disease name, location, date(s), and a one-line synopsis of the source report."
    ),
    agent=gatherer_agent,
)

analyze_task = Task(
    description=(
        "You will receive a raw-text summary of disease events from the OSINT Gatherer.\n\n"
        "For each event, perform the following analysis:\n"
        "  1. Cross-reference the disease name against known epidemic indicators.\n"
        "  2. Assess keyword severity — flag terms such as 'severe', 'outbreak', 'acute', "
        "'epidemic', 'mortality', 'H1N1', 'influenza', 'respiratory distress'.\n"
        "  3. Calculate a mathematically grounded confidence_score on a 0.0–100.0 scale.\n"
        "     Each keyword hit, geographic spread factor, and temporal clustering signal "
        "     should contribute a traceable delta to the final score.\n"
        "  4. Write explainable_reasoning that maps every score component back to its "
        "     source evidence.\n\n"
        "Produce a single, fully validated EpidemicPrediction object. Assign a unique "
        "event_id (integer) based on the current run."
    ),
    expected_output=(
        "A validated EpidemicPrediction JSON object with fields: event_id (int), "
        "disease (str), region (str), confidence_score (float 0.0–100.0), and "
        "explainable_reasoning (str)."
    ),
    agent=analyst_agent,
    context=[gather_task],
    output_pydantic=EpidemicPrediction,
    guardrail_max_retries=3,
)

visualize_task = Task(
    description=(
        "You will receive a validated EpidemicPrediction Pydantic object from the Analyst.\n\n"
        "Write a self-contained Python script that:\n"
        "  1. Imports matplotlib.pyplot and os.\n"
        "  2. Reads the disease name, region, and confidence_score from the prediction.\n"
        "  3. Creates a publication-quality bar plot:\n"
        "     • X-axis label: '<Disease> (<Region>)'\n"
        "     • Y-axis: 'Outbreak Confidence Score' (range 0–100)\n"
        "     • Bar color: #e63946 with edge color #1d3557\n"
        "     • Title: 'Epidemic Risk Analysis — Run: {{run_id}}'\n"
        "     • Grid lines on the Y-axis for readability.\n"
        "  4. Saves the figure to EXACTLY this path:\n"
        "         ../data_outputs/outbreak_risk_{{run_id}}.png\n"
        "     Use os.makedirs to ensure the directory exists before saving.\n"
        "  5. Prints confirmation of the saved file path.\n\n"
        "CRITICAL: DO NOT output the raw Python code as your final answer. "
        "You MUST pass the code into the `python_repl` tool to execute it. "
        "Your task is only complete when the tool successfully runs and saves "
        "the image to `../data_outputs/outbreak_risk_{{run_id}}.png`. "
        "Your final output should just be a confirmation string that the file exists."
    ),
    expected_output=(
        "A short confirmation string such as: "
        "'Plot saved successfully to ../data_outputs/outbreak_risk_{run_id}.png'. "
        "Do NOT include any Python code in the final output."
    ),
    agent=visualizer_agent,
    context=[analyze_task],
)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(run_id: str):
    """Assemble and kick off the three-agent sequential crew."""
    crew = Crew(
        agents=[gatherer_agent, analyst_agent, visualizer_agent],
        tasks=[gather_task, analyze_task, visualize_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff(inputs={"run_id": run_id})

    print("\n" + "=" * 60)
    print(f"  Pipeline complete — run_id: {run_id}")
    print("=" * 60)
    print(result)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_run_id = sys.argv[1] if len(sys.argv) > 1 else "test_001"
    run_pipeline(test_run_id)
