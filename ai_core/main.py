"""
main.py — Silent Outbreak Predictor pipeline orchestrator.

Composes the CrewAI sequential pipeline:
  gather_task  →  analyze_task  →  visualize_task

Usage:
    python main.py              # runs with default run_id "test_001"
    python main.py <run_id>     # runs with a custom run_id
"""

import sys
from datetime import datetime, timedelta

from crewai import Crew, Task, Process

from .agents import gatherer_agent, analyst_agent, visualizer_agent
from .schemas import EpidemicPrediction


# ---------------------------------------------------------------------------
# Task definitions (created fresh per run to avoid concurrency issues)
# ---------------------------------------------------------------------------

def _build_tasks():
    """Create fresh Task instances for each pipeline run.

    Task descriptions use CrewAI template variables:
      {{run_id}}, {{region}}, {{disease}}, {{date_from}}, {{date_to}}
    """
    gather_task = Task(
        description=(
            "Search for disease outbreak and epidemic events in the region '{{region}}' "
            "during the period from {{date_from}} to {{date_to}}.\n\n"
            "{% if disease and disease != 'Any / Auto-detect' %}"
            "TARGETED SEARCH: Focus specifically on '{{disease}}' outbreaks, cases, and "
            "epidemiological reports.\n"
            "{% else %}"
            "BROAD SCAN: Search for ANY respiratory, infectious, or epidemic-potential "
            "illness (e.g. cholera, H1N1, dengue, COVID-19, Ebola, plague, typhoid, etc.).\n"
            "{% endif %}\n"
            "Search strategy:\n"
            "  1. Search Google for: '{{disease}} outbreak {{region}} {{date_from}} {{date_to}}'\n"
            "  2. Scrape official health bulletins: WHO, ProMED, CDC MMWR, local health departments\n"
            "  3. Check news aggregators for outbreak reports in the target region and period\n"
            "  4. For historical dates, look for archived reports, published case studies, "
            "and retrospective epidemiological analyses\n\n"
            "For every event you find, extract:\n"
            "  • Disease / pathogen name\n"
            "  • Affected geographic location (city, district, state, or country)\n"
            "  • Date or date range of the reported incident\n"
            "  • Case count or severity indicators if available\n"
            "  • A brief summary of the report and its source URL\n\n"
            "Ignore advertisements, opinion editorials, and unrelated news. Focus strictly on "
            "verified health data sources."
        ),
        expected_output=(
            "A structured raw-text summary listing each extracted disease event with its "
            "disease name, location, date(s), case counts (if available), and a one-line "
            "synopsis of the source report including the URL."
        ),
        agent=gatherer_agent,
    )

    analyze_task = Task(
        description=(
            "You will receive a raw-text summary of disease events from the OSINT Gatherer "
            "for the region '{{region}}' during {{date_from}} to {{date_to}}.\n\n"
            "{% if disease and disease != 'Any / Auto-detect' %}"
            "The user is specifically investigating '{{disease}}'. Prioritize this pathogen "
            "in your analysis but also note any other disease signals found.\n"
            "{% endif %}\n"
            "For each event, perform the following analysis:\n"
            "  1. Cross-reference the disease name against known epidemic indicators.\n"
            "  2. Assess keyword severity — flag terms such as 'severe', 'outbreak', 'acute', "
            "'epidemic', 'mortality', 'H1N1', 'influenza', 'respiratory distress', 'deaths', "
            "'cases reported', 'cluster'.\n"
            "  3. Calculate a mathematically grounded confidence_score on a 0.0–100.0 scale.\n"
            "     Each keyword hit, geographic spread factor, case count magnitude, and "
            "     temporal clustering signal should contribute a traceable delta to the score.\n"
            "  4. For HISTORICAL analysis: confirmed past outbreaks with documented case counts "
            "     should receive high confidence scores (70+). The score reflects how confident "
            "     we are that an epidemic event occurred, not whether it is currently ongoing.\n"
            "  5. Write explainable_reasoning that maps every score component back to its "
            "     source evidence. Include the time period analyzed.\n\n"
            "Set the 'region' field to '{{region}}' in the output.\n"
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
            "     • Bar color based on score: green (<30), #f59e0b (30-55), #f97316 (55-80), #e63946 (80+)\n"
            "     • Edge color: #1d3557\n"
            "     • Title: 'Epidemic Risk Analysis — {{region}}\\n{{date_from}} to {{date_to}}'\n"
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

    return gather_task, analyze_task, visualize_task


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    run_id: str,
    region: str = "Pune, Maharashtra",
    disease: str = "",
    date_from: str = "",
    date_to: str = "",
):
    """Assemble and kick off the three-agent sequential crew.

    Args:
        run_id: Unique identifier for this analysis run.
        region: Target geographic region to search.
        disease: Specific disease to investigate (empty = auto-detect).
        date_from: Start date for the search window (YYYY-MM-DD).
        date_to: End date for the search window (YYYY-MM-DD).
    """
    # Default dates: last 7 days if not provided
    if not date_to:
        date_to = datetime.now().strftime("%Y-%m-%d")
    if not date_from:
        date_from = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    if not disease:
        disease = "Any / Auto-detect"

    gather_task, analyze_task, visualize_task = _build_tasks()

    crew = Crew(
        agents=[gatherer_agent, analyst_agent, visualizer_agent],
        tasks=[gather_task, analyze_task, visualize_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff(inputs={
        "run_id": run_id,
        "region": region,
        "disease": disease,
        "date_from": date_from,
        "date_to": date_to,
    })

    print("\n" + "=" * 60)
    print(f"  Pipeline complete — run_id: {run_id}")
    print(f"  Region: {region} | Disease: {disease}")
    print(f"  Period: {date_from} to {date_to}")
    print("=" * 60)
    print(result)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_run_id = sys.argv[1] if len(sys.argv) > 1 else "test_001"
    run_pipeline(test_run_id)
