import os
import sys
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapegraphScrapeTool
from config import flash_llm, opus_llm
from schemas import EpidemicPrediction
from crewai.tools import tool
import matplotlib.pyplot as plt

# Define custom tool for visualization
@tool("generate_outbreak_visualization")
def generate_outbreak_visualization(disease: str, region: str, confidence_score: float, run_id: str) -> str:
    """
    Generates a bar plot of the confidence score for a disease outbreak and saves it.
    
    Parameters:
    - disease (str): Name of the disease.
    - region (str): Geographic region.
    - confidence_score (float): Outbreak confidence score between 0.0 and 1.0.
    - run_id (str): Run ID for tracking.
    """
    # Ensure data_outputs directory exists in the parent folder
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_outputs"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the bar plot
    plt.figure(figsize=(8, 5))
    plt.bar([f"{disease}\n({region})"], [confidence_score], color="#e63946", edgecolor="#1d3557", width=0.4)
    plt.ylim(0, 1.0)
    plt.title(f"Epidemic Risk Analysis (Run: {run_id})", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Outbreak Confidence Score", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    output_path = os.path.join(output_dir, f"outbreak_risk_{run_id}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return f"Plot successfully generated and saved to: {output_path}"

# OSINT Gatherer Agent
osint_gatherer = Agent(
    role="OSINT Gatherer",
    goal="Monitor health bulletins and news aggregators in Pune and Maharashtra region to extract recent respiratory illness events.",
    backstory="You are a health intelligence OSINT Gatherer. You specialize in scanning official municipal health bulletins and news reports to semantically extract disease names, locations, and dates.",
    tools=[ScrapegraphScrapeTool()],
    llm=flash_llm,
    verbose=True
)

# Epidemiologist Analyst Agent
epidemiologist_analyst = Agent(
    role="Epidemiologist Analyst",
    goal="Analyze respiratory illness data, calculate confidence scores based on keyword severity, and predict outbreak events.",
    backstory="You are a Senior Epidemiologist. You parse raw disease reports, evaluate illness severity and keyword frequencies (e.g. 'severe', 'acute', 'epidemic', 'H1N1', 'influenza'), and produce structured, validated outbreak predictions.",
    llm=opus_llm,
    verbose=True
)

# Visualization Engineer Agent
visualization_engineer = Agent(
    role="Visualization Engineer",
    goal="Generate outbreak risk visualizations from validated epidemiologist predictions.",
    backstory="You are a data visualization engineer. You read validated outbreak predictions (Pydantic objects) and create clear, publication-quality bar plots of outbreak risk scores.",
    tools=[generate_outbreak_visualization],
    llm=flash_llm,
    verbose=True
)

def create_crew(run_id: str) -> Crew:
    # Task 1: Gather OSINT Data
    gather_task = Task(
        description=(
            "Monitor official municipal health bulletins (e.g. Pune Municipal Corporation website) and news aggregators "
            "in the Pune and Maharashtra region for respiratory illnesses reported over the last 7 days. "
            "Semantically extract disease names, locations, and dates from the content."
        ),
        expected_output="A structured summary of disease name, location, and dates extracted from the health reports.",
        agent=osint_gatherer
    )

    # Task 2: Analyze and Output structured prediction
    analyze_task = Task(
        description=(
            "Analyze the gathered health data. Calculate a confidence score between 0.0 and 1.0 based on keyword severity "
            "(e.g., terms like 'severe', 'outbreak', 'epidemic', 'critical', 'respiratory' raise the confidence). "
            "Map the analysis to the EpidemicPrediction model for run_id: {run_id}."
        ),
        expected_output="A validated EpidemicPrediction schema object.",
        agent=epidemiologist_analyst,
        output_pydantic=EpidemicPrediction,
        guardrail_max_retries=3
    )

    # Task 3: Visualize predictions
    visualize_task = Task(
        description=(
            "Read the validated Pydantic object from the previous task. "
            "Extract the disease, region, confidence_score, and use the generate_outbreak_visualization tool to generate a bar plot. "
            "Ensure you pass the run_id '{run_id}' to the tool so the image is saved exactly to '../data_outputs/outbreak_risk_{run_id}.png'."
        ),
        expected_output="Confirmation that the bar plot has been generated and saved.",
        agent=visualization_engineer
    )

    return Crew(
        agents=[osint_gatherer, epidemiologist_analyst, visualization_engineer],
        tasks=[gather_task, analyze_task, visualize_task],
        process=Process.sequential,
        verbose=True
    )

if __name__ == "__main__":
    # Extract run_id from command line arguments or default to 'test_run'
    run_id = sys.argv[1] if len(sys.argv) > 1 else "test_run"
    
    # Initialize and kickoff the crew
    crew = create_crew(run_id)
    result = crew.kickoff(inputs={"run_id": run_id})
    print("\n--- Kickoff Result ---")
    print(result)
