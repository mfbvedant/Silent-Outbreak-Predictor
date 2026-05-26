"""
config.py — LLM configuration for the Silent Outbreak Predictor pipeline.

Exposes two CrewAI LLM wrappers:
  • fast_llm  — gpt-4o-mini  (low-latency, cost-efficient tasks)
  • smart_llm — gpt-4o       (complex reasoning tasks)

API key is loaded from OPENAI_API_KEY in the environment / .env file.
"""

from pathlib import Path
from dotenv import load_dotenv
from crewai import LLM

# Load environment variables (expects OPENAI_API_KEY)
load_dotenv(Path(__file__).resolve().parent / ".env")

# ---------------------------------------------------------------------------
# LLM wrappers  (CrewAI v1.x native LLM class)
# ---------------------------------------------------------------------------
fast_llm = LLM(
    model="openai/gpt-4o-mini",
    temperature=0.1,
)

smart_llm = LLM(
    model="openai/gpt-4o",
    temperature=0.2,
)
