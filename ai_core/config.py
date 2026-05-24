"""
config.py — LLM configuration for the Silent Outbreak Predictor pipeline.

Exposes two ChatOpenAI wrappers:
  • fast_llm  — gpt-4o-mini  (low-latency, cost-efficient tasks)
  • smart_llm — gpt-4o       (complex reasoning tasks)

API key is loaded from OPENAI_API_KEY in the environment / .env file.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables (expects OPENAI_API_KEY)
load_dotenv()

# ---------------------------------------------------------------------------
# LLM wrappers
# ---------------------------------------------------------------------------
fast_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
)

smart_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
)
