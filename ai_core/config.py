import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

# Load environment variables (e.g. GEMINI_API_KEY / GOOGLE_API_KEY, ANTHROPIC_API_KEY)
load_dotenv()

# Configure the LLM wrappers
flash_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
)

opus_llm = ChatAnthropic(
    model_name="claude-3-opus-20240229",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)
