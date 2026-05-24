"""
api.py — FastAPI backend for the Silent Outbreak Predictor.

Endpoints:
    GET  /api/health              → connectivity check
    POST /api/analyze             → launch CrewAI pipeline (background)
    GET  /api/status/{run_id}     → poll job status + results
    GET  /api/heatmap/{run_id}    → serve the generated outbreak plot
"""

import asyncio
import logging
import os
import sys
import time
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment bootstrapping
# ---------------------------------------------------------------------------
# Ensure emoji-heavy CrewAI output doesn't crash on Windows cp1252 consoles.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Make ai_core importable — it lives one level up from backend/.
_project_root = Path(__file__).resolve().parent.parent
_ai_core_dir = _project_root / "ai_core"
if str(_ai_core_dir) not in sys.path:
    sys.path.insert(0, str(_ai_core_dir))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Load the .env from ai_core (contains OPENAI_API_KEY).
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_ai_core_dir / ".env")

# ---------------------------------------------------------------------------
# FastAPI + CORS
# ---------------------------------------------------------------------------
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

app = FastAPI(title="Silent Outbreak Predictor API", version="0.2.0")

# Allow Streamlit (localhost:8501) and any local dev origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pipeline import (graceful fallback if deps missing)
# ---------------------------------------------------------------------------
try:
    from ai_core.main import run_pipeline

    _run_pipeline_import_error = None
except Exception as exc:
    run_pipeline = None
    _run_pipeline_import_error = exc

# ---------------------------------------------------------------------------
# State & paths
# ---------------------------------------------------------------------------
jobs: dict[str, dict] = {}
DATA_OUTPUT_DIR = _project_root / "data_outputs"
DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRONTEND_DIR = _project_root / "frontend"

# Serve static assets (images, CSS, JS) from frontend/
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("backend_api")
if _run_pipeline_import_error is not None:
    logger.warning(
        "ai_core.main.run_pipeline not available: %s", _run_pipeline_import_error
    )


# ---------------------------------------------------------------------------
# Background analysis job
# ---------------------------------------------------------------------------
def _invoke_run_pipeline(run_id: str):
    """Call the CrewAI pipeline, handling signature variants."""
    if run_pipeline is None:
        return None
    try:
        return run_pipeline(run_id)
    except TypeError:
        return run_pipeline()


def _extract_result_data(run_id: str, crew_result) -> dict:
    """
    Extract confidence_score and explainable_reasoning from CrewAI output.

    CrewAI's crew.kickoff() returns a CrewOutput object. The analyze_task
    has output_pydantic=EpidemicPrediction, so we look for the pydantic
    output in the task results.
    """
    data = {
        "confidence_score": None,
        "explainable_reasoning": None,
        "disease": None,
        "region": None,
        "event_id": None,
    }

    if crew_result is None:
        return data

    # Attempt 1: CrewOutput.pydantic — available if the *last* task has output_pydantic
    pydantic_obj = getattr(crew_result, "pydantic", None)
    if pydantic_obj is not None:
        try:
            obj_dict = pydantic_obj.model_dump()
            data.update({k: v for k, v in obj_dict.items() if k in data})
            return data
        except Exception:
            pass

    # Attempt 2: Walk task_outputs looking for the EpidemicPrediction
    task_outputs = getattr(crew_result, "tasks_output", [])
    for task_output in task_outputs:
        pydantic_obj = getattr(task_output, "pydantic", None)
        if pydantic_obj is not None:
            try:
                obj_dict = pydantic_obj.model_dump()
                data.update({k: v for k, v in obj_dict.items() if k in data})
                return data
            except Exception:
                continue

    # Attempt 3: Try parsing the raw string output as JSON
    raw = getattr(crew_result, "raw", "") or str(crew_result)
    try:
        import json

        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            data.update({k: v for k, v in parsed.items() if k in data})
    except (json.JSONDecodeError, TypeError):
        pass

    return data


async def _run_analysis_job(run_id: str):
    """Execute the CrewAI pipeline in a background thread."""
    logger.info("Running analysis in background for job %s", run_id)
    result_data: dict = {}

    if run_pipeline is not None:
        try:
            crew_result = await asyncio.to_thread(_invoke_run_pipeline, run_id)
            result_data = _extract_result_data(run_id, crew_result)
        except Exception as exc:
            logger.exception(
                "Background CrewAI run failed for job %s: %s", run_id, exc
            )
            result_data = {
                "confidence_score": 0.0,
                "explainable_reasoning": f"Pipeline failed: {exc}",
            }
    else:
        # Simulation mode when ai_core is not installed
        logger.info(
            "No CrewAI pipeline available; simulating analysis for job %s", run_id
        )
        await asyncio.sleep(3)
        result_data = {
            "confidence_score": 72.5,
            "explainable_reasoning": "[SIMULATED] No real pipeline available.",
            "disease": "Simulated Pathogen",
            "region": "Pune, Maharashtra",
            "event_id": 1,
        }

    # Update the job record
    job = jobs.get(run_id)
    if job is not None:
        job.update(
            {
                "status": "completed",
                "confidence_score": result_data.get("confidence_score"),
                "explainable_reasoning": result_data.get("explainable_reasoning"),
                "disease": result_data.get("disease"),
                "region": result_data.get("region"),
                "event_id": result_data.get("event_id"),
            }
        )

    logger.info("Background analysis complete for job %s", run_id)


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------
class AnalyzeResponse(BaseModel):
    run_id: str
    status: str


class StatusResponse(BaseModel):
    run_id: str
    status: str
    confidence_score: float | None = None
    explainable_reasoning: str | None = None
    disease: str | None = None
    region: str | None = None
    event_id: int | None = None


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("Incoming request %s %s", request.method, request.url.path)
    response = await call_next(request)
    logger.info("Response status %s for %s", response.status_code, request.url.path)
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning("HTTP exception for %s: %s", request.url.path, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception for %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    """Simple health check for frontend connectivity verification."""
    return {
        "status": "ok",
        "pipeline_available": run_pipeline is not None,
        "version": "0.2.0",
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(background_tasks: BackgroundTasks):
    """Launch a new CrewAI analysis job in the background."""
    run_id = str(uuid.uuid4())
    jobs[run_id] = {
        "status": "processing",
        "started_at": time.time(),
        "confidence_score": None,
        "explainable_reasoning": None,
        "disease": None,
        "region": None,
        "event_id": None,
    }
    background_tasks.add_task(_run_analysis_job, run_id)
    logger.info("Created new analysis job %s and queued background work", run_id)
    return {"run_id": run_id, "status": "processing"}


@app.get("/api/status/{run_id}", response_model=StatusResponse)
async def status(run_id: str):
    """Poll the status and results of a running analysis job."""
    job = jobs.get(run_id)
    if job is None:
        logger.warning("Status requested for unknown job %s", run_id)
        raise HTTPException(status_code=404, detail="run_id not found")

    return {
        "run_id": run_id,
        "status": job["status"],
        "confidence_score": job["confidence_score"],
        "explainable_reasoning": job["explainable_reasoning"],
        "disease": job.get("disease"),
        "region": job.get("region"),
        "event_id": job.get("event_id"),
    }


@app.get("/api/heatmap/{run_id}")
async def heatmap(run_id: str):
    """Serve the generated outbreak risk plot for a completed run."""
    job = jobs.get(run_id)
    if job is None:
        logger.warning("Heatmap requested for unknown job %s", run_id)
        raise HTTPException(status_code=404, detail="run_id not found")
    if job["status"] != "completed":
        logger.warning("Heatmap requested before completion for job %s", run_id)
        raise HTTPException(status_code=409, detail="Run not complete yet")

    # The CrewAI visualizer saves to: data_outputs/outbreak_risk_{run_id}.png
    heatmap_path = DATA_OUTPUT_DIR / f"outbreak_risk_{run_id}.png"
    if not heatmap_path.exists():
        logger.warning(
            "Heatmap file missing for job %s at %s", run_id, heatmap_path
        )
        raise HTTPException(status_code=404, detail="Heatmap file not found")

    logger.info("Serving heatmap for job %s from %s", run_id, heatmap_path)
    return FileResponse(path=heatmap_path, media_type="image/png")


# ---------------------------------------------------------------------------
# Frontend — serve index.html at root
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main SPA frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))
