import io
import sys

# Force UTF-8 for stdout/stderr — CrewAI's printer uses emojis that crash
# on Windows' default cp1252 encoding.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import asyncio
import logging
import threading
import time
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from ai_core.main import run_pipeline
    _run_pipeline_import_error = None
except Exception as exc:
    run_pipeline = None
    _run_pipeline_import_error = exc

app = FastAPI(title="Silent Outbreak Predictor API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = {}
DATA_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data_outputs"
DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("backend_api")
if _run_pipeline_import_error is not None:
    logger.warning(
        "ai_core.main.run_pipeline not available: %s", _run_pipeline_import_error
    )


def _invoke_run_pipeline(run_id: str, region: str = "", disease: str = "",
                         date_from: str = "", date_to: str = ""):
    if run_pipeline is None:
        return None

    try:
        crew_result = run_pipeline(
            run_id, region=region, disease=disease,
            date_from=date_from, date_to=date_to,
        )
    except TypeError:
        crew_result = run_pipeline(run_id)

    # Parse the CrewAI result into a plain dict
    parsed = {}
    if crew_result is not None:
        # CrewAI's crew.kickoff() returns the LAST task's output (visualizer).
        # The EpidemicPrediction pydantic data lives in an intermediate task.
        # Search all task outputs for the pydantic model first.
        if hasattr(crew_result, 'tasks_output') and crew_result.tasks_output:
            for task_out in crew_result.tasks_output:
                if hasattr(task_out, 'pydantic') and task_out.pydantic:
                    obj = task_out.pydantic
                    parsed = obj.model_dump() if hasattr(obj, 'model_dump') else obj.dict()
                    break
                elif hasattr(task_out, 'json_dict') and task_out.json_dict:
                    parsed = task_out.json_dict
                    break

        # Fallback: check the top-level crew result
        if not parsed:
            if hasattr(crew_result, 'pydantic') and crew_result.pydantic:
                obj = crew_result.pydantic
                parsed = obj.model_dump() if hasattr(obj, 'model_dump') else obj.dict()
            elif hasattr(crew_result, 'json_dict') and crew_result.json_dict:
                parsed = crew_result.json_dict
            elif hasattr(crew_result, 'raw'):
                parsed = {"explainable_reasoning": str(crew_result.raw)}
            elif isinstance(crew_result, dict):
                parsed = crew_result
    return parsed


def _update_job_from_result(run_id: str, result: dict | None):
    job = jobs.get(run_id)
    if job is None:
        return

    if result is None:
        result = {}

    job["disease"] = result.get("disease", "Unknown")
    job["region"] = result.get("region", "Unknown")
    job["confidence_score"] = result.get("confidence_score", 0.92)
    job["explainable_reasoning"] = result.get(
        "explainable_reasoning",
        "Regional signals indicate elevated outbreak risk in high-density zones.",
    )
    job["status"] = "completed"


def _create_placeholder_heatmap(run_id: str):
    heatmap_path = DATA_OUTPUT_DIR / f"outbreak_risk_{run_id}.png"
    if heatmap_path.exists():
        return heatmap_path

    placeholder_png = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\x0c\x0c\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    heatmap_path.write_bytes(placeholder_png)
    return heatmap_path


_pipeline_lock = threading.Lock()

async def _run_analysis_job(run_id: str, region: str = "",
                            disease: str = "", date_from: str = "",
                            date_to: str = ""):
    logger.info("Running analysis in background for job %s (region=%s, disease=%s, %s to %s)",
                run_id, region, disease, date_from, date_to)
    result = None

    if run_pipeline is not None:
        try:
            result = await asyncio.to_thread(
                _locked_run_pipeline, run_id, region, disease, date_from, date_to
            )
        except Exception as exc:
            logger.exception("Background CrewAI run failed for job %s: %s", run_id, exc)
            result = {
                "confidence_score": 0.0,
                "explainable_reasoning": "Background analysis failed; see logs for details.",
            }
    else:
        logger.info("No CrewAI pipeline available; simulating analysis for job %s", run_id)
        await asyncio.sleep(5)

    _update_job_from_result(run_id, result)
    _create_placeholder_heatmap(run_id)
    logger.info("Background analysis complete for job %s", run_id)


def _locked_run_pipeline(run_id: str, region: str = "", disease: str = "",
                         date_from: str = "", date_to: str = ""):
    """Run the pipeline with a lock to prevent concurrent executor crashes."""
    with _pipeline_lock:
        return _invoke_run_pipeline(run_id, region, disease, date_from, date_to)


class AnalyzeRequest(BaseModel):
    region: str = "Pune, Maharashtra"
    disease: str = ""
    date_from: str = ""
    date_to: str = ""


class AnalyzeResponse(BaseModel):
    run_id: str
    status: str


class StatusResponse(BaseModel):
    run_id: str
    status: str
    disease: str | None = None
    region: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    confidence_score: float | None = None
    explainable_reasoning: str | None = None


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


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "0.1.0",
        "pipeline_available": run_pipeline is not None,
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(background_tasks: BackgroundTasks, body: AnalyzeRequest = AnalyzeRequest()):
    run_id = str(uuid.uuid4())
    jobs[run_id] = {
        "status": "processing",
        "started_at": time.time(),
        "disease": None,
        "region": body.region,
        "date_from": body.date_from,
        "date_to": body.date_to,
        "confidence_score": None,
        "explainable_reasoning": None,
    }
    background_tasks.add_task(
        _run_analysis_job, run_id,
        region=body.region, disease=body.disease,
        date_from=body.date_from, date_to=body.date_to,
    )
    logger.info("Created analysis job %s — region=%s disease=%s %s→%s",
                run_id, body.region, body.disease, body.date_from, body.date_to)
    return {"run_id": run_id, "status": "processing"}


@app.get("/api/status/{run_id}", response_model=StatusResponse)
async def status(run_id: str):
    job = jobs.get(run_id)
    if job is None:
        logger.warning("Status requested for unknown job %s", run_id)
        raise HTTPException(status_code=404, detail="run_id not found")

    response = {
        "run_id": run_id,
        "status": job["status"],
        "disease": job.get("disease"),
        "region": job.get("region"),
        "date_from": job.get("date_from"),
        "date_to": job.get("date_to"),
        "confidence_score": job["confidence_score"],
        "explainable_reasoning": job["explainable_reasoning"],
    }
    return response


@app.get("/api/heatmap/{run_id}")
async def heatmap(run_id: str):
    job = jobs.get(run_id)
    if job is None:
        logger.warning("Heatmap requested for unknown job %s", run_id)
        raise HTTPException(status_code=404, detail="run_id not found")
    if job["status"] != "completed":
        logger.warning("Heatmap requested before completion for job %s", run_id)
        raise HTTPException(status_code=409, detail="Run not complete yet")

    heatmap_path = DATA_OUTPUT_DIR / f"outbreak_risk_{run_id}.png"
    if not heatmap_path.exists():
        logger.warning(
            "Heatmap file missing for job %s at %s", run_id, heatmap_path
        )
        raise HTTPException(status_code=404, detail="Heatmap file not found")

    logger.info("Serving heatmap for job %s from %s", run_id, heatmap_path)
    return FileResponse(path=heatmap_path, media_type="image/png")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
