"""FastAPI server exposing CropEnv over HTTP for HF Space deployment."""

from __future__ import annotations

import os
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from crop_env import CropEnv
from crop_env.models import Action, Observation, StepResult

app = FastAPI(title="Crop-Outcome OpenEnv", version="1.0.0")

# One shared env instance per container — the validator expects session state
_env = CropEnv(seed=int(os.environ.get("SEED", "42")))


@app.post("/reset")
def reset(body: Optional[dict[str,Any]]=None) -> JSONResponse:
    """Start a new episode. Body:{"task_name":"ideal_season"}"""
    task_name= (body or {}).get("task_name","ideal_season")
    try:
        obs = _env.reset(task_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(obs.model_dump())


@app.post("/step")
def step(body: Optional[dict[str, Any]]= None) -> JSONResponse:
    """Apply one action. Body: Action fields as JSON."""
    try:
        action = Action(**(body or {}))
        result = _env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(result.model_dump())


@app.get("/state")
def state() -> JSONResponse:
    """Return current environment state."""
    try:
        s = _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(s.model_dump())


@app.get("/grade")
def grade() -> JSONResponse:
    """Return grader score for current episode."""
    try:
        score = _env.grade()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse({"score": score})


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


def main() -> None:
    """Main entry point for the server."""
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
