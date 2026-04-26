"""FastAPI server with integrated DQN agent inference."""

from __future__ import annotations

import os
from typing import Any, Optional, List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from crop_env import CropEnv
from crop_env.models import Action
from agent_inference import (
    DQNAgentManager,
    AgentConfig,
    InferenceResponse,
    AgentStepResponse,
    get_available_models,
    get_model_info,
)


# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="Crop-Outcome OpenEnv with DQN Agent",
    version="2.0.0",
    description="Crop management environment with integrated trained DQN agent"
)

# One shared env instance per container
_env = CropEnv(seed=int(os.environ.get("SEED", "42")))

# Agent manager
_agent_manager = DQNAgentManager(device="cpu")


# ============================================================================
# Request/Response Models
# ============================================================================

class RunInferenceRequest(BaseModel):
    model_path: str = "models/dqn_model.pth"
    task_name: str = "ideal_season"
    num_episodes: int = 5
    deterministic: bool = True


class ModelListResponse(BaseModel):
    available_models: List[str]
    models_dir: str


# ============================================================================
# Environment Endpoints (Original)
# ============================================================================

@app.get("/")
def root():
    return {
        "message": "Crop-Outcome OpenEnv API with DQN Agent is running 🚀",
        "endpoints": {
            "environment": [
                "/reset (POST)",
                "/step (POST)",
                "/state (GET)",
                "/grade (GET)",
            ],
            "agent": [
                "/agent/models (GET)",
                "/agent/model-info (GET)",
                "/agent/run (POST)",
                "/agent/step (POST)",
                "/agent/demo (POST)",
            ],
            "health": ["/health (GET)"]
        }
    }


@app.post("/reset")
def reset(body: Optional[dict[str, Any]] = None) -> JSONResponse:
    """Start a new episode. Body: {"task_name":"ideal_season"}"""
    task_name = (body or {}).get("task_name", "ideal_season")
    try:
        obs = _env.reset(task_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(obs.model_dump())


@app.post("/step")
def step(body: Optional[dict[str, Any]] = None) -> JSONResponse:
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


@app.get("/favicon.ico")
def favicon():
    """Return a simple favicon to suppress 404 errors."""
    return JSONResponse({"status": "ok"})


# ============================================================================
# Agent Endpoints
# ============================================================================

@app.get("/agent/models")
def list_agent_models() -> ModelListResponse:
    """Get list of available trained models."""
    try:
        models = get_available_models(models_dir="models")
        return ModelListResponse(
            available_models=models,
            models_dir="models",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/agent/model-info")
def agent_model_info(model_path: str = "models/dqn_model.pth") -> JSONResponse:
    """Get information about a specific model."""
    try:
        info = get_model_info(model_path)
        return JSONResponse(info)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/agent/run")
def run_agent_inference(request: RunInferenceRequest) -> InferenceResponse:
    """Run agent inference on environment."""
    try:
        result = _agent_manager.run_multiple_episodes(
            model_path=request.model_path,
            task_name=request.task_name,
            num_episodes=request.num_episodes,
            deterministic=request.deterministic,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/agent/step")
def agent_step(body: Optional[dict[str, Any]] = None) -> AgentStepResponse:
    """Take a single step with the agent."""
    config = body or {}
    model_path = config.get("model_path", "models/dqn_model.pth")
    task_name = config.get("task_name", "ideal_season")
    deterministic = config.get("deterministic", True)
    
    try:
        response = _agent_manager.run_step_with_model(
            model_path=model_path,
            current_state={},
            task_name=task_name,
            deterministic=deterministic,
        )
        return response
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/agent/demo")
def run_agent_demo(body: Optional[dict[str, Any]] = None) -> JSONResponse:
    """Run a demo episode with the agent."""
    config = body or {}
    model_path = config.get("model_path", "models/dqn_model.pth")
    task_name = config.get("task_name", "ideal_season")
    num_episodes = config.get("num_episodes", 3)
    
    try:
        result = _agent_manager.run_multiple_episodes(
            model_path=model_path,
            task_name=task_name,
            num_episodes=num_episodes,
            deterministic=True,
        )
        
        # Format for UI
        demo_data = {
            "model": model_path,
            "task": task_name,
            "episodes": [
                {
                    "episode": ep.episode_num,
                    "reward": ep.total_reward,
                    "score": ep.final_score,
                    "steps": ep.steps,
                }
                for ep in result.episodes
            ],
            "summary": {
                "avg_reward": result.avg_reward,
                "max_reward": result.max_reward,
                "min_reward": result.min_reward,
            }
        }
        
        return JSONResponse(demo_data)
    
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================================
# Utility Endpoints
# ============================================================================

@app.get("/agent/tasks")
def available_tasks() -> JSONResponse:
    """Get list of available tasks."""
    return JSONResponse({
        "tasks": [
            "ideal_season",
            "variable_weather",
            "drought_year",
            "supply_chain_disruption",
            "regulatory_shift",
        ]
    })


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    """Main entry point for the server."""
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
