"""DQN agent inference and integration with FastAPI server."""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

from crop_env import CropEnv
from crop_env.models import Action
from train_dqn import (
    SimpleLinearDQN,
    DQNTrainer,
    observation_to_tensor,
    discrete_to_action,
    NUM_ACTIONS,
)


# ============================================================================
# Pydantic Models for API Responses
# ============================================================================

class AgentConfig(BaseModel):
    """Configuration for agent inference."""
    model_path: str
    task_name: str = "ideal_season"
    num_episodes: int = 5
    deterministic: bool = True


class EpisodeResult(BaseModel):
    """Result of a single episode."""
    episode_num: int
    total_reward: float
    final_score: Optional[float] = None
    steps: int
    task_name: str


class InferenceResponse(BaseModel):
    """Response from agent inference."""
    episodes: List[EpisodeResult]
    avg_reward: float
    max_reward: float
    min_reward: float
    model_path: str


class AgentStepResponse(BaseModel):
    """Response from a single agent step."""
    action: Dict[str, str]
    action_idx: int
    q_values: Optional[List[float]] = None
    next_observation: Dict[str, Any]


# ============================================================================
# DQN Agent Manager
# ============================================================================

class DQNAgentManager:
    """Manager for loading and running DQN agents."""
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.loaded_models: Dict[str, SimpleLinearDQN] = {}
        self.env = CropEnv(seed=42)
    
    def load_model(self, model_path: str) -> SimpleLinearDQN:
        """Load a DQN model from disk."""
        if model_path in self.loaded_models:
            return self.loaded_models[model_path]
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model = SimpleLinearDQN(state_size=34, action_size=NUM_ACTIONS)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        self.loaded_models[model_path] = model
        print(f"Loaded model from {model_path}")
        
        return model
    
    def select_action(
        self,
        model: SimpleLinearDQN,
        state: torch.Tensor,
        deterministic: bool = True,
    ) -> tuple[int, Optional[np.ndarray]]:
        """Select action from state using the model."""
        with torch.no_grad():
            state_tensor = state.unsqueeze(0).to(self.device)
            q_values = model(state_tensor).cpu().numpy()[0]
        
        if deterministic:
            action_idx = np.argmax(q_values)
        else:
            # Softmax exploration
            probs = np.exp(q_values) / np.sum(np.exp(q_values))
            action_idx = np.random.choice(NUM_ACTIONS, p=probs)
        
        return action_idx, q_values
    
    def run_episode(
        self,
        model: SimpleLinearDQN,
        task_name: str = "ideal_season",
        deterministic: bool = True,
        return_trajectory: bool = False,
    ) -> tuple[float, float, int, List]:
        """Run a single episode with the agent."""
        obs = self.env.reset(task_name)
        state = observation_to_tensor(obs)
        
        episode_reward = 0
        steps = 0
        trajectory = []
        
        while True:
            action_idx, q_values = self.select_action(
                model, state, deterministic=deterministic
            )
            action = discrete_to_action(action_idx)
            
            result = self.env.step(action)
            next_obs = result.observation
            reward = result.reward.total
            done = result.done
            
            episode_reward += reward
            steps += 1
            
            if return_trajectory:
                trajectory.append({
                    "step": steps,
                    "action": action.model_dump(),
                    "reward": reward,
                    "observation": {
                        "day": next_obs.day,
                        "crop_health": float(next_obs.metrics.crop_health),
                        "growth_rate": float(next_obs.metrics.growth_rate),
                    }
                })
            
            state = observation_to_tensor(next_obs)
            
            if done:
                final_score = self.env.grade()
                break
        
        return episode_reward, final_score, steps, trajectory
    
    def run_multiple_episodes(
        self,
        model_path: str,
        task_name: str = "ideal_season",
        num_episodes: int = 5,
        deterministic: bool = True,
    ) -> InferenceResponse:
        """Run multiple episodes and collect statistics."""
        model = self.load_model(model_path)
        
        episodes = []
        rewards = []
        
        for ep in range(num_episodes):
            episode_reward, final_score, steps, _ = self.run_episode(
                model, task_name=task_name, deterministic=deterministic
            )
            
            episodes.append(
                EpisodeResult(
                    episode_num=ep + 1,
                    total_reward=episode_reward,
                    final_score=final_score,
                    steps=steps,
                    task_name=task_name,
                )
            )
            rewards.append(episode_reward)
        
        return InferenceResponse(
            episodes=episodes,
            avg_reward=float(np.mean(rewards)),
            max_reward=float(np.max(rewards)),
            min_reward=float(np.min(rewards)),
            model_path=model_path,
        )
    
    def run_step_with_model(
        self,
        model_path: str,
        current_state: dict,
        task_name: str = "ideal_season",
        deterministic: bool = True,
    ) -> AgentStepResponse:
        """Run a single step with the model and return the action."""
        model = self.load_model(model_path)
        
        # Reconstruct observation from state (for now, reset and use fresh)
        obs = self.env.reset(task_name)
        state = observation_to_tensor(obs)
        
        action_idx, q_values = self.select_action(
            model, state, deterministic=deterministic
        )
        action = discrete_to_action(action_idx)
        
        result = self.env.step(action)
        next_obs = result.observation
        
        return AgentStepResponse(
            action=action.model_dump(),
            action_idx=action_idx,
            q_values=q_values.tolist() if q_values is not None else None,
            next_observation=next_obs.model_dump(),
        )


# ============================================================================
# Utility Functions
# ============================================================================

def get_available_models(models_dir: str = "models") -> List[str]:
    """Get list of available trained models."""
    if not os.path.exists(models_dir):
        return []
    
    return [
        f for f in os.listdir(models_dir)
        if f.endswith(".pth") and os.path.isfile(os.path.join(models_dir, f))
    ]


def get_model_info(model_path: str) -> Dict[str, Any]:
    """Get information about a trained model."""
    if not os.path.exists(model_path):
        return {"error": f"Model not found: {model_path}"}
    
    stats_path = model_path.replace(".pth", "_stats.json")
    stats = {}
    
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
    
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    
    return {
        "path": model_path,
        "file_size_mb": round(file_size, 2),
        "stats": stats,
        "exists": True,
    }


def create_agent_demo(
    model_path: str,
    task_name: str = "ideal_season",
    num_episodes: int = 3,
) -> Dict[str, Any]:
    """Create a demo of the agent in action."""
    manager = DQNAgentManager(device="cpu")
    
    try:
        model = manager.load_model(model_path)
        
        episodes_data = []
        for ep in range(num_episodes):
            reward, score, steps, trajectory = manager.run_episode(
                model,
                task_name=task_name,
                deterministic=True,
                return_trajectory=True,
            )
            
            episodes_data.append({
                "episode": ep + 1,
                "reward": reward,
                "score": score,
                "steps": steps,
                "trajectory_sample": trajectory[:5],  # First 5 steps
            })
        
        return {
            "status": "success",
            "model": model_path,
            "task": task_name,
            "episodes": episodes_data,
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


# ============================================================================
# Main Entry Point for Integration
# ============================================================================

if __name__ == "__main__":
    # Example usage
    manager = DQNAgentManager(device="cpu")
    
    # List available models
    models = get_available_models()
    print(f"Available models: {models}")
    
    if models:
        model_path = f"models/{models[0]}"
        print(f"\nTesting model: {model_path}")
        print(f"Model info: {get_model_info(model_path)}")
        
        # Run inference
        result = manager.run_multiple_episodes(
            model_path,
            task_name="ideal_season",
            num_episodes=3,
        )
        
        print(f"\nInference results:")
        print(f"  Avg reward: {result.avg_reward:.2f}")
        print(f"  Max reward: {result.max_reward:.2f}")
        print(f"  Min reward: {result.min_reward:.2f}")
