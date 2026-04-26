"""DQN training script for CropEnv."""

import os
import json
import torch
import numpy as np
from collections import deque
from typing import Optional
from pathlib import Path
import requests

from crop_env import CropEnv
from crop_env.models import Action, IrrigationLevel, FertilizerType, PestManagement

# Discrete action mappings
IRRIGATION_LEVELS = [IrrigationLevel.NONE, IrrigationLevel.LIGHT, IrrigationLevel.MODERATE, IrrigationLevel.HEAVY]
FERTILIZER_TYPES = [FertilizerType.NONE, FertilizerType.NITROGEN, FertilizerType.PHOSPHORUS, FertilizerType.BALANCED, FertilizerType.ORGANIC]
PEST_MANAGEMENT = [PestManagement.NONE, PestManagement.SCOUTING, PestManagement.BIOLOGICAL, PestManagement.CHEMICAL_LIGHT, PestManagement.CHEMICAL_HEAVY]

NUM_ACTIONS = len(IRRIGATION_LEVELS) * len(FERTILIZER_TYPES) * len(PEST_MANAGEMENT)  # 100


def action_to_discrete(action: Action) -> int:
    """Convert Action object to discrete action index."""
    irr_idx = IRRIGATION_LEVELS.index(action.irrigation)
    fert_idx = FERTILIZER_TYPES.index(action.fertilizer)
    pest_idx = PEST_MANAGEMENT.index(action.pest_management)
    return irr_idx * (len(FERTILIZER_TYPES) * len(PEST_MANAGEMENT)) + fert_idx * len(PEST_MANAGEMENT) + pest_idx


def discrete_to_action(action_idx: int) -> Action:
    """Convert discrete action index to Action object."""
    pest_idx = action_idx % len(PEST_MANAGEMENT)
    fert_idx = (action_idx // len(PEST_MANAGEMENT)) % len(FERTILIZER_TYPES)
    irr_idx = action_idx // (len(FERTILIZER_TYPES) * len(PEST_MANAGEMENT))
    
    return Action(
        irrigation=IRRIGATION_LEVELS[irr_idx],
        fertilizer=FERTILIZER_TYPES[fert_idx],
        pest_management=PEST_MANAGEMENT[pest_idx]
    )


def safe_step(env: CropEnv, action: Action, max_retries: int = 3) -> tuple:
    """Safely step the environment, converting invalid actions to valid ones."""
    for attempt in range(max_retries):
        try:
            result = env.step(action)
            return result, True
        except RuntimeError as e:
            # Action violated constraints, try fallback actions
            error_msg = str(e).lower()
            
            if "fertilizer" in error_msg:
                # Fallback to "none" fertilizer
                action.fertilizer = FertilizerType.NONE
            elif "pesticide" in error_msg or "chemical" in error_msg:
                # Fallback to "none" pest management
                action.pest_management = PestManagement.NONE
            elif "irrigation" in error_msg:
                # Fallback to "none" irrigation
                action.irrigation = IrrigationLevel.NONE
            else:
                # Unknown constraint, use all "none"
                action.irrigation = IrrigationLevel.NONE
                action.fertilizer = FertilizerType.NONE
                action.pest_management = PestManagement.NONE
            
            if attempt == max_retries - 1:
                # Last retry failed, return with penalty
                result = env.step(action)  # This might still fail, let it propagate
                return result, False
    
    return None, False


def observation_to_tensor(obs) -> torch.Tensor:
    """Convert Observation object to a tensor representation."""
    features = []
    
    # Basic state (3 features)
    features.append(obs.day)
    features.append(obs.total_days)
    features.append(GrowthStageEncoder.encode(obs.growth_stage))
    
    # Metrics (8 features)
    features.extend([
        obs.metrics.crop_health,
        obs.metrics.growth_rate,
        obs.metrics.soil_health,
        obs.metrics.water_stress,
        obs.metrics.nutrient_stress,
        obs.metrics.pest_pressure,
        obs.metrics.crop_quality,
        obs.metrics.environmental_score,
    ])
    
    # Deltas (8 features)
    features.extend([
        obs.deltas.crop_health,
        obs.deltas.growth_rate,
        obs.deltas.soil_health,
        obs.deltas.water_stress,
        obs.deltas.nutrient_stress,
        obs.deltas.pest_pressure,
        obs.deltas.crop_quality,
        obs.deltas.environmental_score,
    ])
    
    # Weather (3 features)
    features.extend([
        obs.weather.temperature,
        obs.weather.rainfall_mm,
        float(obs.weather.is_extreme_event),
    ])
    
    # Soil & water (2 features)
    features.extend([
        obs.soil_moisture,
        obs.water_used_total,
    ])
    
    # Budget (2 features)
    remaining = obs.budget.remaining_usd if obs.budget.remaining_usd is not None else 10000.0
    features.extend([
        obs.budget.spent_usd,
        remaining,
    ])
    
    # Trends (8 features, use zeros if not available yet)
    if obs.trends:
        features.extend([
            obs.trends.crop_health_trend,
            obs.trends.growth_rate_trend,
            obs.trends.soil_health_trend,
            obs.trends.water_stress_trend,
            obs.trends.nutrient_stress_trend,
            obs.trends.pest_pressure_trend,
            obs.trends.reward_trend,
            obs.trends.reward_consistency,
        ])
    else:
        features.extend([0.0] * 8)
    
    return torch.tensor(features, dtype=torch.float32)


class GrowthStageEncoder:
    """Map growth stages to numeric values."""
    STAGES = {
        "germination": 0,
        "vegetative": 1,
        "flowering": 2,
        "fruiting": 3,
        "maturity": 4,
    }
    
    @staticmethod
    def encode(stage) -> float:
        stage_str = stage.value if hasattr(stage, 'value') else str(stage).lower()
        return float(GrowthStageEncoder.STAGES.get(stage_str, 0))


class SimpleLinearDQN(torch.nn.Module):
    """Simple linear DQN for CropEnv."""
    
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_size),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DQNTrainer:
    """DQN trainer for CropEnv."""
    
    def __init__(
        self,
        state_size: int = 34,  # Fixed: actual tensor size (3+8+8+3+2+2+8 = 34)
        action_size: int = NUM_ACTIONS,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # Networks
        self.q_network = SimpleLinearDQN(state_size, action_size).to(self.device)
        self.target_network = SimpleLinearDQN(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Training stats
        self.episode_rewards = []
        self.episode_losses = []
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            state_tensor = state.unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def remember(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool):
        """Store experience in replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size: Optional[int] = None):
        """Train on a batch of experiences."""
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample random batch
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Compute Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(q_values, target_q_values)
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """Save trained model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_network.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load trained model."""
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())
        print(f"Model loaded from {path}")


def train_dqn(
    env: CropEnv,
    num_episodes: int = 50,
    task_name: str = "ideal_season",
    save_path: str = "models/dqn_model.pth",
    device: str = "cpu",
):
    """Train DQN on CropEnv."""
    
    trainer = DQNTrainer(device=device)
    
    for episode in range(num_episodes):
        obs = env.reset(task_name)
        state = observation_to_tensor(obs)
        
        episode_reward = 0
        episode_loss = 0
        steps = 0
        
        while True:
            # Select and execute action
            action_idx = trainer.select_action(state, training=True)
            action = discrete_to_action(action_idx)
            
            result, action_valid = safe_step(env, action)
            next_obs = result.observation
            reward = result.reward.total  # Extract scalar reward from RewardBreakdown
            # Penalize invalid actions slightly
            if not action_valid:
                reward *= 0.9
            done = result.done
            
            next_state = observation_to_tensor(next_obs)
            
            # Store experience
            trainer.remember(state, action_idx, reward, next_state, done)
            
            # Train on batch
            loss = trainer.replay()
            if loss is not None:
                episode_loss += loss
            
            episode_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        # Update target network every 10 episodes
        if (episode + 1) % 10 == 0:
            trainer.update_target_network()
        
        trainer.decay_epsilon()
        trainer.episode_rewards.append(episode_reward)
        if steps > 0:
            trainer.episode_losses.append(episode_loss / steps)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(trainer.episode_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {trainer.epsilon:.3f}")
    
    # Save model
    trainer.save(save_path)
    
    # Save training stats
    stats_path = save_path.replace(".pth", "_stats.json")
    stats = {
        "episode_rewards": trainer.episode_rewards,
        "final_epsilon": trainer.epsilon,
    }
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Training stats saved to {stats_path}")
    
    return trainer


def evaluate_dqn(
    env: CropEnv,
    trainer: DQNTrainer,
    num_episodes: int = 5,
    task_name: str = "ideal_season",
):
    """Evaluate trained DQN."""
    eval_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset(task_name)
        state = observation_to_tensor(obs)
        
        episode_reward = 0
        
        while True:
            action_idx = trainer.select_action(state, training=False)
            action = discrete_to_action(action_idx)
            
            result, action_valid = safe_step(env, action)
            next_obs = result.observation
            reward = result.reward.total  # Extract scalar reward from RewardBreakdown
            done = result.done
            
            episode_reward += reward
            state = observation_to_tensor(next_obs)
            
            if done:
                break
        
        eval_rewards.append(episode_reward)
        print(f"Eval Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    print(f"Average Eval Reward: {np.mean(eval_rewards):.2f}")
    return eval_rewards


if __name__ == "__main__":
    # Initialize environment
    env = CropEnv(seed=42)
    
    # Train (use CPU due to GPU compatibility issues with GTX 1070)
    print("Starting DQN training...")
    trainer = train_dqn(
        env,
        num_episodes=50,
        task_name="ideal_season",
        save_path="models/dqn_model.pth",
        device="cpu",
    )
    
    # Evaluate
    print("\nEvaluating trained model...")
    evaluate_dqn(env, trainer, num_episodes=5, task_name="ideal_season")
    
    # Run agent
    API_URL = "http://localhost:7860"

    response = requests.post(
        f"{API_URL}/agent/run",
        json={
            "model_path": "models/dqn_model.pth",
            "task_name": "ideal_season",
            "num_episodes": 5,
        },
    )
    print(response.json())
