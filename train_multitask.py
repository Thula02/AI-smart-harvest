"""Multi-task and task-specific DQN trainers."""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from crop_env import CropEnv
from crop_env.models import Action, FertilizerType
from train_dqn import (
    DQNTrainer,
    observation_to_tensor,
    discrete_to_action,
    NUM_ACTIONS,
)


# Task configurations
TASKS = [
    "ideal_season",
    "variable_weather",
    "drought_year",
    "supply_chain_disruption",
    "regulatory_shift",
]

TASK_DIFFICULTY = {
    "ideal_season": 1,  # Easiest
    "supply_chain_disruption": 2,
    "variable_weather": 2,
    "regulatory_shift": 3,
    "drought_year": 3,  # Hardest
}


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
                from crop_env.models import PestManagement
                action.pest_management = PestManagement.NONE
            elif "irrigation" in error_msg:
                # Fallback to "none" irrigation
                from crop_env.models import IrrigationLevel
                action.irrigation = IrrigationLevel.NONE
            else:
                # Unknown constraint, use all "none"
                from crop_env.models import IrrigationLevel, PestManagement
                action.irrigation = IrrigationLevel.NONE
                action.fertilizer = FertilizerType.NONE
                action.pest_management = PestManagement.NONE
            
            if attempt == max_retries - 1:
                # Last retry failed, return with penalty
                print(f"Warning: Action constraint violated after {max_retries} retries: {e}")
                result = env.step(action)  # This might still fail, let it propagate
                return result, False
    
    return None, False


def train_multi_task_dqn(
    num_episodes_per_task: int = 20,
    total_episodes: int = 100,
    save_path: str = "models/dqn_multitask.pth",
    device: str = "cpu",
) -> Tuple[DQNTrainer, Dict]:
    """Train DQN on all tasks together using task rotation."""
    
    env = CropEnv(seed=42)
    trainer = DQNTrainer(device=device)
    
    all_episode_rewards = []
    task_rewards = {task: [] for task in TASKS}
    
    print(f"\n🎯 Starting Multi-Task Training ({total_episodes} episodes total)")
    print(f"Rotating through tasks: {TASKS}")
    print("=" * 60)
    
    episode_count = 0
    while episode_count < total_episodes:
        for task in TASKS:
            if episode_count >= total_episodes:
                break
            
            obs = env.reset(task)
            state = observation_to_tensor(obs)
            
            episode_reward = 0
            steps = 0
            
            while True:
                action_idx = trainer.select_action(state, training=True)
                action = discrete_to_action(action_idx)
                
                result, action_valid = safe_step(env, action)
                next_obs = result.observation
                reward = result.reward.total
                # Penalize invalid actions slightly
                if not action_valid:
                    reward *= 0.9
                done = result.done
                
                next_state = observation_to_tensor(next_obs)
                
                trainer.remember(state, action_idx, reward, next_state, done)
                loss = trainer.replay()
                
                episode_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            
            # Update target network every 5 episodes
            if (episode_count + 1) % 5 == 0:
                trainer.update_target_network()
            
            trainer.decay_epsilon()
            all_episode_rewards.append(episode_reward)
            task_rewards[task].append(episode_reward)
            episode_count += 1
            
            if episode_count % 10 == 0:
                avg_reward = np.mean(all_episode_rewards[-10:])
                print(f"Episode {episode_count}/{total_episodes} | Task: {task:30s} | "
                      f"Reward: {episode_reward:6.2f} | Avg(10): {avg_reward:6.2f} | "
                      f"Epsilon: {trainer.epsilon:.3f}")
    
    # Save model
    trainer.save(save_path)
    
    # Save training stats
    stats = {
        "total_episodes": total_episodes,
        "tasks": TASKS,
        "final_epsilon": trainer.epsilon,
        "task_rewards": {task: [float(r) for r in rewards] 
                        for task, rewards in task_rewards.items()},
        "task_averages": {task: float(np.mean(task_rewards[task])) 
                         for task in TASKS},
    }
    
    stats_path = save_path.replace(".pth", "_stats.json")
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 60)
    print("📊 Multi-Task Training Complete!")
    print(f"Model saved to: {save_path}")
    print(f"Stats saved to: {stats_path}")
    print("\nTask Performance Summary:")
    for task in TASKS:
        avg = np.mean(task_rewards[task])
        print(f"  {task:30s}: Avg Reward = {avg:6.2f}")
    
    return trainer, stats


def train_task_specific_dqn(
    task_name: str,
    num_episodes: int = 50,
    device: str = "cpu",
) -> Tuple[DQNTrainer, Dict]:
    """Train DQN specialized for a specific task."""
    
    env = CropEnv(seed=42)
    trainer = DQNTrainer(device=device)
    
    episode_rewards = []
    
    print(f"\n🎯 Training Task-Specific DQN for: {task_name}")
    print("=" * 60)
    
    for episode in range(num_episodes):
        obs = env.reset(task_name)
        state = observation_to_tensor(obs)
        
        episode_reward = 0
        steps = 0
        
        while True:
            action_idx = trainer.select_action(state, training=True)
            action = discrete_to_action(action_idx)
            
            result, action_valid = safe_step(env, action)
            next_obs = result.observation
            reward = result.reward.total
            # Penalize invalid actions slightly
            if not action_valid:
                reward *= 0.9
            done = result.done
            
            next_state = observation_to_tensor(next_obs)
            
            trainer.remember(state, action_idx, reward, next_state, done)
            loss = trainer.replay()
            
            episode_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        # Update target network every 10 episodes
        if (episode + 1) % 10 == 0:
            trainer.update_target_network()
        
        trainer.decay_epsilon()
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes} | Reward: {episode_reward:6.2f} | "
                  f"Avg(10): {avg_reward:6.2f} | Epsilon: {trainer.epsilon:.3f}")
    
    # Save model
    save_path = f"models/dqn_{task_name}.pth"
    trainer.save(save_path)
    
    # Save stats
    stats = {
        "task": task_name,
        "difficulty": TASK_DIFFICULTY.get(task_name, 2),
        "episodes": num_episodes,
        "final_epsilon": trainer.epsilon,
        "episode_rewards": [float(r) for r in episode_rewards],
        "avg_reward": float(np.mean(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
    }
    
    stats_path = save_path.replace(".pth", "_stats.json")
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"✅ Task-Specific Training Complete for {task_name}!")
    print(f"Model saved to: {save_path}")
    print(f"  Avg Reward: {stats['avg_reward']:.2f}")
    print(f"  Max Reward: {stats['max_reward']:.2f}")
    print(f"  Min Reward: {stats['min_reward']:.2f}")
    
    return trainer, stats


def train_all_task_specific_models(
    num_episodes_per_task: int = 50,
    device: str = "cpu",
):
    """Train separate models for each task."""
    
    print("\n" + "=" * 60)
    print("🚀 Training Task-Specific Models for All Tasks")
    print("=" * 60)
    
    all_stats = {}
    
    for task in TASKS:
        print(f"\n{'█' * 60}")
        trainer, stats = train_task_specific_dqn(
            task_name=task,
            num_episodes=num_episodes_per_task,
            device=device,
        )
        all_stats[task] = stats
    
    # Save comprehensive summary
    summary_path = "models/training_summary.json"
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    
    print("\n" + "=" * 60)
    print("🎉 All Task-Specific Models Trained!")
    print(f"Summary saved to: {summary_path}")
    print("\nPerformance Overview:")
    for task in TASKS:
        avg = all_stats[task]["avg_reward"]
        difficulty = all_stats[task]["difficulty"]
        print(f"  [{difficulty}★] {task:30s}: Avg Reward = {avg:6.2f}")
    
    return all_stats


# ============================================================================
# Task Reward Normalization
# ============================================================================

def get_reward_normalization_factors() -> Dict[str, Tuple[float, float]]:
    """Get min/max reward ranges per task for normalization."""
    # These are estimated based on typical performance
    # Will be updated as models train
    return {
        "ideal_season": (20.0, 80.0),
        "variable_weather": (10.0, 60.0),
        "drought_year": (5.0, 40.0),
        "supply_chain_disruption": (10.0, 50.0),
        "regulatory_shift": (10.0, 50.0),
    }


def normalize_reward(task: str, reward: float) -> float:
    """Normalize reward to 0-100 scale per task."""
    normalization = get_reward_normalization_factors()
    min_val, max_val = normalization.get(task, (0.0, 100.0))
    normalized = ((reward - min_val) / (max_val - min_val)) * 100
    return max(0, min(100, normalized))  # Clamp to 0-100


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-task and task-specific DQN training")
    parser.add_argument("--mode", choices=["multitask", "task-specific", "all"], 
                       default="all", help="Training mode")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per task/total")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device")
    
    args = parser.parse_args()
    
    if args.mode == "multitask":
        train_multi_task_dqn(
            total_episodes=args.episodes,
            device=args.device,
        )
    
    elif args.mode == "task-specific":
        train_all_task_specific_models(
            num_episodes_per_task=args.episodes,
            device=args.device,
        )
    
    else:  # all
        print("\n🔥 COMPREHENSIVE TRAINING PIPELINE")
        print("Training both multi-task and task-specific models...")
        
        # Train multi-task
        train_multi_task_dqn(
            total_episodes=args.episodes,
            device=args.device,
        )
        
        print("\n" + "=" * 60)
        print("Now training task-specific models...")
        print("=" * 60)
        
        # Train task-specific
        train_all_task_specific_models(
            num_episodes_per_task=args.episodes,
            device=args.device,
        )
        
        print("\n✨ All training complete!")
        print("Available models:")
        print(f"  - models/dqn_multitask.pth (general purpose)")
        print(f"  - models/dqn_ideal_season.pth")
        print(f"  - models/dqn_variable_weather.pth")
        print(f"  - models/dqn_drought_year.pth")
        print(f"  - models/dqn_supply_chain_disruption.pth")
        print(f"  - models/dqn_regulatory_shift.pth")
