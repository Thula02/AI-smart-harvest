"""
Gradio UI for Crop Environment with DQN Agent Simulator and Analytics
"""

import gradio as gr
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from pathlib import Path

from crop_env import CropEnv
from crop_env.models import Action, IrrigationLevel, FertilizerType, PestManagement
from agent_inference import DQNAgentManager, observation_to_tensor, discrete_to_action, get_available_models
from train_dqn import SimpleLinearDQN, NUM_ACTIONS
from train_multitask import normalize_reward, TASK_DIFFICULTY


# ============================================================================
# Reward Normalization
# ============================================================================

def get_normalized_reward(task: str, raw_reward: float) -> float:
    """Get normalized reward (0-100) for a task."""
    return normalize_reward(task, raw_reward)


def get_task_recommendation(task: str) -> str:
    """Get model recommendation for a task."""
    difficulty = TASK_DIFFICULTY.get(task, 2)
    model_names = {
        "ideal_season": "dqn_ideal_season (specialized)",
        "variable_weather": "dqn_multitask (general purpose)",
        "drought_year": "dqn_drought_year (specialized)",
        "supply_chain_disruption": "dqn_multitask (general purpose)",
        "regulatory_shift": "dqn_multitask (general purpose)",
    }
    stars = "⭐" * difficulty
    recommended = model_names.get(task, "dqn_multitask")
    return f"{stars} Difficulty {difficulty}/3 - Recommended: {recommended}"


# ============================================================================
# Global State
# ============================================================================

env = CropEnv(seed=42)
agent_manager = DQNAgentManager(device="cpu")

# Track simulation history
simulation_history = {
    "rewards": [],
    "crop_health": [],
    "growth_rate": [],
    "soil_health": [],
    "water_stress": [],
    "pest_pressure": [],
    "days": [],
    "actions": [],
}


# ============================================================================
# Utility Functions
# ============================================================================

def reset_history():
    """Reset simulation history."""
    global simulation_history
    simulation_history = {
        "rewards": [],
        "crop_health": [],
        "growth_rate": [],
        "soil_health": [],
        "water_stress": [],
        "pest_pressure": [],
        "days": [],
        "actions": [],
    }


def run_dqn_simulation(
    model_path: str,
    task_name: str,
    num_episodes: int = 1,
) -> Tuple[Dict, str, plt.Figure, plt.Figure]:
    """Run DQN agent simulation and collect metrics."""
    
    reset_history()
    
    try:
        model = agent_manager.load_model(model_path)
    except Exception as e:
        return {}, f"❌ Error loading model: {str(e)}", None, None
    
    all_episode_rewards = []
    all_episode_scores = []
    all_episodes_data = []
    
    for episode in range(num_episodes):
        obs = env.reset(task_name)
        state = observation_to_tensor(obs)
        
        episode_reward = 0
        episode_data = {
            "rewards": [],
            "crop_health": [],
            "growth_rate": [],
            "soil_health": [],
            "water_stress": [],
            "pest_pressure": [],
            "actions": [],
        }
        
        while True:
            action_idx, _ = agent_manager.select_action(model, state, deterministic=True)
            action = discrete_to_action(action_idx)
            
            result = env.step(action)
            next_obs = result.observation
            reward = result.reward.total
            done = result.done
            
            episode_reward += reward
            
            # Collect metrics
            episode_data["rewards"].append(reward)
            episode_data["crop_health"].append(float(next_obs.metrics.crop_health))
            episode_data["growth_rate"].append(float(next_obs.metrics.growth_rate))
            episode_data["soil_health"].append(float(next_obs.metrics.soil_health))
            episode_data["water_stress"].append(float(next_obs.metrics.water_stress))
            episode_data["pest_pressure"].append(float(next_obs.metrics.pest_pressure))
            episode_data["actions"].append({
                "irrigation": action.irrigation.value,
                "fertilizer": action.fertilizer.value,
                "pest_management": action.pest_management.value,
            })
            
            state = observation_to_tensor(next_obs)
            
            if done:
                final_score = env.grade()
                break
        
        # Accumulate episode data
        all_episode_rewards.append(episode_reward)
        all_episode_scores.append(final_score)
        all_episodes_data.append(episode_data)
    
    # Store history for visualization (use last episode or aggregate)
    if num_episodes == 1:
        # Single episode: store as-is
        episode_data = all_episodes_data[0]
        simulation_history["rewards"] = episode_data["rewards"]
        simulation_history["crop_health"] = episode_data["crop_health"]
        simulation_history["growth_rate"] = episode_data["growth_rate"]
        simulation_history["soil_health"] = episode_data["soil_health"]
        simulation_history["water_stress"] = episode_data["water_stress"]
        simulation_history["pest_pressure"] = episode_data["pest_pressure"]
        simulation_history["days"] = list(range(1, len(episode_data["rewards"]) + 1))
        simulation_history["actions"] = episode_data["actions"]
    else:
        # Multiple episodes: aggregate by averaging across episodes (show episode-level metrics)
        simulation_history["rewards"] = all_episode_rewards  # Cumulative rewards per episode
        simulation_history["crop_health"] = [np.mean([ep["crop_health"][-1] for ep in all_episodes_data])] * num_episodes
        simulation_history["growth_rate"] = [np.mean([ep["growth_rate"][-1] for ep in all_episodes_data])] * num_episodes
        simulation_history["soil_health"] = [np.mean([ep["soil_health"][-1] for ep in all_episodes_data])] * num_episodes
        simulation_history["water_stress"] = [np.mean([ep["water_stress"][-1] for ep in all_episodes_data])] * num_episodes
        simulation_history["pest_pressure"] = [np.mean([ep["pest_pressure"][-1] for ep in all_episodes_data])] * num_episodes
        simulation_history["days"] = list(range(1, num_episodes + 1))
        simulation_history["actions"] = []  # Aggregate: skip detailed actions for multi-episode
    
    # Create summary
    summary = {
        "Total Episodes": num_episodes,
        "Average Raw Reward": f"{np.mean(all_episode_rewards):.2f}",
        "Average Normalized Reward": f"{np.mean([get_normalized_reward(task_name, r) for r in all_episode_rewards]):.1f}/100",
        "Max Reward": f"{np.max(all_episode_rewards):.2f}",
        "Min Reward": f"{np.min(all_episode_rewards):.2f}",
        "Average Score": f"{np.mean(all_episode_scores):.2f}",
        "Task": task_name,
        "Difficulty": TASK_DIFFICULTY.get(task_name, 2),
        "Model": Path(model_path).name,
        "Recommendation": get_task_recommendation(task_name),
    }
    
    # Create figures
    fig1 = create_reward_plot()
    fig2 = create_metrics_plot()
    
    status_msg = f"✅ Simulation completed! {num_episodes} episode(s) run successfully."
    
    return summary, status_msg, fig1, fig2


def create_reward_plot() -> plt.Figure:
    """Create reward fluctuation plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if simulation_history["days"]:
        days = simulation_history["days"]
        rewards = simulation_history["rewards"]
        
        # Determine if single episode or multiple episodes
        if len(days) > 20:  # Likely single episode (many days)
            ax.plot(days, rewards, marker='o', linewidth=2, markersize=4, color='#2E86AB', label='Daily Reward')
            ax.fill_between(days, rewards, alpha=0.3, color='#2E86AB')
            ax.set_xlabel("Day", fontsize=12, fontweight='bold')
            title = "DQN Agent: Daily Reward Fluctuations"
        else:  # Multiple episodes
            ax.plot(days, rewards, marker='o', linewidth=2, markersize=6, color='#2E86AB', label='Episode Reward')
            ax.bar(days, rewards, alpha=0.3, color='#2E86AB')
            ax.set_xlabel("Episode", fontsize=12, fontweight='bold')
            title = "DQN Agent: Episode Rewards"
        
        # Add trend line if enough points
        if len(days) > 2:
            z = np.polyfit(days, rewards, min(2, len(days)-1))
            p = np.poly1d(z)
            ax.plot(days, p(days), "--", color='red', linewidth=2, label='Trend')
        
        ax.set_ylabel("Reward", fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    return fig


def create_metrics_plot() -> plt.Figure:
    """Create crop metrics plot."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("DQN Agent: Crop Metrics Over Time", fontsize=16, fontweight='bold')
    
    if simulation_history["days"]:
        # Determine if single episode or multiple
        is_multi_episode = len(simulation_history["days"]) <= 20
        x_label = "Episode" if is_multi_episode else "Day"
        
        metrics = [
            ("crop_health", "Crop Health", axes[0, 0]),
            ("growth_rate", "Growth Rate", axes[0, 1]),
            ("soil_health", "Soil Health", axes[0, 2]),
            ("water_stress", "Water Stress", axes[1, 0]),
            ("pest_pressure", "Pest Pressure", axes[1, 1]),
        ]
        
        for key, title, ax in metrics:
            if key in simulation_history and simulation_history[key]:
                values = simulation_history[key]
                if is_multi_episode:
                    ax.bar(simulation_history["days"], values, color='#A23B72', alpha=0.6)
                else:
                    ax.plot(simulation_history["days"], values, 
                           marker='o', linewidth=2, markersize=4, color='#A23B72')
                    ax.fill_between(simulation_history["days"], values, alpha=0.3, color='#A23B72')
                
                ax.set_xlabel(x_label, fontsize=10)
                ax.set_ylabel("Value", fontsize=10)
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
        
        # Remove extra subplot
        if 5 < len(axes.flat):
            fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    return fig


def get_model_list() -> List[str]:
    """Get list of available models."""
    models = get_available_models("models")
    return [f"models/{m}" for m in models] if models else ["No models found"]


def create_episode_summary_table() -> pd.DataFrame:
    """Create summary table for episode actions."""
    if not simulation_history["actions"]:
        # Multi-episode mode: show episode-level summary
        if simulation_history["rewards"] and simulation_history["days"]:
            df_data = []
            for day, reward in zip(simulation_history["days"], simulation_history["rewards"]):
                df_data.append({
                    "Episode": day,
                    "Total Reward": f"{reward:.2f}",
                    "Avg Health": f"{simulation_history['crop_health'][day-1]:.1f}" if day <= len(simulation_history['crop_health']) else "N/A",
                })
            return pd.DataFrame(df_data)
        return pd.DataFrame()
    
    # Single episode mode: show day-by-day actions
    df_data = []
    for day, action in enumerate(simulation_history["actions"], 1):
        reward = simulation_history["rewards"][day - 1] if day - 1 < len(simulation_history["rewards"]) else 0
        df_data.append({
            "Day": day,
            "Irrigation": action["irrigation"],
            "Fertilizer": action["fertilizer"],
            "Pest Mgmt": action["pest_management"],
            "Reward": f"{reward:.2f}",
        })
    
    return pd.DataFrame(df_data)


# ============================================================================
# Gradio Interface
# ============================================================================

def create_gradio_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="🌾 Crop Environment DQN Simulator", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("# 🌾 Crop Management DQN Agent Simulator")
        gr.Markdown("*Visualize and analyze Deep Q-Network agent performance on crop management tasks*")
        
        with gr.Tabs():
            
            # ====== TAB 1: AUTO SIMULATOR ======
            with gr.Tab("⚡ Auto Simulator"):
                gr.Markdown("### Automatic DQN Simulation")
                gr.Markdown("Run the DQN agent with default settings and watch it manage crops!")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        auto_task = gr.Dropdown(
                            choices=[
                                "ideal_season",
                                "variable_weather",
                                "drought_year",
                                "supply_chain_disruption",
                                "regulatory_shift",
                            ],
                            value="ideal_season",
                            label="📋 Task Type",
                            info="Select crop management scenario"
                        )
                        
                        auto_episodes = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=1,
                            step=1,
                            label="🔄 Number of Episodes",
                            info="How many episodes to simulate"
                        )
                    
                    with gr.Column(scale=1):
                        auto_recommendation = gr.Textbox(
                            label="💡 Task Info",
                            interactive=False,
                            lines=3,
                            value="⭐ Difficulty 1/3 - Recommended: dqn_ideal_season"
                        )
                
                # Update recommendation when task changes
                auto_task.change(
                    fn=lambda task: get_task_recommendation(task),
                    inputs=auto_task,
                    outputs=auto_recommendation
                )
                
                with gr.Row():
                    auto_run_btn = gr.Button("🚀 Run Simulation", scale=1, size="lg", variant="primary")
                
                with gr.Row():
                    auto_status = gr.Textbox(label="Status", interactive=False, lines=2)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        auto_summary = gr.JSON(label="📊 Summary Statistics")
                    with gr.Column(scale=1):
                        gr.Markdown("### 📈 Key Metrics\n- **Reward**: Total task reward\n- **Score**: Final grader score\n- **Episodes**: Number of simulations")
                
                with gr.Row():
                    auto_reward_plot = gr.Plot(label="Reward Fluctuations")
                
                with gr.Row():
                    auto_metrics_plot = gr.Plot(label="Crop Metrics Over Time")
                
                with gr.Row():
                    auto_actions_table = gr.Dataframe(label="📋 Agent Actions Trace", interactive=False)
                
                auto_run_btn.click(
                    fn=lambda task, episodes: run_dqn_simulation(
                        model_path="models/dqn_model.pth",
                        task_name=task,
                        num_episodes=episodes
                    ),
                    inputs=[auto_task, auto_episodes],
                    outputs=[auto_summary, auto_status, auto_reward_plot, auto_metrics_plot]
                ).then(
                    fn=create_episode_summary_table,
                    outputs=auto_actions_table
                )
            
            # ====== TAB 2: CUSTOM SIMULATOR ======
            with gr.Tab("⚙️ Custom Simulator"):
                gr.Markdown("### Custom DQN Configuration")
                gr.Markdown("Fine-tune DQN parameters and run custom simulations")
                
                with gr.Row():
                    with gr.Column():
                        custom_model = gr.Dropdown(
                            choices=get_model_list(),
                            value="models/dqn_model.pth",
                            label="🤖 Model Selection",
                            info="Choose: general (multitask) or task-specific models"
                        )
                        
                        gr.Markdown("""
                        **Model Types:**
                        - `dqn_multitask` — Trained on all tasks (general purpose)
                        - `dqn_[task]` — Specialized for specific tasks (better performance)
                        """)
                    
                    with gr.Column():
                        custom_task = gr.Dropdown(
                            choices=[
                                "ideal_season",
                                "variable_weather",
                                "drought_year",
                                "supply_chain_disruption",
                                "regulatory_shift",
                            ],
                            value="variable_weather",
                            label="📋 Task Type"
                        )
                        
                        custom_task_info = gr.Textbox(
                            label="📊 Task Info",
                            interactive=False,
                            lines=2,
                            value="⭐⭐ Difficulty 2/3"
                        )
                        
                        custom_task.change(
                            fn=lambda task: get_task_recommendation(task),
                            inputs=custom_task,
                            outputs=custom_task_info
                        )
                
                with gr.Row():
                    with gr.Column():
                        custom_episodes = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=3,
                            step=1,
                            label="🔄 Episodes"
                        )
                    
                    with gr.Column():
                        custom_seed = gr.Slider(
                            minimum=0,
                            maximum=1000,
                            value=42,
                            step=1,
                            label="🎲 Random Seed"
                        )
                
                with gr.Row():
                    custom_run_btn = gr.Button("🚀 Run Custom Simulation", size="lg", variant="primary")
                
                with gr.Row():
                    custom_status = gr.Textbox(label="Status", interactive=False, lines=2)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        custom_summary = gr.JSON(label="📊 Results")
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### 📈 Understanding Rewards
                        
                        **Raw Reward**: Actual reward from environment (varies by task difficulty)
                        
                        **Normalized Reward**: Scaled to 0-100 for fair comparison across tasks
                        
                        - **Easy tasks** (ideal_season): Higher raw scores
                        - **Hard tasks** (drought_year): Lower raw scores but normalized fairly
                        
                        ### 💡 Model Selection
                        Choose **task-specific models** for best performance on that task!
                        """)
                
                with gr.Row():
                    custom_reward_plot = gr.Plot(label="Reward Analysis")
                
                with gr.Row():
                    custom_metrics_plot = gr.Plot(label="Performance Metrics")
                
                custom_run_btn.click(
                    fn=lambda model, task, episodes, seed: run_dqn_simulation(
                        model_path=model,
                        task_name=task,
                        num_episodes=episodes
                    ),
                    inputs=[custom_model, custom_task, custom_episodes, custom_seed],
                    outputs=[custom_summary, custom_status, custom_reward_plot, custom_metrics_plot]
                )
            
            # ====== TAB 3: ANALYTICS ======
            with gr.Tab("📊 Analytics"):
                gr.Markdown("### DQN Performance Analytics & Reward Normalization")
                gr.Markdown("Detailed analysis of agent behavior and fair task comparison")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("""
                        ### 📈 Reward Normalization Explained
                        
                        Different tasks have different **reward ranges**:
                        - **Ideal Season** (Easy): Raw rewards 20-80 → Normalized 0-100
                        - **Variable Weather** (Medium): Raw rewards 10-60 → Normalized 0-100
                        - **Drought Year** (Hard): Raw rewards 5-40 → Normalized 0-100
                        
                        This ensures **fair comparison** across difficulties!
                        
                        ### 🎯 How to Use
                        1. **Single Episode**: See detailed day-by-day breakdown
                        2. **Multiple Episodes**: Compare consistency across runs
                        3. **Different Tasks**: Use normalized scores for fair comparison
                        """)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### ⭐ Task Difficulties
                        
                        **⭐ Easy (Difficulty 1)**
                        - ideal_season
                        
                        **⭐⭐ Medium (Difficulty 2)**
                        - variable_weather
                        - supply_chain_disruption
                        
                        **⭐⭐⭐ Hard (Difficulty 3)**
                        - drought_year
                        - regulatory_shift
                        """)
                
                with gr.Row():
                    analytics_actions_table = gr.Dataframe(
                        label="📋 Detailed Action History",
                        interactive=False
                    )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 💡 Interpreting Results")
                        gr.Markdown(
                            "- **High normalized score** = Excellent policy for that task\n"
                            "- **Low raw but high normalized** = Good for hard tasks\n"
                            "- **Consistent across episodes** = Stable agent behavior\n"
                            "- **Improving trend** = Agent is learning over time"
                        )
                    
                    with gr.Column():
                        gr.Markdown("### 🎯 Model Selection Tips")
                        gr.Markdown(
                            "- Use **task-specific models** for 10-20% better performance\n"
                            "- Use **multitask model** for unknown/mixed scenarios\n"
                            "- Compare across tasks with **normalized scores**\n"
                            "- Train new models with `train_multitask.py`"
                        )
    
    return interface


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(share=True, server_name="0.0.0.0", server_port=7860)
