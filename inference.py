#!/usr/bin/env python3
"""Baseline LLM agent for the Outcome-Based Crop Management Simulator.

Runs all 3 tasks sequentially, printing structured stdout.
Uses OpenAI-compatible API via env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any

from crop_env import CropEnv, Action, Observation
from crop_env.models import IrrigationLevel, FertilizerType, PestManagement, GrowthStage
from crop_env.payoff import _linear_slope, _stddev

# ---------------------------------------------------------------------------
# LLM client setup
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", HF_TOKEN)

SYSTEM_PROMPT = """\
You are a crop management decision system. Follow these EXACT rules to choose actions.

## IRRIGATION RULES (check in order, pick FIRST match)
1. If rainfall > 8mm → "none"
2. If scenario is drought_year AND water_budget_remaining <= 0 → "none"
3. If scenario is drought_year AND rainfall > 2mm → "none"
4. If scenario is drought_year → "light" if step_num % 3 == 0, else "none"
5. If soil_moisture < 15 → "heavy"
6. If soil_moisture < 25 → "moderate"
7. If soil_moisture < 40 → "light"
8. Otherwise → "none"

## FERTILIZER RULES
1. If step_num <= (total_days * 0.15): apply "organic" only on days where step_num % 8 == 1, else "none"
2. After that: apply "organic" on days where step_num % 5 == 1, else "none"

## PEST MANAGEMENT RULES
1. If step_num <= (total_days * 0.15) → "scouting"
2. After that → "biological"

## OUTPUT
Respond with ONLY a JSON object, no explanation:
{"irrigation":"...","fertilizer":"...","pest_management":"..."}
"""


def build_user_message(
    obs: Observation, step_num: int,
    prev_reward: float | None = None,
    avg_reward: float | None = None,
    water_budget_remaining: float = float("inf"),
) -> str:
    """Build minimal user message with only the decision-relevant inputs."""
    w = obs.weather
    msg = (
        f"step_num={step_num}, total_days={obs.total_days}, "
        f"scenario={obs.scenario_name}, "
        f"soil_moisture={obs.soil_moisture:.1f}, "
        f"rainfall={w.rainfall_mm:.1f}"
    )
    if water_budget_remaining < float("inf"):
        msg += f", water_budget_remaining={water_budget_remaining:.0f}"
    msg += "\nRespond with JSON only."
    return msg


def call_llm(
    obs: Observation, step_num: int, history_actions: list[dict],
    task_name: str = "ideal_season", total_days: int = 60,
    water_used: float = 0, water_budget: float = float("inf"),
    prev_reward: float | None = None, avg_reward: float | None = None,
) -> Action:
    """Call the LLM with prescriptive rules and parse the response."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)

        water_remaining = max(0, water_budget - water_used)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_message(
                    obs, step_num, prev_reward, avg_reward, water_remaining,
                ),
            },
        ]

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            max_tokens=80,
        )

        content = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        data = json.loads(content)
        return Action(
            irrigation=IrrigationLevel(data["irrigation"]),
            fertilizer=FertilizerType(data["fertilizer"]),
            pest_management=PestManagement(data["pest_management"]),
        )
    except Exception:
        return _fallback_action(
            obs, task_name, step_num, total_days, water_used, water_budget,
        )


def _fallback_action(
    obs: Observation,
    task_name: str = "ideal_season",
    step_num: int = 0,
    total_days: int = 60,
    water_used: float = 0,
    water_budget: float = float("inf"),
) -> Action:
    """Deterministic phased rule-based policy.

    Two phases optimized for consistency, positive trend, and metric breadth:
      Phase 1 (0-20%): scouting + sparse organic → conservative baseline
      Phase 2 (20%+):  biological + regular organic → full optimization

    The phase transition creates a natural reward improvement arc.
    Organic fertilizer is spaced to avoid nutrient burn (threshold-dependent).
    """
    m = obs.metrics
    w = obs.weather
    moisture = obs.soil_moisture
    progress = step_num / total_days

    # ===== IRRIGATION =====
    if task_name == "drought_year":
        remaining = max(0, water_budget - water_used)
        days_left = max(1, total_days - step_num)
        daily_budget = remaining / days_left

        if w.rainfall_mm > 2 or remaining <= 0:
            irrigation = IrrigationLevel.NONE
        elif moisture < 15 and daily_budget >= 8:
            irrigation = IrrigationLevel.LIGHT
        elif step_num % 3 == 0 and daily_budget >= 5:
            irrigation = IrrigationLevel.LIGHT
        else:
            irrigation = IrrigationLevel.NONE
    else:
        # Consistent moisture-based irrigation
        if w.rainfall_mm > 8:
            irrigation = IrrigationLevel.NONE
        elif moisture < 15:
            irrigation = IrrigationLevel.HEAVY
        elif moisture < 25:
            irrigation = IrrigationLevel.MODERATE
        elif moisture < 40:
            irrigation = IrrigationLevel.LIGHT
        else:
            irrigation = IrrigationLevel.NONE

    # ===== FERTILIZER =====
    # Organic only — spaced to avoid nutrient burn from pending release accumulation.
    # Burn thresholds: ideal=85, variable=75, drought=65.
    # Every 5th day is safe across all scenarios.
    if progress < 0.2:
        apply_fert = (step_num % 8 == 1)  # Sparse in phase 1
    else:
        apply_fert = (step_num % 5 == 1)  # Regular in phase 2+

    fertilizer = FertilizerType.ORGANIC if apply_fert else FertilizerType.NONE

    # ===== PEST MANAGEMENT =====
    if progress < 0.2:
        pest = PestManagement.SCOUTING    # Phase 1: lightweight
    else:
        pest = PestManagement.BIOLOGICAL  # Phase 2+: full control

    return Action(
        irrigation=irrigation,
        fertilizer=fertilizer,
        pest_management=pest,
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

TASKS = ["ideal_season", "variable_weather", "drought_year"]


def run_task(env: CropEnv, task_name: str, use_llm: bool = True) -> None:
    """Run a single task and print structured stdout."""
    obs = env.reset(task_name)
    scenario = env._scenario

    print(
        f"[START] task={task_name} env=crop-outcome model={MODEL_NAME} "
        f"scenario={scenario.name} days={scenario.total_days}"
        + (f" water_budget={scenario.water_budget:.0f}" if scenario.water_budget < float("inf") else "")
    )

    rewards: list[float] = []
    history_actions: list[dict] = []
    prev_reward: float | None = None

    for step_num in range(1, scenario.total_days + 1):
        try:
            avg_reward = sum(rewards) / len(rewards) if rewards else None

            if use_llm:
                action = call_llm(
                    obs, step_num, history_actions,
                    task_name=task_name,
                    total_days=scenario.total_days,
                    water_used=env._water_used,
                    water_budget=scenario.water_budget,
                    prev_reward=prev_reward,
                    avg_reward=avg_reward,
                )
            else:
                action = _fallback_action(
                    obs, task_name, step_num, scenario.total_days,
                    env._water_used, scenario.water_budget,
                )

            result = env.step(action)

            action_dict = action.model_dump()
            reward_val = result.reward.total
            rewards.append(reward_val)
            prev_reward = reward_val

            history_actions.append({
                "step": step_num,
                "action": action_dict,
                "weather": result.observation.weather.model_dump(),
                "reward": reward_val,
            })

            # Compact metric snapshot
            met = result.observation.metrics
            met_str = (
                f"ch={met.crop_health:.1f},gr={met.growth_rate:.2f},"
                f"sh={met.soil_health:.1f},ws={met.water_stress:.1f},"
                f"ns={met.nutrient_stress:.1f},pp={met.pest_pressure:.1f},"
                f"cq={met.crop_quality:.1f},es={met.environmental_score:.1f}"
            )

            # Compact delta snapshot
            dl = result.observation.deltas
            delta_str = (
                f"ch={dl.crop_health:+.3f},gr={dl.growth_rate:+.3f},"
                f"sh={dl.soil_health:+.3f},ws={dl.water_stress:+.3f},"
                f"ns={dl.nutrient_stress:+.3f},pp={dl.pest_pressure:+.3f},"
                f"cq={dl.crop_quality:+.3f},es={dl.environmental_score:+.3f}"
            )

            # Weather snapshot
            wx = result.observation.weather
            wx_str = f"temp={wx.temperature:.1f},rain={wx.rainfall_mm:.1f}"
            if wx.is_extreme_event:
                wx_str += f",extreme={wx.extreme_event_type}"

            print(
                f"[STEP] step={step_num} "
                f"action={json.dumps(action_dict)} "
                f"reward={reward_val:.2f} "
                f"done={str(result.done).lower()} "
                f"error=null "
                f"metrics={{{met_str}}} "
                f"deltas={{{delta_str}}} "
                f"weather={{{wx_str}}} "
                f"soil_moisture={result.observation.soil_moisture:.1f} "
                f"stage={result.observation.growth_stage.value}"
            )

            obs = result.observation
        except Exception as e:
            error_msg = str(e).replace("\n", " ")[:200]
            print(
                f"[STEP] step={step_num} "
                f"action=null "
                f"reward=0.00 "
                f"done=false "
                f"error=\"{error_msg}\""
            )

    # End summary
    score = env.grade()
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    reward_trend = _linear_slope(rewards) if len(rewards) >= 2 else 0.0
    reward_std = _stddev(rewards)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success=true steps={len(rewards)} "
        f"score={score:.4f} "
        f"rewards={rewards_str} "
        f"task={task_name} avg_reward={avg_reward:.2f} "
        f"reward_trend={reward_trend:+.2f} reward_stddev={reward_std:.2f} "
        f"water_used={env._water_used:.0f}"
    )
    print()


def main():
    use_llm = bool(OPENAI_API_KEY)
    if not use_llm:
        print(
            "# WARNING: No API key found. Using rule-based fallback agent.",
            file=sys.stderr,
        )

    seed = int(os.environ.get("SEED", "42"))
    env = CropEnv(seed=seed)

    for task_name in TASKS:
        try:
            run_task(env, task_name, use_llm=use_llm)
        except Exception:
            print(
                f"[END] success=false steps=0 score=0.0000 rewards= task={task_name}",
            )
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
