"""CropEnv — Outcome-based OpenEnv-compliant crop management environment."""

from __future__ import annotations

import random
from typing import Any

from .graders import grade_drought_year, grade_ideal_season, grade_variable_weather
from .models import (
    Action,
    CropMetrics,
    CropMetricDeltas,
    EnvState,
    GrowthStage,
    Observation,
    OutcomeTrends,
    RewardBreakdown,
    StepResult,
    WeatherObservation,
    IRRIGATION_MM,
)
from .payoff import _linear_slope, _stddev, compute_reward
from .scenarios import (
    SCENARIOS,
    ScenarioConfig,
    generate_daily_weather,
    get_growth_stage,
)
from .simulator import apply_deltas, compute_metric_changes


# ---------------------------------------------------------------------------
# Task configs
# ---------------------------------------------------------------------------

TASK_CONFIGS: dict[str, dict[str, Any]] = {
    "ideal_season": {
        "scenario": "ideal_season",
        "description": "Easy: Optimize crop growth in favorable conditions over 60 days",
    },
    "variable_weather": {
        "scenario": "variable_weather",
        "description": "Medium: Balance all metrics under variable weather with rising pests over 90 days",
    },
    "drought_year": {
        "scenario": "drought_year",
        "description": "Hard: Maximize outcomes under drought with limited water budget over 90 days",
    },
}

GRADERS = {
    "ideal_season": grade_ideal_season,
    "variable_weather": grade_variable_weather,
    "drought_year": grade_drought_year,
}


class CropEnv:
    """Outcome-based crop management simulator implementing the OpenEnv interface.

    Rewards are driven by measured crop/field metric changes (outcomes),
    not predefined action-quality scores.  Each scenario has hidden
    soil/weather/pest dynamics the agent must learn through experience.
    """

    def __init__(self, seed: int | None = None):
        self._seed = seed
        self._rng = random.Random(seed)
        self._task_name: str | None = None
        self._config: dict[str, Any] = {}
        self._scenario: ScenarioConfig | None = None
        self._day: int = 0
        self._total_days: int = 0
        self._metrics: CropMetrics | None = None
        self._soil_moisture: float = 0.0
        self._water_used: float = 0.0
        self._prev_deltas: CropMetricDeltas = CropMetricDeltas(
            crop_health=0, growth_rate=0, soil_health=0, water_stress=0,
            nutrient_stress=0, pest_pressure=0, crop_quality=0, environmental_score=0,
        )
        self._prev_weather: WeatherObservation = WeatherObservation(
            temperature=25.0, rainfall_mm=0.0,
        )
        self._active_extreme: dict | None = None
        self._history: list[dict[str, Any]] = []
        self._cumulative_reward: float = 0.0
        self._done: bool = False

    # -------------------------------------------------------------------
    # OpenEnv interface
    # -------------------------------------------------------------------

    def reset(self, task_name: str) -> Observation:
        """Initialize a new episode for the given task."""
        if task_name not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task: {task_name}. Available: {list(TASK_CONFIGS.keys())}"
            )

        self._task_name = task_name
        self._config = TASK_CONFIGS[task_name]
        self._scenario = SCENARIOS[self._config["scenario"]]
        self._day = 0
        self._total_days = self._scenario.total_days

        # Clone starting metrics
        self._metrics = self._scenario.starting_metrics.model_copy()
        self._soil_moisture = self._scenario.starting_soil_moisture
        self._water_used = 0.0
        self._prev_deltas = CropMetricDeltas(
            crop_health=0, growth_rate=0, soil_health=0, water_stress=0,
            nutrient_stress=0, pest_pressure=0, crop_quality=0, environmental_score=0,
        )
        self._active_extreme = None
        self._history = []
        self._cumulative_reward = 0.0
        self._done = False

        # Generate initial weather
        self._prev_weather, self._active_extreme = generate_daily_weather(
            self._scenario.field_model, 0, self._total_days,
            self._rng, self._active_extreme,
        )

        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        """Apply action, advance one day."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._scenario is None or self._metrics is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._day += 1

        # Check water budget
        water_mm = IRRIGATION_MM[action.irrigation]
        budget_exceeded = False
        if self._scenario.water_budget < float("inf"):
            if self._water_used + water_mm > self._scenario.water_budget:
                # Over budget: cap irrigation at what's left
                water_mm = max(0.0, self._scenario.water_budget - self._water_used)
                budget_exceeded = True
        self._water_used += water_mm

        # Generate weather for this day
        weather, self._active_extreme = generate_daily_weather(
            self._scenario.field_model, self._day, self._total_days,
            self._rng, self._active_extreme,
        )
        self._prev_weather = weather

        # Growth stage
        growth_stage = get_growth_stage(self._day, self._total_days)

        # Compute metric changes (the hidden response model)
        deltas, new_moisture = compute_metric_changes(
            action, self._metrics, self._soil_moisture, weather,
            self._scenario, growth_stage, self._history, self._day, self._rng,
        )
        self._soil_moisture = new_moisture

        # Apply deltas
        self._metrics = apply_deltas(self._metrics, deltas)
        self._prev_deltas = deltas

        # Compute outcome-based reward (blended: deltas + state quality)
        reward = compute_reward(deltas, self._task_name, self._metrics)

        # Record history
        entry: dict[str, Any] = {
            "day": self._day,
            "action": action.model_dump(),
            "metrics": self._metrics.model_dump(),
            "deltas": deltas.model_dump(),
            "weather": weather.model_dump(),
            "soil_moisture": self._soil_moisture,
            "water_used_total": self._water_used,
            "growth_stage": growth_stage.value,
            "reward_total": reward.total,
        }
        self._history.append(entry)
        self._cumulative_reward += reward.total

        # Check done
        self._done = self._day >= self._total_days

        # Build observation
        obs = self._make_observation()

        # Info dict
        info: dict[str, Any] = {
            "growth_stage": growth_stage.value,
            "water_budget_remaining": max(
                0.0, self._scenario.water_budget - self._water_used
            ) if self._scenario.water_budget < float("inf") else None,
            "budget_exceeded": budget_exceeded,
        }

        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info=info,
        )

    def state(self) -> EnvState:
        """Return full internal state for debugging/grading."""
        growth_stage = get_growth_stage(self._day, self._total_days) if self._day > 0 else GrowthStage.GERMINATION
        return EnvState(
            day=self._day,
            total_days=self._total_days,
            growth_stage=growth_stage,
            scenario_name=self._scenario.name if self._scenario else "",
            metrics=self._metrics or CropMetrics(
                crop_health=50, growth_rate=1, soil_health=50, water_stress=50,
                nutrient_stress=50, pest_pressure=50, crop_quality=50, environmental_score=50,
            ),
            soil_moisture=self._soil_moisture,
            water_used_total=self._water_used,
            history=self._history,
            cumulative_reward=round(self._cumulative_reward, 2),
        )

    def grade(self) -> float:
        """Compute the grader score (0.0–1.0) for the current task."""
        if self._task_name is None:
            raise RuntimeError("No task has been run. Call reset() first.")
        grader = GRADERS[self._task_name]
        return grader(self._history)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _make_observation(self) -> Observation:
        """Build observation from current state."""
        trends = None
        if len(self._history) >= 7:
            trends = self._compute_trends()

        growth_stage = get_growth_stage(self._day, self._total_days) if self._day > 0 else GrowthStage.GERMINATION

        return Observation(
            day=self._day,
            total_days=self._total_days,
            growth_stage=growth_stage,
            metrics=self._metrics or CropMetrics(
                crop_health=50, growth_rate=1, soil_health=50, water_stress=50,
                nutrient_stress=50, pest_pressure=50, crop_quality=50, environmental_score=50,
            ),
            deltas=self._prev_deltas,
            weather=self._prev_weather,
            trends=trends,
            scenario_name=self._scenario.name if self._scenario else "",
            soil_moisture=self._soil_moisture,
            water_used_total=self._water_used,
        )

    def _compute_trends(self) -> OutcomeTrends:
        """Compute 7-day trends for each metric and reward."""
        recent = self._history[-7:]
        rewards = [h["reward_total"] for h in recent]

        def _metric_trend(key: str) -> float:
            vals = [h["metrics"][key] for h in recent]
            return round(_linear_slope(vals), 4)

        return OutcomeTrends(
            crop_health_trend=_metric_trend("crop_health"),
            growth_rate_trend=_metric_trend("growth_rate"),
            soil_health_trend=_metric_trend("soil_health"),
            water_stress_trend=_metric_trend("water_stress"),
            nutrient_stress_trend=_metric_trend("nutrient_stress"),
            pest_pressure_trend=_metric_trend("pest_pressure"),
            reward_trend=round(_linear_slope(rewards), 4),
            reward_consistency=round(_stddev(rewards), 4),
        )
