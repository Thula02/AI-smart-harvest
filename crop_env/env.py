"""CropEnv — Outcome-based OpenEnv-compliant crop management environment."""

from __future__ import annotations

import random
from typing import Any

from .graders import (
    grade_drought_year,
    grade_ideal_season,
    grade_regulatory_shift,
    grade_supply_chain_disruption,
    grade_variable_weather,
)
from .models import (
    Action,
    BudgetState,
    CropMetrics,
    CropMetricDeltas,
    EnvState,
    FertilizerType,
    GrowthStage,
    IrrigationLevel,
    Observation,
    OutcomeTrends,
    PestManagement,
    RewardBreakdown,
    StepResult,
    ToolAction,
    ToolCallType,
    ToolResult,
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
    "supply_chain_disruption": {
        "scenario": "supply_chain_disruption",
        "description": "Medium-Hard: Balanced fertilizer unavailable days 30–60 over 90 days",
    },
    "regulatory_shift": {
        "scenario": "regulatory_shift",
        "description": "Medium-Hard: Chemical pesticides banned after day 30 over 90 days",
    },
}

GRADERS = {
    "ideal_season": grade_ideal_season,
    "variable_weather": grade_variable_weather,
    "drought_year": grade_drought_year,
    "supply_chain_disruption": grade_supply_chain_disruption,
    "regulatory_shift": grade_regulatory_shift,
}


# ---------------------------------------------------------------------------
# Budget + Costs (Phase 1)
# ---------------------------------------------------------------------------

# High enough to not interfere with existing tests; can be made scenario-dependent later.
DEFAULT_BUDGET_TOTAL_USD: float = 10_000.0

# Keep consistent with inference.py defaults.
ACTION_COSTS_USD: dict[str, dict[str, float]] = {
    "irrigation": {"none": 0, "light": 10, "moderate": 25, "heavy": 50},
    "fertilizer": {"none": 0, "nitrogen": 30, "phosphorus": 25, "balanced": 45, "organic": 20},
    "pest_management": {"none": 0, "scouting": 5, "biological": 20, "chemical_light": 35, "chemical_heavy": 60},
    "tools": {"soil_test": 50, "weather_forecast": 30, "market_prices": 10},
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

        # Budget + tools (Phase 1)
        self._budget_total_usd: float | None = DEFAULT_BUDGET_TOTAL_USD
        self._budget_spent_usd: float = 0.0
        self._tool_result: ToolResult | None = None
        self._pending_tool_result: ToolResult | None = None

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

        # Reset budget + tool carryover
        self._budget_total_usd = getattr(self._scenario, "budget", DEFAULT_BUDGET_TOTAL_USD)
        self._budget_spent_usd = 0.0
        self._tool_result = None
        self._pending_tool_result = None

        # Generate initial weather
        self._prev_weather, self._active_extreme = generate_daily_weather(
            self._scenario.field_model, 0, self._total_days,
            self._rng, self._active_extreme,
        )

        return self._make_observation()

    def step(self, action: Action | ToolAction) -> StepResult:
        """Apply one action (farm OR tool), advance one day."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._scenario is None or self._metrics is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._task_name is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Tool result appears in the observation AFTER the tool call.
        self._tool_result = self._pending_tool_result
        self._pending_tool_result = None

        self._day += 1

        tool_called: ToolCallType | None = None
        tool_result_generated: ToolResult | None = None

        # Tool call replaces farm action for the day.
        if isinstance(action, ToolAction):
            tool_called = action.tool
            tool_cost_usd = float(ACTION_COSTS_USD["tools"][tool_called.value])
            self._deduct_budget(tool_cost_usd)

            tool_result_generated = self._execute_tool(tool_called, tool_cost_usd)
            self._pending_tool_result = tool_result_generated

            # Simulator still needs a farm action to advance hidden dynamics.
            effective_action = Action(
                irrigation=IrrigationLevel.NONE,
                fertilizer=FertilizerType.NONE,
                pest_management=PestManagement.NONE,
            )
            action_cost_usd = tool_cost_usd
        else:
            effective_action = self._enforce_scenario_constraints(action)
            action_cost_usd = self._compute_farm_cost(effective_action)
            self._deduct_budget(action_cost_usd)

        # Check water budget
        water_mm = IRRIGATION_MM[effective_action.irrigation]
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
            effective_action, self._metrics, self._soil_moisture, weather,
            self._scenario, growth_stage, self._history, self._day, self._rng,
        )
        self._soil_moisture = new_moisture

        # Apply deltas
        self._metrics = apply_deltas(self._metrics, deltas)
        self._prev_deltas = deltas

        # Economic component (Phase 2)
        econ_reward, profit_usd, revenue_usd = self._compute_economic_reward(
            action_cost_usd=float(action_cost_usd),
        )

        # Compute outcome-based reward (agronomy + economic blend)
        reward = compute_reward(
            deltas,
            self._task_name,
            self._metrics,
            economic_reward=econ_reward,
            economic_weight=0.5,
            profit_usd=profit_usd,
            revenue_usd=revenue_usd,
            cost_usd=float(action_cost_usd),
        )

        # Record history
        entry: dict[str, Any] = {
            "day": self._day,
            "action": effective_action.model_dump(),
            "metrics": self._metrics.model_dump(),
            "deltas": deltas.model_dump(),
            "weather": weather.model_dump(),
            "soil_moisture": self._soil_moisture,
            "water_used_total": self._water_used,
            "growth_stage": growth_stage.value,
            "reward_total": reward.total,
            "budget": self._budget_state().model_dump(),
            "action_cost_usd": round(float(action_cost_usd), 2),
            "profit_usd": reward.profit_usd,
            "revenue_usd": reward.revenue_usd,
            "economic_total": reward.economic_total,
        }
        if tool_called is not None:
            entry["tool_called"] = tool_called.value
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
            "budget": self._budget_state().model_dump(),
            "action_cost_usd": round(float(action_cost_usd), 2),
            "profit_usd": reward.profit_usd,
            "revenue_usd": reward.revenue_usd,
            "economic_total": reward.economic_total,
        }
        if tool_called is not None:
            info["tool_called"] = tool_called.value
        if tool_result_generated is not None:
            info["tool_result_generated"] = tool_result_generated.model_dump()

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
            budget=self._budget_state(),
            tool_result=self._tool_result,
        )

    # -------------------------------------------------------------------
    # Budget + tools (Phase 1)
    # -------------------------------------------------------------------

    def tool_call(self, tool: ToolCallType) -> ToolResult:
        """Out-of-band tool call: does NOT advance the day.

        Deducts budget and stores the result to appear in the NEXT Observation.
        Returns the tool result immediately (for HTTP tool endpoints).
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._scenario is None or self._metrics is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        cost_usd = float(ACTION_COSTS_USD["tools"][tool.value])
        self._deduct_budget(cost_usd)
        result = self._execute_tool(tool, cost_usd)
        self._pending_tool_result = result
        return result

    def _budget_state(self) -> BudgetState:
        if self._budget_total_usd is None:
            remaining = None
        else:
            remaining = max(0.0, self._budget_total_usd - self._budget_spent_usd)

        return BudgetState(
            total_usd=self._budget_total_usd,
            spent_usd=round(self._budget_spent_usd, 2),
            remaining_usd=None if remaining is None else round(remaining, 2),
        )

    def _deduct_budget(self, cost_usd: float) -> None:
        if cost_usd <= 0:
            return
        if self._budget_total_usd is not None and (self._budget_spent_usd + cost_usd) > self._budget_total_usd:
            raise RuntimeError("Budget exhausted")
        self._budget_spent_usd += float(cost_usd)

    def _compute_farm_cost(self, action: Action) -> float:
        return float(
            ACTION_COSTS_USD["irrigation"][action.irrigation.value]
            + ACTION_COSTS_USD["fertilizer"][action.fertilizer.value]
            + ACTION_COSTS_USD["pest_management"][action.pest_management.value]
        )

    def _execute_tool(self, tool: ToolCallType, cost_usd: float) -> ToolResult:
        if self._scenario is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if tool == ToolCallType.SOIL_TEST:
            # Reveal hidden FieldModel parameters.
            data = dict(vars(self._scenario.field_model))
        elif tool == ToolCallType.WEATHER_FORECAST:
            # Deterministic 7-day outlook (doesn't consume env RNG).
            seed = (self._seed or 0) * 1_000_003 + self._day * 9_176 + 31
            rng = random.Random(seed)

            rainfall_7d: list[float] = []
            for _ in range(7):
                r = 0.0
                if rng.random() < self._scenario.field_model.rain_probability:
                    r = max(
                        0.0,
                        rng.gauss(
                            self._scenario.field_model.rain_intensity_mean,
                            4.0,
                        ),
                    )
                rainfall_7d.append(round(r, 1))

            data = {
                "rainfall_mm_7d": rainfall_7d,
                "extreme_event_probability": 0.05,
            }
        else:  # ToolCallType.MARKET_PRICES
            seed = (self._seed or 0) * 1_000_003 + self._day * 9_176 + 97
            rng = random.Random(seed)
            base = 1.20
            seasonal = 0.15 * (self._day / max(1, self._total_days))
            price = max(0.50, rng.gauss(base + seasonal, 0.08))
            data = {"price_per_unit": round(price, 3), "currency": "USD"}

        return ToolResult(
            tool=tool,
            day_requested=self._day,
            day_available=self._day + 1,
            cost_usd=float(cost_usd),
            data=data,
        )

    def _enforce_scenario_constraints(self, action: Action) -> Action:
        if self._scenario is None:
            return action

        if self._scenario.name == "supply_chain_disruption":
            # Balanced fertilizer unavailable on days 30–60 (inclusive)
            if 30 <= self._day <= 60 and action.fertilizer == FertilizerType.BALANCED:
                raise RuntimeError(
                    "Balanced fertilizer unavailable (days 30–60) in supply_chain_disruption"
                )

        if self._scenario.name == "regulatory_shift":
            # Chemical pesticides banned after day 30
            if self._day >= 30 and action.pest_management in (
                PestManagement.CHEMICAL_LIGHT,
                PestManagement.CHEMICAL_HEAVY,
            ):
                raise RuntimeError(
                    "Chemical pesticides banned after day 30 in regulatory_shift"
                )

        return action

    def _hidden_market_price_usd(self) -> float:
        seed = (self._seed or 0) * 1_000_003 + self._day * 9_176 + 97
        rng = random.Random(seed)
        base = 1.20
        seasonal = 0.15 * (self._day / max(1, self._total_days))
        return max(0.50, rng.gauss(base + seasonal, 0.08))

    def _compute_economic_reward(self, *, action_cost_usd: float) -> tuple[float, float, float]:
        """Return (economic_reward_0_100, profit_usd, revenue_usd)."""
        revenue_usd = 0.0
        if self._metrics is not None and self._day >= self._total_days:
            price = self._hidden_market_price_usd()
            expected_units = (
                1000.0
                * (self._metrics.crop_health / 100.0)
                * (self._metrics.crop_quality / 100.0)
            )
            revenue_usd = max(0.0, expected_units) * price

        profit_usd = float(revenue_usd) - float(action_cost_usd)

        # Neutral baseline=50 when profit==0; scale is intentionally gentle.
        economic_reward = 50.0 + (profit_usd / 100.0)
        economic_reward = max(0.0, min(100.0, economic_reward))
        return economic_reward, profit_usd, revenue_usd

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
