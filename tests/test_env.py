"""Comprehensive tests for the Crop Management environment."""

import random
from typing import Any

import pytest

from crop_env import (
    Action,
    CropEnv,
    CropMetrics,
    CropMetricDeltas,
    FertilizerType,
    GrowthStage,
    IrrigationLevel,
    Observation,
    PestManagement,
    RewardBreakdown,
    StepResult,
)
from crop_env.models import EnvState, OutcomeTrends, WeatherObservation
from crop_env.scenarios import SCENARIOS, get_growth_stage


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------

class TestModels:
    def test_action_creation(self):
        a = Action(
            irrigation=IrrigationLevel.MODERATE,
            fertilizer=FertilizerType.BALANCED,
            pest_management=PestManagement.SCOUTING,
        )
        assert a.irrigation == IrrigationLevel.MODERATE
        assert a.fertilizer == FertilizerType.BALANCED
        assert a.pest_management == PestManagement.SCOUTING

    def test_crop_metrics_validation(self):
        m = CropMetrics(
            crop_health=75, growth_rate=3.0, soil_health=60,
            water_stress=20, nutrient_stress=15, pest_pressure=10,
            crop_quality=70, environmental_score=80,
        )
        assert m.crop_health == 75

    def test_crop_metrics_out_of_range(self):
        with pytest.raises(Exception):
            CropMetrics(
                crop_health=150, growth_rate=3.0, soil_health=60,
                water_stress=20, nutrient_stress=15, pest_pressure=10,
                crop_quality=70, environmental_score=80,
            )

    def test_deltas_allow_negative(self):
        d = CropMetricDeltas(
            crop_health=-5, growth_rate=-0.5, soil_health=-2,
            water_stress=10, nutrient_stress=5, pest_pressure=3,
            crop_quality=-1, environmental_score=-3,
        )
        assert d.crop_health == -5

    def test_action_enum_values(self):
        assert len(IrrigationLevel) == 4
        assert len(FertilizerType) == 5
        assert len(PestManagement) == 5
        # Total action space: 4 × 5 × 5 = 100
        total = len(IrrigationLevel) * len(FertilizerType) * len(PestManagement)
        assert total == 100

    def test_growth_stages(self):
        assert len(GrowthStage) == 5


# ---------------------------------------------------------------------------
# Scenario tests
# ---------------------------------------------------------------------------

class TestScenarios:
    def test_all_scenarios_exist(self):
        assert set(SCENARIOS.keys()) == {"ideal_season", "variable_weather", "drought_year"}

    def test_ideal_season_config(self):
        s = SCENARIOS["ideal_season"]
        assert s.total_days == 60
        assert s.water_budget == float("inf")
        assert s.field_model.rain_probability == 0.35

    def test_drought_year_has_water_budget(self):
        s = SCENARIOS["drought_year"]
        assert s.water_budget == 800.0
        assert s.field_model.rain_probability == 0.12

    def test_starting_metrics_valid(self):
        for name, scenario in SCENARIOS.items():
            m = scenario.starting_metrics
            assert 0 <= m.crop_health <= 100
            assert 0 <= m.growth_rate <= 10
            assert 0 <= m.soil_health <= 100

    def test_growth_stage_progression(self):
        assert get_growth_stage(0, 90) == GrowthStage.GERMINATION
        assert get_growth_stage(20, 90) == GrowthStage.VEGETATIVE
        assert get_growth_stage(45, 90) == GrowthStage.FLOWERING
        assert get_growth_stage(60, 90) == GrowthStage.FRUITING
        assert get_growth_stage(80, 90) == GrowthStage.MATURITY

    def test_difficulty_ordering(self):
        """Harder scenarios have worse starting conditions."""
        easy = SCENARIOS["ideal_season"].starting_metrics
        hard = SCENARIOS["drought_year"].starting_metrics
        assert easy.crop_health > hard.crop_health
        assert easy.soil_health > hard.soil_health
        assert easy.water_stress < hard.water_stress


# ---------------------------------------------------------------------------
# Environment interface
# ---------------------------------------------------------------------------

class TestEnvInterface:
    def test_reset_returns_observation(self):
        env = CropEnv(seed=42)
        obs = env.reset("ideal_season")
        assert isinstance(obs, Observation)
        assert obs.day == 0
        assert obs.total_days == 60
        assert obs.scenario_name == "ideal_season"

    def test_step_returns_step_result(self):
        env = CropEnv(seed=42)
        env.reset("ideal_season")
        action = Action(
            irrigation=IrrigationLevel.MODERATE,
            fertilizer=FertilizerType.BALANCED,
            pest_management=PestManagement.NONE,
        )
        result = env.step(action)
        assert isinstance(result, StepResult)
        assert isinstance(result.observation, Observation)
        assert isinstance(result.reward, RewardBreakdown)
        assert isinstance(result.done, bool)

    def test_state_returns_env_state(self):
        env = CropEnv(seed=42)
        env.reset("ideal_season")
        state = env.state()
        assert isinstance(state, EnvState)
        assert state.day == 0
        assert state.total_days == 60

    def test_grade_returns_float(self):
        env = CropEnv(seed=42)
        env.reset("ideal_season")
        action = Action(
            irrigation=IrrigationLevel.MODERATE,
            fertilizer=FertilizerType.BALANCED,
            pest_management=PestManagement.NONE,
        )
        for _ in range(60):
            env.step(action)
        score = env.grade()
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_reset_clears_state(self):
        env = CropEnv(seed=42)
        env.reset("ideal_season")
        action = Action(
            irrigation=IrrigationLevel.MODERATE,
            fertilizer=FertilizerType.BALANCED,
            pest_management=PestManagement.NONE,
        )
        env.step(action)
        obs = env.reset("ideal_season")
        assert obs.day == 0
        assert env.state().cumulative_reward == 0.0

    def test_step_after_done_raises(self):
        env = CropEnv(seed=42)
        env.reset("ideal_season")
        action = Action(
            irrigation=IrrigationLevel.NONE,
            fertilizer=FertilizerType.NONE,
            pest_management=PestManagement.NONE,
        )
        for _ in range(60):
            env.step(action)
        with pytest.raises(RuntimeError):
            env.step(action)

    def test_step_without_reset_raises(self):
        env = CropEnv(seed=42)
        action = Action(
            irrigation=IrrigationLevel.NONE,
            fertilizer=FertilizerType.NONE,
            pest_management=PestManagement.NONE,
        )
        with pytest.raises(RuntimeError):
            env.step(action)

    def test_invalid_task_raises(self):
        env = CropEnv(seed=42)
        with pytest.raises(ValueError):
            env.reset("nonexistent_task")

    def test_all_tasks_run(self):
        env = CropEnv(seed=42)
        for task in ["ideal_season", "variable_weather", "drought_year"]:
            obs = env.reset(task)
            assert obs.day == 0
            action = Action(
                irrigation=IrrigationLevel.LIGHT,
                fertilizer=FertilizerType.NONE,
                pest_management=PestManagement.NONE,
            )
            result = env.step(action)
            assert result.observation.day == 1

    def test_observation_has_weather(self):
        env = CropEnv(seed=42)
        obs = env.reset("ideal_season")
        assert isinstance(obs.weather, WeatherObservation)
        assert obs.weather.temperature > 0

    def test_trends_appear_after_day_7(self):
        env = CropEnv(seed=42)
        env.reset("ideal_season")
        action = Action(
            irrigation=IrrigationLevel.MODERATE,
            fertilizer=FertilizerType.BALANCED,
            pest_management=PestManagement.NONE,
        )
        for i in range(10):
            result = env.step(action)
            if i < 6:
                assert result.observation.trends is None
            else:
                assert result.observation.trends is not None

    def test_water_budget_enforcement(self):
        """Drought year: water budget should be enforced."""
        env = CropEnv(seed=42)
        env.reset("drought_year")
        action = Action(
            irrigation=IrrigationLevel.HEAVY,  # 30mm per step
            fertilizer=FertilizerType.NONE,
            pest_management=PestManagement.NONE,
        )
        # 90 days × 30mm = 2700mm, but budget is 800mm
        for _ in range(90):
            result = env.step(action)
        assert env._water_used <= 800.0 + 30.0  # May overshoot by one step

    def test_day_increments(self):
        env = CropEnv(seed=42)
        env.reset("ideal_season")
        action = Action(
            irrigation=IrrigationLevel.NONE,
            fertilizer=FertilizerType.NONE,
            pest_management=PestManagement.NONE,
        )
        for expected_day in range(1, 11):
            result = env.step(action)
            assert result.observation.day == expected_day

    def test_done_at_end(self):
        env = CropEnv(seed=42)
        env.reset("ideal_season")
        action = Action(
            irrigation=IrrigationLevel.NONE,
            fertilizer=FertilizerType.NONE,
            pest_management=PestManagement.NONE,
        )
        for i in range(59):
            result = env.step(action)
            assert not result.done
        result = env.step(action)
        assert result.done


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_episode_ideal_season(self):
        env = CropEnv(seed=42)
        env.reset("ideal_season")
        action = Action(
            irrigation=IrrigationLevel.MODERATE,
            fertilizer=FertilizerType.BALANCED,
            pest_management=PestManagement.BIOLOGICAL,
        )
        for _ in range(60):
            result = env.step(action)
        score = env.grade()
        assert 0.0 <= score <= 1.0
        state = env.state()
        assert len(state.history) == 60

    def test_full_episode_variable_weather(self):
        env = CropEnv(seed=42)
        env.reset("variable_weather")
        action = Action(
            irrigation=IrrigationLevel.LIGHT,
            fertilizer=FertilizerType.ORGANIC,
            pest_management=PestManagement.SCOUTING,
        )
        for _ in range(90):
            result = env.step(action)
        score = env.grade()
        assert 0.0 <= score <= 1.0

    def test_full_episode_drought_year(self):
        env = CropEnv(seed=42)
        env.reset("drought_year")
        action = Action(
            irrigation=IrrigationLevel.LIGHT,
            fertilizer=FertilizerType.ORGANIC,
            pest_management=PestManagement.BIOLOGICAL,
        )
        for _ in range(90):
            result = env.step(action)
        score = env.grade()
        assert 0.0 <= score <= 1.0

    def test_deterministic_with_seed(self):
        """Same seed produces same results."""
        def run_episode(seed):
            env = CropEnv(seed=seed)
            env.reset("ideal_season")
            action = Action(
                irrigation=IrrigationLevel.MODERATE,
                fertilizer=FertilizerType.BALANCED,
                pest_management=PestManagement.NONE,
            )
            rewards = []
            for _ in range(60):
                result = env.step(action)
                rewards.append(result.reward.total)
            return rewards

        r1 = run_episode(42)
        r2 = run_episode(42)
        assert r1 == r2

    def test_different_seeds_differ(self):
        def run_episode(seed):
            env = CropEnv(seed=seed)
            env.reset("ideal_season")
            action = Action(
                irrigation=IrrigationLevel.MODERATE,
                fertilizer=FertilizerType.BALANCED,
                pest_management=PestManagement.NONE,
            )
            rewards = []
            for _ in range(60):
                result = env.step(action)
                rewards.append(result.reward.total)
            return rewards

        r1 = run_episode(42)
        r2 = run_episode(99)
        assert r1 != r2

    def test_reward_baseline_near_50(self):
        """Doing nothing should produce rewards near 50 (baseline)."""
        env = CropEnv(seed=42)
        env.reset("ideal_season")
        action = Action(
            irrigation=IrrigationLevel.NONE,
            fertilizer=FertilizerType.NONE,
            pest_management=PestManagement.NONE,
        )
        # First few steps: baseline is around 50
        result = env.step(action)
        # Reward should be in a reasonable range around 50
        assert 10 <= result.reward.total <= 90
