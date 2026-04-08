"""Tests for the payoff / reward computation module."""

import pytest

from crop_env.models import CropMetricDeltas
from crop_env.payoff import (
    DELTA_SCALES,
    LOWER_IS_BETTER,
    TASK_WEIGHTS,
    compute_reward,
)


class TestTaskWeights:
    def test_all_tasks_present(self):
        assert set(TASK_WEIGHTS.keys()) == {"ideal_season", "variable_weather", "drought_year"}

    def test_weights_sum_to_one(self):
        for task, weights in TASK_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-6, f"{task} weights sum to {total}"

    def test_weights_are_positive(self):
        for task, weights in TASK_WEIGHTS.items():
            for metric, w in weights.items():
                assert w > 0, f"{task}/{metric} has non-positive weight {w}"

    def test_all_metrics_covered(self):
        expected = {
            "crop_health", "growth_rate", "soil_health",
            "water_stress", "nutrient_stress", "pest_pressure",
            "crop_quality", "environmental_score",
        }
        for task, weights in TASK_WEIGHTS.items():
            assert set(weights.keys()) == expected, f"{task} missing metrics"


class TestDeltaScales:
    def test_all_metrics_have_scales(self):
        expected = {
            "crop_health", "growth_rate", "soil_health",
            "water_stress", "nutrient_stress", "pest_pressure",
            "crop_quality", "environmental_score",
        }
        assert set(DELTA_SCALES.keys()) == expected

    def test_scales_are_positive(self):
        for metric, scale in DELTA_SCALES.items():
            assert scale > 0, f"{metric} has non-positive scale {scale}"


class TestLowerIsBetter:
    def test_stress_metrics_are_lower_is_better(self):
        assert "water_stress" in LOWER_IS_BETTER
        assert "nutrient_stress" in LOWER_IS_BETTER
        assert "pest_pressure" in LOWER_IS_BETTER

    def test_positive_metrics_not_in_set(self):
        assert "crop_health" not in LOWER_IS_BETTER
        assert "growth_rate" not in LOWER_IS_BETTER
        assert "crop_quality" not in LOWER_IS_BETTER


class TestComputeReward:
    def test_zero_deltas_give_baseline(self):
        deltas = CropMetricDeltas(
            crop_health=0, growth_rate=0, soil_health=0,
            water_stress=0, nutrient_stress=0, pest_pressure=0,
            crop_quality=0, environmental_score=0,
        )
        rb = compute_reward(deltas, "ideal_season")
        assert abs(rb.total - 50.0) < 1e-6

    def test_positive_improvement_above_baseline(self):
        deltas = CropMetricDeltas(
            crop_health=5, growth_rate=0.5, soil_health=3,
            water_stress=-3, nutrient_stress=-2, pest_pressure=-2,
            crop_quality=3, environmental_score=3,
        )
        rb = compute_reward(deltas, "ideal_season")
        assert rb.total > 50.0

    def test_negative_changes_below_baseline(self):
        deltas = CropMetricDeltas(
            crop_health=-5, growth_rate=-0.5, soil_health=-3,
            water_stress=5, nutrient_stress=5, pest_pressure=5,
            crop_quality=-3, environmental_score=-3,
        )
        rb = compute_reward(deltas, "ideal_season")
        assert rb.total < 50.0

    def test_reward_clamped_0_100(self):
        huge = CropMetricDeltas(
            crop_health=100, growth_rate=10, soil_health=100,
            water_stress=-100, nutrient_stress=-100, pest_pressure=-100,
            crop_quality=100, environmental_score=100,
        )
        rb = compute_reward(huge, "ideal_season")
        assert rb.total <= 100.0

        terrible = CropMetricDeltas(
            crop_health=-100, growth_rate=-10, soil_health=-100,
            water_stress=100, nutrient_stress=100, pest_pressure=100,
            crop_quality=-100, environmental_score=-100,
        )
        rb = compute_reward(terrible, "ideal_season")
        assert rb.total >= 0.0

    def test_lower_is_better_flipping(self):
        """Decreasing water_stress should contribute positively."""
        deltas_good = CropMetricDeltas(
            crop_health=0, growth_rate=0, soil_health=0,
            water_stress=-5, nutrient_stress=0, pest_pressure=0,
            crop_quality=0, environmental_score=0,
        )
        deltas_bad = CropMetricDeltas(
            crop_health=0, growth_rate=0, soil_health=0,
            water_stress=5, nutrient_stress=0, pest_pressure=0,
            crop_quality=0, environmental_score=0,
        )
        rb_good = compute_reward(deltas_good, "ideal_season")
        rb_bad = compute_reward(deltas_bad, "ideal_season")
        assert rb_good.total > rb_bad.total

    def test_reward_has_all_components(self):
        deltas = CropMetricDeltas(
            crop_health=1, growth_rate=0.1, soil_health=1,
            water_stress=-1, nutrient_stress=-1, pest_pressure=-1,
            crop_quality=1, environmental_score=1,
        )
        rb = compute_reward(deltas, "ideal_season")
        for field in [
            "crop_health_reward", "growth_rate_reward", "soil_health_reward",
            "water_stress_reward", "nutrient_stress_reward", "pest_pressure_reward",
            "crop_quality_reward", "environmental_score_reward",
        ]:
            assert hasattr(rb, field)

    def test_all_tasks_produce_reward(self):
        deltas = CropMetricDeltas(
            crop_health=2, growth_rate=0.2, soil_health=1,
            water_stress=-1, nutrient_stress=-1, pest_pressure=-1,
            crop_quality=1, environmental_score=1,
        )
        for task in ["ideal_season", "variable_weather", "drought_year"]:
            rb = compute_reward(deltas, task)
            assert isinstance(rb.total, float)
            assert 0 <= rb.total <= 100
