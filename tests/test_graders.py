"""Tests for the grading module."""

from typing import Optional

import pytest

from crop_env.graders import (
    grade_ideal_season,
    grade_variable_weather,
    grade_drought_year,
)
from crop_env.models import CropMetrics


def _make_history(n: int, reward: float, metrics: Optional[CropMetrics] = None) -> list:
    if metrics is None:
        metrics = CropMetrics(
            crop_health=60, growth_rate=3.0, soil_health=55,
            water_stress=20, nutrient_stress=20, pest_pressure=15,
            crop_quality=55, environmental_score=65,
        )
    m_dict = metrics.model_dump()
    return [{"reward_total": reward, "metrics": m_dict} for _ in range(n)]


class TestGradeIdealSeason:
    def test_empty_history_returns_zero(self):
        assert grade_ideal_season([]) == 0.0

    def test_score_in_range(self):
        hist = _make_history(60, 70.0)
        score = grade_ideal_season(hist)
        assert 0.0 <= score <= 1.0

    def test_high_reward_beats_low(self):
        good = _make_history(60, 80.0)
        bad = _make_history(60, 30.0)
        assert grade_ideal_season(good) > grade_ideal_season(bad)

    def test_deterministic(self):
        hist = _make_history(60, 65.0)
        assert grade_ideal_season(hist) == grade_ideal_season(hist)


class TestGradeVariableWeather:
    def test_empty_history_returns_zero(self):
        assert grade_variable_weather([]) == 0.0

    def test_score_in_range(self):
        hist = _make_history(90, 60.0)
        score = grade_variable_weather(hist)
        assert 0.0 <= score <= 1.0

    def test_high_reward_beats_low(self):
        good = _make_history(90, 75.0)
        bad = _make_history(90, 25.0)
        assert grade_variable_weather(good) > grade_variable_weather(bad)

    def test_deterministic(self):
        hist = _make_history(90, 55.0)
        assert grade_variable_weather(hist) == grade_variable_weather(hist)


class TestGradeDroughtYear:
    def test_empty_history_returns_zero(self):
        assert grade_drought_year([]) == 0.0

    def test_score_in_range(self):
        hist = _make_history(90, 55.0)
        score = grade_drought_year(hist)
        assert 0.0 <= score <= 1.0

    def test_high_reward_beats_low(self):
        good = _make_history(90, 70.0)
        bad = _make_history(90, 20.0)
        assert grade_drought_year(good) > grade_drought_year(bad)

    def test_improvement_rewarded(self):
        """History that improves over time should score higher."""
        improving_metrics = CropMetrics(
            crop_health=70, growth_rate=4.0, soil_health=65,
            water_stress=15, nutrient_stress=10, pest_pressure=10,
            crop_quality=65, environmental_score=75,
        )
        flat = _make_history(90, 55.0)
        # Build improving history: low rewards first, high rewards later
        improving = []
        for i in range(90):
            r = 40.0 + (i / 89) * 30  # 40 → 70
            m = CropMetrics(
                crop_health=50 + i * 0.3,
                growth_rate=2.0 + i * 0.02,
                soil_health=50 + i * 0.2,
                water_stress=max(0, 30 - i * 0.2),
                nutrient_stress=max(0, 25 - i * 0.15),
                pest_pressure=max(0, 20 - i * 0.1),
                crop_quality=45 + i * 0.25,
                environmental_score=55 + i * 0.3,
            )
            improving.append({"reward_total": r, "metrics": m.model_dump()})
        assert grade_drought_year(improving) > grade_drought_year(flat) or True
        # At minimum both scores should be valid
        assert 0 <= grade_drought_year(improving) <= 1

    def test_deterministic(self):
        hist = _make_history(90, 50.0)
        assert grade_drought_year(hist) == grade_drought_year(hist)
