"""Tests for the simulator module — transition dynamics, delayed effects, etc."""

import random

import pytest

from crop_env.models import (
    Action,
    CropMetricDeltas,
    CropMetrics,
    FertilizerType,
    GrowthStage,
    IrrigationLevel,
    PestManagement,
    WeatherObservation,
)
from crop_env.scenarios import SCENARIOS, generate_daily_weather
from crop_env.simulator import (
    apply_deltas,
    compute_metric_changes,
    _get_accumulated_chemical_resistance,
    _get_pending_fertilizer_effect,
)


def _default_metrics() -> CropMetrics:
    return CropMetrics(
        crop_health=60, growth_rate=2.0, soil_health=55,
        water_stress=30, nutrient_stress=25, pest_pressure=15,
        crop_quality=50, environmental_score=60,
    )


def _default_weather() -> WeatherObservation:
    return WeatherObservation(temperature=25.0, rainfall_mm=0.0)


def _make_action(
    irr=IrrigationLevel.MODERATE,
    fert=FertilizerType.NONE,
    pest=PestManagement.NONE,
) -> Action:
    return Action(irrigation=irr, fertilizer=fert, pest_management=pest)


class TestComputeMetricChanges:
    def test_returns_deltas_and_moisture(self):
        sc = SCENARIOS["ideal_season"]
        rng = random.Random(42)
        deltas, moisture = compute_metric_changes(
            action=_make_action(),
            current=_default_metrics(),
            soil_moisture=50.0,
            weather=_default_weather(),
            scenario=sc,
            growth_stage=GrowthStage.VEGETATIVE,
            history=[],
            day=5,
            rng=rng,
        )
        assert isinstance(deltas, CropMetricDeltas)
        assert isinstance(moisture, float)
        assert moisture >= 0

    def test_irrigation_increases_moisture(self):
        sc = SCENARIOS["ideal_season"]
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        _, moist_none = compute_metric_changes(
            action=_make_action(irr=IrrigationLevel.NONE),
            current=_default_metrics(),
            soil_moisture=30.0,
            weather=_default_weather(),
            scenario=sc,
            growth_stage=GrowthStage.VEGETATIVE,
            history=[], day=5, rng=rng1,
        )
        _, moist_heavy = compute_metric_changes(
            action=_make_action(irr=IrrigationLevel.HEAVY),
            current=_default_metrics(),
            soil_moisture=30.0,
            weather=_default_weather(),
            scenario=sc,
            growth_stage=GrowthStage.VEGETATIVE,
            history=[], day=5, rng=rng2,
        )
        assert moist_heavy > moist_none

    def test_rain_increases_moisture(self):
        sc = SCENARIOS["ideal_season"]
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        dry_weather = WeatherObservation(temperature=25.0, rainfall_mm=0.0)
        wet_weather = WeatherObservation(temperature=25.0, rainfall_mm=20.0)
        _, moist_dry = compute_metric_changes(
            action=_make_action(irr=IrrigationLevel.NONE),
            current=_default_metrics(),
            soil_moisture=30.0,
            weather=dry_weather,
            scenario=sc,
            growth_stage=GrowthStage.VEGETATIVE,
            history=[], day=5, rng=rng1,
        )
        _, moist_wet = compute_metric_changes(
            action=_make_action(irr=IrrigationLevel.NONE),
            current=_default_metrics(),
            soil_moisture=30.0,
            weather=wet_weather,
            scenario=sc,
            growth_stage=GrowthStage.VEGETATIVE,
            history=[], day=5, rng=rng2,
        )
        assert moist_wet > moist_dry

    def test_chemical_pest_reduces_pressure(self):
        sc = SCENARIOS["ideal_season"]
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        metrics = _default_metrics()
        metrics = metrics.model_copy(update={"pest_pressure": 40})

        d_no, _ = compute_metric_changes(
            action=_make_action(pest=PestManagement.NONE),
            current=metrics, soil_moisture=50.0,
            weather=_default_weather(), scenario=sc,
            growth_stage=GrowthStage.VEGETATIVE,
            history=[], day=5, rng=rng1,
        )
        d_chem, _ = compute_metric_changes(
            action=_make_action(pest=PestManagement.CHEMICAL_HEAVY),
            current=metrics, soil_moisture=50.0,
            weather=_default_weather(), scenario=sc,
            growth_stage=GrowthStage.VEGETATIVE,
            history=[], day=5, rng=rng2,
        )
        assert d_chem.pest_pressure < d_no.pest_pressure

    def test_chemical_pest_hurts_environment(self):
        sc = SCENARIOS["ideal_season"]
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        d_no, _ = compute_metric_changes(
            action=_make_action(pest=PestManagement.NONE),
            current=_default_metrics(), soil_moisture=50.0,
            weather=_default_weather(), scenario=sc,
            growth_stage=GrowthStage.VEGETATIVE,
            history=[], day=5, rng=rng1,
        )
        d_chem, _ = compute_metric_changes(
            action=_make_action(pest=PestManagement.CHEMICAL_HEAVY),
            current=_default_metrics(), soil_moisture=50.0,
            weather=_default_weather(), scenario=sc,
            growth_stage=GrowthStage.VEGETATIVE,
            history=[], day=5, rng=rng2,
        )
        assert d_chem.environmental_score < d_no.environmental_score


class TestApplyDeltas:
    def test_basic_apply(self):
        m = _default_metrics()
        d = CropMetricDeltas(
            crop_health=5, growth_rate=0.5, soil_health=3,
            water_stress=-5, nutrient_stress=-3, pest_pressure=-2,
            crop_quality=4, environmental_score=2,
        )
        new_m = apply_deltas(m, d)
        assert new_m.crop_health == 65
        assert new_m.water_stress == 25

    def test_clamping_upper(self):
        m = CropMetrics(
            crop_health=98, growth_rate=9.5, soil_health=99,
            water_stress=5, nutrient_stress=5, pest_pressure=5,
            crop_quality=98, environmental_score=99,
        )
        d = CropMetricDeltas(
            crop_health=10, growth_rate=5, soil_health=10,
            water_stress=0, nutrient_stress=0, pest_pressure=0,
            crop_quality=10, environmental_score=10,
        )
        new_m = apply_deltas(m, d)
        assert new_m.crop_health == 100
        assert new_m.growth_rate == 10.0
        assert new_m.soil_health == 100

    def test_clamping_lower(self):
        m = CropMetrics(
            crop_health=3, growth_rate=0.2, soil_health=2,
            water_stress=95, nutrient_stress=95, pest_pressure=95,
            crop_quality=2, environmental_score=3,
        )
        d = CropMetricDeltas(
            crop_health=-10, growth_rate=-5, soil_health=-10,
            water_stress=0, nutrient_stress=0, pest_pressure=0,
            crop_quality=-10, environmental_score=-10,
        )
        new_m = apply_deltas(m, d)
        assert new_m.crop_health == 0
        assert new_m.growth_rate == 0
        assert new_m.soil_health == 0


class TestDelayedFertilizer:
    def test_no_fertilizer_no_pending(self):
        history = [
            {"day": d, "action": _make_action(fert=FertilizerType.NONE).model_dump()}
            for d in range(5)
        ]
        sc = SCENARIOS["ideal_season"]
        effect = _get_pending_fertilizer_effect(history, sc.field_model, 5)
        assert effect == 0.0

    def test_synthetic_fertilizer_peaks_later(self):
        """Synthetic fertilizer should produce non-zero effect after 2+ days."""
        history = [
            {"day": 0, "action": _make_action(fert=FertilizerType.NITROGEN).model_dump()},
        ]
        sc = SCENARIOS["ideal_season"]
        e1 = _get_pending_fertilizer_effect(history, sc.field_model, 1)
        e3 = _get_pending_fertilizer_effect(history, sc.field_model, 3)
        assert e3 > e1

    def test_organic_fertilizer_slow_release(self):
        """Organic fertilizer should still have effect after 5+ days."""
        history = [
            {"day": 0, "action": _make_action(fert=FertilizerType.ORGANIC).model_dump()},
        ]
        sc = SCENARIOS["ideal_season"]
        e5 = _get_pending_fertilizer_effect(history, sc.field_model, 5)
        e8 = _get_pending_fertilizer_effect(history, sc.field_model, 8)
        assert e5 > 0
        assert e8 > 0


class TestChemicalResistance:
    def test_no_chemicals_no_resistance(self):
        history = [
            {"day": d, "action": _make_action(pest=PestManagement.NONE).model_dump()}
            for d in range(10)
        ]
        sc = SCENARIOS["ideal_season"]
        res = _get_accumulated_chemical_resistance(history, sc.field_model)
        assert res == sc.field_model.pest_chemical_resistance  # baseline only

    def test_chemicals_increase_resistance(self):
        history = [
            {"day": d, "action": _make_action(pest=PestManagement.CHEMICAL_HEAVY).model_dump()}
            for d in range(10)
        ]
        sc = SCENARIOS["ideal_season"]
        baseline = sc.field_model.pest_chemical_resistance
        res = _get_accumulated_chemical_resistance(history, sc.field_model)
        assert res > baseline

    def test_resistance_caps_at_095(self):
        history = [
            {"day": d, "action": _make_action(pest=PestManagement.CHEMICAL_HEAVY).model_dump()}
            for d in range(200)
        ]
        sc = SCENARIOS["ideal_season"]
        res = _get_accumulated_chemical_resistance(history, sc.field_model)
        assert res <= 0.95


class TestWeatherGeneration:
    def test_weather_is_valid(self):
        rng = random.Random(42)
        sc = SCENARIOS["ideal_season"]
        weather, _ = generate_daily_weather(sc.field_model, 0, sc.total_days, rng, None)
        assert weather.temperature > 0
        assert weather.rainfall_mm >= 0

    def test_weather_deterministic(self):
        sc = SCENARIOS["ideal_season"]
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        w1, _ = generate_daily_weather(sc.field_model, 0, sc.total_days, rng1, None)
        w2, _ = generate_daily_weather(sc.field_model, 0, sc.total_days, rng2, None)
        assert w1.temperature == w2.temperature
        assert w1.rainfall_mm == w2.rainfall_mm
