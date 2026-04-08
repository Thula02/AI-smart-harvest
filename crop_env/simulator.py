"""Crop simulator — computes how farming actions change crop/field metrics.

This is the hidden response model.  Each scenario has different soil, weather,
and pest dynamics that govern how actions translate to metric changes.
The agent never sees the FieldModel parameters; it only observes the
resulting metric deltas and weather.

Key design: effects can be DELAYED.  Fertilizer applied today may not
show up in nutrient_stress reduction for 2-4 days.  Organic fertilizer
releases nutrients over 5-10 days.  Chemical pest resistance accumulates
over weeks.
"""

from __future__ import annotations

import random as stdlib_random
from typing import Any

from .models import (
    Action,
    CropMetrics,
    CropMetricDeltas,
    FertilizerType,
    GrowthStage,
    IrrigationLevel,
    IRRIGATION_MM,
    PestManagement,
    WeatherObservation,
)
from .scenarios import (
    FieldModel,
    ScenarioConfig,
    STAGE_GROWTH_MULTIPLIER,
    STAGE_NUTRIENT_SENSITIVITY,
    STAGE_WATER_SENSITIVITY,
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _get_pending_fertilizer_effect(
    history: list[dict[str, Any]],
    field_model: FieldModel,
    day: int,
) -> float:
    """Calculate delayed nutrient release from past fertilizer applications.

    Synthetic fertilizer: peak effect at day+2, fades by day+4.
    Organic fertilizer: slow release from day+3 through day+10.
    """
    total_release = 0.0
    for h in history:
        fert = h.get("action", {}).get("fertilizer", "none")
        applied_day = h.get("day", 0)
        days_ago = day - applied_day

        if fert == FertilizerType.ORGANIC.value:
            # Organic: slow release over days 3-10
            if 3 <= days_ago <= 10:
                release = field_model.organic_decomposition_rate * (1.0 - (days_ago - 3) / 8.0)
                total_release += max(0.0, release)
        elif fert in {
            FertilizerType.NITROGEN.value,
            FertilizerType.PHOSPHORUS.value,
            FertilizerType.BALANCED.value,
        }:
            # Synthetic: peak at day+2, gone by day+5
            if 2 <= days_ago <= 4:
                release = field_model.nutrient_absorption_rate * (1.0 - (days_ago - 2) / 3.0)
                total_release += max(0.0, release)

    return total_release


def _get_accumulated_chemical_resistance(history: list[dict[str, Any]], field_model: FieldModel) -> float:
    """Calculate accumulated pest resistance from past chemical applications."""
    total = field_model.pest_chemical_resistance
    for h in history:
        pest_mgmt = h.get("action", {}).get("pest_management", "none")
        if pest_mgmt in {PestManagement.CHEMICAL_LIGHT.value, PestManagement.CHEMICAL_HEAVY.value}:
            total += field_model.resistance_buildup_rate
    return min(0.95, total)  # Cap at 95% resistance


def compute_metric_changes(
    action: Action,
    current: CropMetrics,
    soil_moisture: float,
    weather: WeatherObservation,
    scenario: ScenarioConfig,
    growth_stage: GrowthStage,
    history: list[dict[str, Any]],
    day: int,
    rng: stdlib_random.Random,
) -> tuple[CropMetricDeltas, float]:
    """Compute the change in each crop metric for one day.

    Returns (deltas, new_soil_moisture).
    """
    fm = scenario.field_model
    water_sens = STAGE_WATER_SENSITIVITY[growth_stage]
    nutrient_sens = STAGE_NUTRIENT_SENSITIVITY[growth_stage]
    growth_mult = STAGE_GROWTH_MULTIPLIER[growth_stage]

    # --- Soil moisture dynamics ---
    irrigation_mm = IRRIGATION_MM[action.irrigation]
    moisture_input = irrigation_mm + weather.rainfall_mm
    # Evapotranspiration: higher temp = more water loss
    evapotranspiration = max(0.0, (weather.temperature - 10) * 0.15)
    # Retention: some moisture is retained, some drains
    new_moisture = soil_moisture + moisture_input * 0.3 - evapotranspiration
    new_moisture *= fm.moisture_retention + (1 - fm.moisture_retention) * 0.3
    new_moisture = _clamp(new_moisture, 0, 100)

    # --- Water stress ---
    d_water_stress = 0.0
    is_waterlogged = new_moisture > fm.waterlogging_threshold
    is_drought = new_moisture < 25.0

    if is_waterlogged:
        excess = (new_moisture - fm.waterlogging_threshold) / 15.0
        d_water_stress += excess * fm.crop_water_sensitivity * water_sens * 8.0
    elif is_drought:
        deficit = (25.0 - new_moisture) / 25.0
        d_water_stress += deficit * fm.crop_water_sensitivity * water_sens * 10.0
    else:
        # Good moisture range: stress decreases
        d_water_stress -= 3.0 * water_sens
    # Heat wave increases water stress
    if weather.is_extreme_event and weather.extreme_event_type == "heat_wave":
        d_water_stress += 5.0 * water_sens
    d_water_stress += rng.gauss(0, 1.0)

    # --- Nutrient stress ---
    d_nutrient_stress = 0.0
    # Delayed fertilizer effects
    pending_nutrients = _get_pending_fertilizer_effect(history, fm, day)
    # Today's direct fertilizer (minor immediate effect for synthetic)
    today_nutrient = 0.0
    if action.fertilizer == FertilizerType.NITROGEN:
        today_nutrient = 2.0
    elif action.fertilizer == FertilizerType.PHOSPHORUS:
        today_nutrient = 1.5
    elif action.fertilizer == FertilizerType.BALANCED:
        today_nutrient = 2.5
    elif action.fertilizer == FertilizerType.ORGANIC:
        today_nutrient = 0.5  # Minimal immediate effect

    total_nutrient_input = today_nutrient + pending_nutrients * 10.0

    # Check for nutrient burn (too much fertilizer)
    nutrient_level = 100 - current.nutrient_stress + total_nutrient_input
    if nutrient_level > fm.nutrient_burn_threshold:
        burn = (nutrient_level - fm.nutrient_burn_threshold) / 20.0
        d_nutrient_stress += burn * fm.crop_nutrient_sensitivity * nutrient_sens * 5.0
    else:
        # Normal nutrient supply reduces stress
        d_nutrient_stress -= total_nutrient_input * nutrient_sens * 0.8

    # Natural nutrient depletion
    d_nutrient_stress += 0.5 * nutrient_sens
    # Low soil moisture reduces nutrient availability
    if new_moisture < 20:
        d_nutrient_stress += 2.0 * nutrient_sens
    d_nutrient_stress += rng.gauss(0, 0.8)

    # --- Pest pressure ---
    d_pest = 0.0
    # Natural pest growth (accelerates in warm/wet conditions)
    temp_factor = max(0.5, (weather.temperature - 15) / 15.0)
    moisture_pest_factor = 1.0 + (new_moisture / 100.0) * 0.5
    daily_pest_growth = fm.pest_growth_rate * temp_factor * moisture_pest_factor * 0.1
    d_pest += daily_pest_growth

    # Variable weather: pest surge mid-season
    if scenario.name == "variable_weather" and day > scenario.total_days * 0.3:
        d_pest += 0.3

    # Pest management actions
    resistance = _get_accumulated_chemical_resistance(history, fm)

    if action.pest_management == PestManagement.BIOLOGICAL:
        # Biological: moderate effect, better with healthy soil
        soil_bonus = current.soil_health / 100.0
        d_pest -= 3.0 * (0.5 + soil_bonus * 0.5)
    elif action.pest_management == PestManagement.CHEMICAL_LIGHT:
        effectiveness = max(0.1, 1.0 - resistance)
        d_pest -= 5.0 * effectiveness
        # Chemical washes off in rain
        if weather.rainfall_mm > 5.0:
            d_pest += 2.0 * effectiveness  # Reduced kill due to runoff
    elif action.pest_management == PestManagement.CHEMICAL_HEAVY:
        effectiveness = max(0.1, 1.0 - resistance)
        d_pest -= 8.0 * effectiveness
        if weather.rainfall_mm > 5.0:
            d_pest += 3.0 * effectiveness
    elif action.pest_management == PestManagement.SCOUTING:
        # Scouting: no direct pest reduction, small benefit from awareness
        d_pest -= 0.3

    d_pest += rng.gauss(0, 0.5)

    # --- Soil health ---
    d_soil = 0.0
    # Organic fertilizer improves soil health over time
    if action.fertilizer == FertilizerType.ORGANIC:
        d_soil += 1.5
    elif action.fertilizer in {FertilizerType.NITROGEN, FertilizerType.PHOSPHORUS}:
        d_soil -= 0.3  # Synthetic fertilizer slightly degrades soil
    # Chemical pesticides degrade soil
    if action.pest_management == PestManagement.CHEMICAL_LIGHT:
        d_soil -= 0.5
    elif action.pest_management == PestManagement.CHEMICAL_HEAVY:
        d_soil -= 1.2
    # Biological pest control is soil-friendly
    if action.pest_management == PestManagement.BIOLOGICAL:
        d_soil += 0.3
    # Waterlogging degrades soil structure
    if is_waterlogged:
        d_soil -= 1.0
    # Natural soil recovery (slow)
    if current.soil_health < 60:
        d_soil += 0.2
    d_soil += rng.gauss(0, 0.3)

    # --- Growth rate ---
    d_growth = 0.0
    # Base growth depends on stage
    ideal_growth = fm.base_growth_rate * growth_mult
    # Stress penalties
    stress_penalty = 1.0
    effective_water_stress = _clamp(current.water_stress + d_water_stress, 0, 100)
    effective_nutrient_stress = _clamp(current.nutrient_stress + d_nutrient_stress, 0, 100)
    effective_pest = _clamp(current.pest_pressure + d_pest, 0, 100)

    stress_penalty *= max(0.1, 1.0 - effective_water_stress / 120.0)
    stress_penalty *= max(0.1, 1.0 - effective_nutrient_stress / 120.0)
    stress_penalty *= max(0.3, 1.0 - effective_pest / 150.0)

    target_growth = ideal_growth * stress_penalty
    d_growth = (target_growth - current.growth_rate) * 0.3  # Smooth transition
    d_growth += rng.gauss(0, 0.1)

    # --- Crop health ---
    d_health = 0.0
    # Stress reduces health
    total_stress = (effective_water_stress + effective_nutrient_stress + effective_pest) / 3.0
    if total_stress > 50:
        d_health -= (total_stress - 50) * 0.08
    elif total_stress < 30:
        d_health += (30 - total_stress) * 0.05  # Recovery when stress is low
    # Storm damage
    if weather.is_extreme_event and weather.extreme_event_type == "storm":
        d_health -= rng.uniform(3.0, 8.0)
    # Heat wave damage
    if weather.is_extreme_event and weather.extreme_event_type == "heat_wave":
        d_health -= 2.0
    d_health += rng.gauss(0, 0.5)

    # --- Crop quality ---
    d_quality = 0.0
    # Quality tracks health trends
    if current.crop_health > 70:
        d_quality += 0.5
    elif current.crop_health < 40:
        d_quality -= 1.0
    # Balanced nutrition improves quality
    if action.fertilizer == FertilizerType.BALANCED:
        d_quality += 0.3
    elif action.fertilizer == FertilizerType.ORGANIC:
        d_quality += 0.5
    # High pest pressure ruins quality
    if effective_pest > 50:
        d_quality -= (effective_pest - 50) * 0.03
    # Stress during flowering/fruiting is especially bad for quality
    if growth_stage in {GrowthStage.FLOWERING, GrowthStage.FRUITING}:
        if total_stress > 40:
            d_quality -= (total_stress - 40) * 0.04
    d_quality += rng.gauss(0, 0.3)

    # --- Environmental score ---
    d_env = 0.0
    # Chemical pesticides hurt environmental score
    if action.pest_management == PestManagement.CHEMICAL_LIGHT:
        d_env -= 2.0
    elif action.pest_management == PestManagement.CHEMICAL_HEAVY:
        d_env -= 4.0
        # Even worse in rain (runoff)
        if weather.rainfall_mm > 5.0:
            d_env -= 2.0
    # Organic/biological practices improve it
    if action.pest_management == PestManagement.BIOLOGICAL:
        d_env += 1.0
    if action.fertilizer == FertilizerType.ORGANIC:
        d_env += 0.8
    # Scouting is environmentally neutral-positive
    if action.pest_management == PestManagement.SCOUTING:
        d_env += 0.3
    # Over-irrigation wastes water
    if action.irrigation == IrrigationLevel.HEAVY and new_moisture > 70:
        d_env -= 1.0
    # Natural environmental recovery
    if current.environmental_score < 70:
        d_env += 0.2
    d_env += rng.gauss(0, 0.2)

    deltas = CropMetricDeltas(
        crop_health=round(d_health, 3),
        growth_rate=round(d_growth, 3),
        soil_health=round(d_soil, 3),
        water_stress=round(d_water_stress, 3),
        nutrient_stress=round(d_nutrient_stress, 3),
        pest_pressure=round(d_pest, 3),
        crop_quality=round(d_quality, 3),
        environmental_score=round(d_env, 3),
    )

    return deltas, round(new_moisture, 2)


def apply_deltas(current: CropMetrics, deltas: CropMetricDeltas) -> CropMetrics:
    """Apply deltas to current metrics, clamping to valid ranges."""
    return CropMetrics(
        crop_health=_clamp(current.crop_health + deltas.crop_health, 0, 100),
        growth_rate=_clamp(current.growth_rate + deltas.growth_rate, 0, 10),
        soil_health=_clamp(current.soil_health + deltas.soil_health, 0, 100),
        water_stress=_clamp(current.water_stress + deltas.water_stress, 0, 100),
        nutrient_stress=_clamp(current.nutrient_stress + deltas.nutrient_stress, 0, 100),
        pest_pressure=_clamp(current.pest_pressure + deltas.pest_pressure, 0, 100),
        crop_quality=_clamp(current.crop_quality + deltas.crop_quality, 0, 100),
        environmental_score=_clamp(current.environmental_score + deltas.environmental_score, 0, 100),
    )
