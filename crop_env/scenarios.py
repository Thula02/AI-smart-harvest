"""Scenario definitions with hidden field/soil/weather response models.

Each scenario has a unique FieldModel — the mapping from actions to
crop metric changes is governed by hidden soil/weather parameters.
The agent never sees these parameters; it only sees the resulting
metric deltas and weather observations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .models import (
    CropMetrics,
    GrowthStage,
    WeatherObservation,
)


@dataclass
class FieldModel:
    """Hidden parameters controlling how a field responds to actions.

    These are NOT visible to the agent.  They define this scenario's
    unique soil, crop, and pest dynamics.
    """

    # ----- Soil properties -----
    # How well soil retains moisture (0-1, higher = retains more)
    moisture_retention: float = 0.7
    # How quickly crops absorb applied nutrients (0-1)
    nutrient_absorption_rate: float = 0.5
    # How fast organic matter decomposes into available nutrients (0-1)
    organic_decomposition_rate: float = 0.3
    # Soil moisture % above which roots are damaged (waterlogging)
    waterlogging_threshold: float = 85.0
    # Nitrogen level above which crop suffers nutrient burn
    nutrient_burn_threshold: float = 80.0

    # ----- Crop response -----
    # How much water stress affects growth (higher = more sensitive)
    crop_water_sensitivity: float = 1.0
    # How much nutrient stress affects growth
    crop_nutrient_sensitivity: float = 1.0
    # Growth rate under ideal conditions (cm/day)
    base_growth_rate: float = 3.0

    # ----- Pest dynamics -----
    # Base daily pest population growth rate (% per day)
    pest_growth_rate: float = 2.0
    # Baseline pest resistance to chemical pesticides (0-1, higher = more resistant)
    pest_chemical_resistance: float = 0.1
    # How fast chemical resistance builds per application (additive)
    resistance_buildup_rate: float = 0.02

    # ----- Weather -----
    # Daily probability of rain
    rain_probability: float = 0.3
    # Average rainfall when it does rain (mm)
    rain_intensity_mean: float = 10.0
    # Average daily temperature (°C)
    temperature_mean: float = 25.0


@dataclass
class ScenarioConfig:
    """Configuration for a scenario (task)."""
    name: str
    field_model: FieldModel
    starting_metrics: CropMetrics
    starting_soil_moisture: float
    total_days: int
    water_budget: float  # Total mm of irrigation allowed (inf = no limit)
    description: str


# ---------------------------------------------------------------------------
# Growth stage transitions by day fraction of episode
# ---------------------------------------------------------------------------

def get_growth_stage(day: int, total_days: int) -> GrowthStage:
    """Determine growth stage based on progress through the episode."""
    frac = day / total_days
    if frac < 0.10:
        return GrowthStage.GERMINATION
    elif frac < 0.35:
        return GrowthStage.VEGETATIVE
    elif frac < 0.55:
        return GrowthStage.FLOWERING
    elif frac < 0.80:
        return GrowthStage.FRUITING
    else:
        return GrowthStage.MATURITY


# ---------------------------------------------------------------------------
# Growth stage sensitivity multipliers
# ---------------------------------------------------------------------------

# How sensitive the crop is to water/nutrient stress at each stage
STAGE_WATER_SENSITIVITY: dict[GrowthStage, float] = {
    GrowthStage.GERMINATION: 1.5,   # Seedlings are fragile
    GrowthStage.VEGETATIVE: 1.0,    # Normal
    GrowthStage.FLOWERING: 1.8,     # Flowering very sensitive to drought
    GrowthStage.FRUITING: 1.3,      # Moderate sensitivity
    GrowthStage.MATURITY: 0.6,      # Less water needed at maturity
}

STAGE_NUTRIENT_SENSITIVITY: dict[GrowthStage, float] = {
    GrowthStage.GERMINATION: 0.5,   # Low needs early
    GrowthStage.VEGETATIVE: 1.5,    # High nutrient demand for leaf growth
    GrowthStage.FLOWERING: 1.2,     # Moderate-high
    GrowthStage.FRUITING: 1.0,      # Normal
    GrowthStage.MATURITY: 0.4,      # Minimal needs
}

STAGE_GROWTH_MULTIPLIER: dict[GrowthStage, float] = {
    GrowthStage.GERMINATION: 0.3,   # Slow initial growth
    GrowthStage.VEGETATIVE: 1.5,    # Peak growth phase
    GrowthStage.FLOWERING: 0.8,     # Growth slows during flowering
    GrowthStage.FRUITING: 0.5,      # Growth redirected to fruit
    GrowthStage.MATURITY: 0.1,      # Minimal growth
}


# ---------------------------------------------------------------------------
# Weather generation
# ---------------------------------------------------------------------------

def generate_daily_weather(
    field_model: FieldModel,
    day: int,
    total_days: int,
    rng: random.Random,
    active_extreme: dict | None = None,
) -> tuple[WeatherObservation, dict | None]:
    """Generate today's weather conditions.

    Returns (weather_observation, active_extreme_event_state).
    """
    temp = rng.gauss(field_model.temperature_mean, 3.0)

    # Check for new extreme events (5% chance)
    if active_extreme is None and rng.random() < 0.05:
        if rng.random() < 0.5:
            # Heat wave: 3-5 days, high temps
            active_extreme = {
                "type": "heat_wave",
                "days_remaining": rng.randint(3, 5),
                "temp_boost": rng.uniform(8, 15),
            }
        else:
            # Storm: 1-2 days, heavy rain + possible crop damage
            active_extreme = {
                "type": "storm",
                "days_remaining": rng.randint(1, 2),
                "rain_mult": rng.uniform(3.0, 6.0),
            }

    is_extreme = active_extreme is not None
    extreme_type = active_extreme["type"] if active_extreme else None

    # Apply extreme event effects
    if active_extreme:
        if active_extreme["type"] == "heat_wave":
            temp += active_extreme["temp_boost"]
        active_extreme["days_remaining"] -= 1
        if active_extreme["days_remaining"] <= 0:
            active_extreme = None

    # Rainfall
    rainfall = 0.0
    if rng.random() < field_model.rain_probability:
        rainfall = max(0.0, rng.gauss(field_model.rain_intensity_mean, 4.0))
        if extreme_type == "storm" and is_extreme:
            # Re-read from the original active_extreme before we potentially set it to None
            rainfall *= 4.0  # storms bring heavy rain

    weather = WeatherObservation(
        temperature=round(temp, 1),
        rainfall_mm=round(rainfall, 1),
        is_extreme_event=is_extreme,
        extreme_event_type=extreme_type,
    )

    return weather, active_extreme


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, ScenarioConfig] = {
    "ideal_season": ScenarioConfig(
        name="ideal_season",
        field_model=FieldModel(
            moisture_retention=0.75,
            nutrient_absorption_rate=0.6,
            organic_decomposition_rate=0.35,
            waterlogging_threshold=90.0,
            nutrient_burn_threshold=85.0,
            crop_water_sensitivity=0.8,
            crop_nutrient_sensitivity=0.8,
            base_growth_rate=3.5,
            pest_growth_rate=1.5,
            pest_chemical_resistance=0.05,
            resistance_buildup_rate=0.01,
            rain_probability=0.35,
            rain_intensity_mean=10.0,
            temperature_mean=24.0,
        ),
        starting_metrics=CropMetrics(
            crop_health=70,
            growth_rate=2.0,
            soil_health=75,
            water_stress=15,
            nutrient_stress=20,
            pest_pressure=10,
            crop_quality=65,
            environmental_score=80,
        ),
        starting_soil_moisture=55.0,
        total_days=60,
        water_budget=float("inf"),
        description="Easy: Fertile soil, regular rainfall, low pest risk over 60 days",
    ),
    "variable_weather": ScenarioConfig(
        name="variable_weather",
        field_model=FieldModel(
            moisture_retention=0.60,
            nutrient_absorption_rate=0.45,
            organic_decomposition_rate=0.30,
            waterlogging_threshold=82.0,
            nutrient_burn_threshold=75.0,
            crop_water_sensitivity=1.0,
            crop_nutrient_sensitivity=1.0,
            base_growth_rate=3.0,
            pest_growth_rate=2.5,
            pest_chemical_resistance=0.15,
            resistance_buildup_rate=0.025,
            rain_probability=0.25,
            rain_intensity_mean=12.0,
            temperature_mean=26.0,
        ),
        starting_metrics=CropMetrics(
            crop_health=60,
            growth_rate=1.5,
            soil_health=60,
            water_stress=30,
            nutrient_stress=35,
            pest_pressure=25,
            crop_quality=55,
            environmental_score=75,
        ),
        starting_soil_moisture=45.0,
        total_days=90,
        water_budget=float("inf"),
        description="Medium: Variable weather with alternating wet/dry spells and rising pest pressure",
    ),
    "drought_year": ScenarioConfig(
        name="drought_year",
        field_model=FieldModel(
            moisture_retention=0.50,
            nutrient_absorption_rate=0.35,
            organic_decomposition_rate=0.25,
            waterlogging_threshold=75.0,
            nutrient_burn_threshold=65.0,
            crop_water_sensitivity=1.5,
            crop_nutrient_sensitivity=1.3,
            base_growth_rate=2.0,
            pest_growth_rate=2.5,
            pest_chemical_resistance=0.20,
            resistance_buildup_rate=0.035,
            rain_probability=0.12,
            rain_intensity_mean=6.0,
            temperature_mean=30.0,
        ),
        starting_metrics=CropMetrics(
            crop_health=50,
            growth_rate=1.0,
            soil_health=45,
            water_stress=45,
            nutrient_stress=40,
            pest_pressure=35,
            crop_quality=45,
            environmental_score=70,
        ),
        starting_soil_moisture=30.0,
        total_days=90,
        water_budget=800.0,  # Limited water budget (mm total)
        description="Hard: Drought conditions, depleted soil, high pest risk, limited water budget",
    ),
}
