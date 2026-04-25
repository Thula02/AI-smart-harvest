"""Pydantic models for the Outcome-Based Crop Management Simulator."""

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class IrrigationLevel(str, Enum):
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"


class FertilizerType(str, Enum):
    NONE = "none"
    NITROGEN = "nitrogen"
    PHOSPHORUS = "phosphorus"
    BALANCED = "balanced"
    ORGANIC = "organic"


class PestManagement(str, Enum):
    NONE = "none"
    SCOUTING = "scouting"
    BIOLOGICAL = "biological"
    CHEMICAL_LIGHT = "chemical_light"
    CHEMICAL_HEAVY = "chemical_heavy"


class GrowthStage(str, Enum):
    GERMINATION = "germination"
    VEGETATIVE = "vegetative"
    FLOWERING = "flowering"
    FRUITING = "fruiting"
    MATURITY = "maturity"


class ToolCallType(str, Enum):
    SOIL_TEST = "soil_test"
    WEATHER_FORECAST = "weather_forecast"
    MARKET_PRICES = "market_prices"


# ---------------------------------------------------------------------------
# Irrigation water amounts (mm per application)
# ---------------------------------------------------------------------------

IRRIGATION_MM: dict[IrrigationLevel, float] = {
    IrrigationLevel.NONE: 0.0,
    IrrigationLevel.LIGHT: 5.0,
    IrrigationLevel.MODERATE: 15.0,
    IrrigationLevel.HEAVY: 30.0,
}


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    irrigation: IrrigationLevel = Field(description="Irrigation level for today")
    fertilizer: FertilizerType = Field(description="Fertilizer type for today")
    pest_management: PestManagement = Field(description="Pest management action for today")


# ---------------------------------------------------------------------------
# Tools + Budget (Round 2 — Phase 1)
# ---------------------------------------------------------------------------


class ToolAction(BaseModel):
    action_type: Literal["tool"] = Field(
        default="tool", description="Discriminator for tool calls"
    )
    tool: ToolCallType = Field(description="Which tool to call")


class ToolResult(BaseModel):
    tool: ToolCallType = Field(description="Tool that produced this result")
    day_requested: int = Field(ge=0, description="Day the tool was requested")
    day_available: int = Field(
        ge=0, description="Day the tool result becomes visible in Observation"
    )
    cost_usd: float = Field(
        ge=0, description="USD cost charged to budget for this tool call"
    )
    data: dict[str, Any] = Field(
        default_factory=dict, description="Tool payload (schema depends on tool)"
    )


class BudgetState(BaseModel):
    total_usd: Optional[float] = Field(
        default=None, description="Total budget (null = unlimited)"
    )
    spent_usd: float = Field(default=0.0, ge=0, description="Amount spent so far")
    remaining_usd: Optional[float] = Field(
        default=None, description="Remaining budget (null = unlimited)"
    )


# ---------------------------------------------------------------------------
# Crop Metrics — the outcome variables the agent observes
# ---------------------------------------------------------------------------

class CropMetrics(BaseModel):
    """Measurable crop and field outcomes."""
    crop_health: float = Field(ge=0, le=100, description="Overall plant vigor (0-100)")
    growth_rate: float = Field(ge=0, le=10, description="Daily growth rate (cm/day)")
    soil_health: float = Field(ge=0, le=100, description="Combined soil quality index (0-100)")
    water_stress: float = Field(ge=0, le=100, description="Water stress level (0=none, 100=severe)")
    nutrient_stress: float = Field(ge=0, le=100, description="Nutrient stress (0=none, 100=severe)")
    pest_pressure: float = Field(ge=0, le=100, description="Pest/disease threat level (0=none, 100=severe)")
    crop_quality: float = Field(ge=0, le=100, description="Expected harvest quality grade (0-100)")
    environmental_score: float = Field(ge=0, le=100, description="Sustainability score (0-100)")


class CropMetricDeltas(BaseModel):
    """Change in each metric since last step. This IS the outcome signal."""
    crop_health: float = Field(description="Δ crop health (positive = improving)")
    growth_rate: float = Field(description="Δ growth rate (positive = improving)")
    soil_health: float = Field(description="Δ soil health (positive = improving)")
    water_stress: float = Field(description="Δ water stress (negative = improving)")
    nutrient_stress: float = Field(description="Δ nutrient stress (negative = improving)")
    pest_pressure: float = Field(description="Δ pest pressure (negative = improving)")
    crop_quality: float = Field(description="Δ crop quality (positive = improving)")
    environmental_score: float = Field(description="Δ environmental score (positive = improving)")


# ---------------------------------------------------------------------------
# Weather — observable by the agent
# ---------------------------------------------------------------------------

class WeatherObservation(BaseModel):
    """Today's weather (visible to the agent)."""
    temperature: float = Field(description="Average temperature (°C)")
    rainfall_mm: float = Field(ge=0, description="Rainfall today (mm)")
    is_extreme_event: bool = Field(default=False, description="Heat wave or storm active")
    extreme_event_type: Optional[str] = Field(
        default=None, description="Type of extreme event if active"
    )


# ---------------------------------------------------------------------------
# Observation — what the agent sees
# ---------------------------------------------------------------------------

class OutcomeTrends(BaseModel):
    """7-day trends for outcome variables."""
    crop_health_trend: float = Field(description="7-day slope of crop health")
    growth_rate_trend: float = Field(description="7-day slope of growth rate")
    soil_health_trend: float = Field(description="7-day slope of soil health")
    water_stress_trend: float = Field(description="7-day slope of water stress")
    nutrient_stress_trend: float = Field(description="7-day slope of nutrient stress")
    pest_pressure_trend: float = Field(description="7-day slope of pest pressure")
    reward_trend: float = Field(description="7-day slope of total reward")
    reward_consistency: float = Field(description="Stddev of reward over last 7 days")


class Observation(BaseModel):
    day: int = Field(ge=0, description="Current day in the episode")
    total_days: int = Field(description="Total days in this episode")
    growth_stage: GrowthStage = Field(description="Current crop growth stage")
    metrics: CropMetrics = Field(description="Current crop/field metric values")
    deltas: CropMetricDeltas = Field(description="Change since last step")
    weather: WeatherObservation = Field(description="Today's weather conditions")
    trends: Optional[OutcomeTrends] = Field(
        default=None, description="7-day outcome trends (available after day 7)"
    )
    scenario_name: str = Field(description="Name of the current scenario")
    soil_moisture: float = Field(ge=0, le=100, description="Current soil moisture %")
    water_used_total: float = Field(ge=0, description="Cumulative irrigation water used (mm)")
    budget: BudgetState = Field(
        default_factory=BudgetState, description="Budget totals/spend/remaining"
    )
    tool_result: Optional[ToolResult] = Field(
        default=None,
        description="Most recent tool output (appears the step AFTER the call)",
    )


# ---------------------------------------------------------------------------
# Reward — based on outcome deltas weighted by task
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    crop_health_reward: float = Field(description="Reward from crop health change")
    growth_rate_reward: float = Field(description="Reward from growth rate change")
    soil_health_reward: float = Field(description="Reward from soil health change")
    water_stress_reward: float = Field(description="Reward from water stress change")
    nutrient_stress_reward: float = Field(description="Reward from nutrient stress change")
    pest_pressure_reward: float = Field(description="Reward from pest pressure change")
    crop_quality_reward: float = Field(description="Reward from crop quality change")
    environmental_score_reward: float = Field(description="Reward from environmental score change")
    agronomy_total: float = Field(
        default=50.0,
        description="Agronomy-only reward component before economic blending (0-100)",
    )
    economic_total: float = Field(
        default=50.0,
        description="Economic reward component (0-100); defaults to neutral baseline when not provided",
    )
    profit_usd: Optional[float] = Field(
        default=None, description="Per-step profit estimate (revenue - costs), if computed"
    )
    revenue_usd: Optional[float] = Field(
        default=None, description="Per-step revenue estimate (typically only at harvest), if computed"
    )
    cost_usd: Optional[float] = Field(
        default=None, description="Per-step action/tool cost in USD, if computed"
    )
    total: float = Field(description="Task-weighted total reward (0-100)")


# ---------------------------------------------------------------------------
# Step Result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: Observation
    reward: RewardBreakdown
    done: bool = Field(description="Whether the episode has ended")
    info: dict = Field(
        default_factory=dict,
        description="Additional metadata (weather events, delayed effects, etc.)",
    )


# ---------------------------------------------------------------------------
# Full Environment State (for grading)
# ---------------------------------------------------------------------------

class EnvState(BaseModel):
    day: int
    total_days: int
    growth_stage: GrowthStage
    scenario_name: str
    metrics: CropMetrics
    soil_moisture: float
    water_used_total: float
    history: list  # Full day-by-day history
    cumulative_reward: float
    action_space_description: str = Field(
        default="irrigation × fertilizer × pest_management = 4 × 5 × 5 = 100 combos"
    )
