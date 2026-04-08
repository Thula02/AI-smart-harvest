"""Outcome-based reward computation for crop management.

Reward = sum of task-weighted crop metric deltas, normalized to [0, 100].
Each task defines which outcomes matter most.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from .models import CropMetricDeltas, CropMetrics, RewardBreakdown


# ---------------------------------------------------------------------------
# Task-specific weights for each metric delta.
#
# Convention:
#   - Negative deltas are "good" for water_stress, nutrient_stress, pest_pressure
#     → we negate them before weighting so positive = better for all.
#   - All weights for a task sum to ~1.0.
# ---------------------------------------------------------------------------

TASK_WEIGHTS: dict[str, dict[str, float]] = {
    "ideal_season": {
        "crop_health": 0.20,
        "growth_rate": 0.25,
        "soil_health": 0.05,
        "water_stress": 0.05,
        "nutrient_stress": 0.10,
        "pest_pressure": 0.10,
        "crop_quality": 0.20,
        "environmental_score": 0.05,
    },
    "variable_weather": {
        "crop_health": 0.15,
        "growth_rate": 0.15,
        "soil_health": 0.15,
        "water_stress": 0.15,
        "nutrient_stress": 0.10,
        "pest_pressure": 0.15,
        "crop_quality": 0.10,
        "environmental_score": 0.05,
    },
    "drought_year": {
        "crop_health": 0.20,
        "growth_rate": 0.10,
        "soil_health": 0.10,
        "water_stress": 0.25,
        "nutrient_stress": 0.10,
        "pest_pressure": 0.10,
        "crop_quality": 0.05,
        "environmental_score": 0.10,
    },
}

# ---------------------------------------------------------------------------
# Normalization scales: how much delta is "very good" for each metric.
# ---------------------------------------------------------------------------

DELTA_SCALES: dict[str, float] = {
    "crop_health": 3.0,           # +3/day is excellent
    "growth_rate": 0.5,           # +0.5 cm/day improvement is excellent
    "soil_health": 1.0,           # +1/day is excellent (soil changes slowly)
    "water_stress": 3.0,          # -3/day stress reduction is excellent
    "nutrient_stress": 3.0,       # -3/day stress reduction is excellent
    "pest_pressure": 3.0,         # -3/day pressure reduction is excellent
    "crop_quality": 1.5,          # +1.5/day is excellent
    "environmental_score": 1.5,   # +1.5/day is excellent
}

# Metrics where LOWER is better (we negate so positive = good)
LOWER_IS_BETTER = {"water_stress", "nutrient_stress", "pest_pressure"}


# ---------------------------------------------------------------------------
# State quality ranges for normalizing absolute metric values.
# Each metric is normalized to [0, 1] based on its achievable range.
# ---------------------------------------------------------------------------

STATE_RANGES: dict[str, tuple[float, float]] = {
    "crop_health": (0, 100),
    "growth_rate": (0, 10),
    "soil_health": (0, 100),
    "water_stress": (0, 100),
    "nutrient_stress": (0, 100),
    "pest_pressure": (0, 100),
    "crop_quality": (0, 100),
    "environmental_score": (0, 100),
}

# Blend ratio: how much of the reward comes from deltas vs state quality
_DELTA_WEIGHT = 0.7
_STATE_WEIGHT = 0.3


def _compute_state_quality(metrics: CropMetrics, weights: dict[str, float]) -> float:
    """Compute state quality score from absolute metric values.

    Returns a value in [0, 1] representing how good the current state is.
    Uses the same task weights as the delta component.
    """
    metrics_dict = metrics.model_dump()
    quality = 0.0
    for metric, value in metrics_dict.items():
        lo, hi = STATE_RANGES[metric]
        # Normalize to [0, 1]
        normalized = (value - lo) / (hi - lo) if hi > lo else 0.5
        # Flip for "lower is better" metrics
        if metric in LOWER_IS_BETTER:
            normalized = 1.0 - normalized
        quality += normalized * weights[metric]
    return quality


def compute_reward(
    deltas: CropMetricDeltas,
    task_name: str,
    current_metrics: Optional[CropMetrics] = None,
) -> RewardBreakdown:
    """Compute blended reward from metric deltas and absolute state quality.

    Reward = delta_score * 0.7 + state_score * 0.3

    - delta_score (0-100): rewards improving metrics (same as original formula).
    - state_score (0-100): rewards maintaining good absolute metric values.

    When current_metrics is None (e.g. in tests), state quality defaults to
    0.5 so the baseline remains 50.
    """
    weights = TASK_WEIGHTS[task_name]
    delta_dict = deltas.model_dump()

    per_metric: dict[str, float] = {}
    weighted_sum = 0.0

    for metric, raw_delta in delta_dict.items():
        w = weights[metric]
        scale = DELTA_SCALES[metric]

        # Normalize: flip sign for "lower is better" metrics
        if metric in LOWER_IS_BETTER:
            normalized = -raw_delta / scale
        else:
            normalized = raw_delta / scale

        # Clamp to [-2, +2] to prevent outlier noise from dominating
        normalized = max(-2.0, min(2.0, normalized))

        # Scale to a per-metric reward contribution
        metric_reward = normalized * w * 100.0
        per_metric[metric] = round(metric_reward, 3)
        weighted_sum += metric_reward

    # Delta score: baseline 50 + improvements
    delta_score = max(0.0, min(100.0, 50.0 + weighted_sum))

    # State quality score: how good are current absolute values?
    if current_metrics is not None:
        state_quality = _compute_state_quality(current_metrics, weights)
    else:
        state_quality = 0.5  # Neutral default preserves baseline = 50
    state_score = state_quality * 100.0

    # Blend delta and state components
    total = delta_score * _DELTA_WEIGHT + state_score * _STATE_WEIGHT
    total = max(0.0, min(100.0, total))

    return RewardBreakdown(
        crop_health_reward=per_metric["crop_health"],
        growth_rate_reward=per_metric["growth_rate"],
        soil_health_reward=per_metric["soil_health"],
        water_stress_reward=per_metric["water_stress"],
        nutrient_stress_reward=per_metric["nutrient_stress"],
        pest_pressure_reward=per_metric["pest_pressure"],
        crop_quality_reward=per_metric["crop_quality"],
        environmental_score_reward=per_metric["environmental_score"],
        total=round(total, 2),
    )


# ---------------------------------------------------------------------------
# Utility functions (used by graders)
# ---------------------------------------------------------------------------

def _stddev(values: list[float]) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _linear_slope(values: list[float]) -> float:
    """Simple linear regression slope over indices 0..n-1."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return num / den
