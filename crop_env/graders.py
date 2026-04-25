"""Task graders — compute 0.0–1.0 scores from episode history.

These evaluate based on actual crop outcome improvements (metric deltas),
not action quality.  Same structural pattern as the wellness env graders.
"""

from __future__ import annotations

from typing import Any

from .payoff import _linear_slope, _stddev


def _normalize(x: float, lo: float, hi: float) -> float:
    """Normalize x to [0, 1] given expected range [lo, hi]."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _get_primary_metric_key(task_name: str) -> str:
    """Return the primary metric for each task."""
    primary_map = {
        "ideal_season": "growth_rate",
        "variable_weather": "crop_health",
        "drought_year": "water_stress",
    }
    return primary_map.get(task_name, "crop_health")


def _metric_breadth(history: list[dict[str, Any]]) -> float:
    """Fraction of 8 metrics that improved from start to end of episode."""
    if len(history) < 2:
        return 0.0

    first = history[0].get("metrics", {})
    last = history[-1].get("metrics", {})
    lower_is_better = {"water_stress", "nutrient_stress", "pest_pressure"}

    improved = 0
    total = 0
    for key in first:
        if key not in last:
            continue
        total += 1
        delta = last[key] - first[key]
        if key in lower_is_better:
            if delta < 0:
                improved += 1
        else:
            if delta > 0:
                improved += 1

    if total == 0:
        return 0.0
    return improved / total


# ---------------------------------------------------------------------------
# Task 1: Ideal Season (Easy)
# ---------------------------------------------------------------------------

def grade_ideal_season(history: list[dict[str, Any]]) -> float:
    """Easy task: optimize growth and crop quality in favorable conditions.

    Score = 0.6 * normalize(avg_reward, 40, 75)
          + 0.2 * normalize(primary_metric_improvement, 0, scale)
          + 0.2 * reward_trend
    """
    if not history:
        return 0.0

    rewards = [h["reward_total"] for h in history]
    avg_reward = sum(rewards) / len(rewards)
    trend = _linear_slope(rewards)

    # Primary metric: growth_rate improvement
    primary_key = "growth_rate"
    first_val = history[0].get("metrics", {}).get(primary_key, 0)
    last_val = history[-1].get("metrics", {}).get(primary_key, 0)
    primary_improvement = last_val - first_val  # Higher is better

    score = (
        0.6 * _normalize(avg_reward, 40, 75)
        + 0.2 * _normalize(primary_improvement, 0, 3.0)
        + 0.2 * _normalize(trend, -0.5, 1.0)
    )
    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Task 2: Variable Weather (Medium)
# ---------------------------------------------------------------------------

def grade_variable_weather(history: list[dict[str, Any]]) -> float:
    """Medium task: balance all metrics under variable conditions.

    Score = 0.35 * normalize(avg_reward, 25, 60)
          + 0.25 * metric_breadth
          + 0.20 * consistency (1 - normalize(stddev, 0, 15))
          + 0.20 * max(0, reward_trend)
    """
    if not history:
        return 0.0

    rewards = [h["reward_total"] for h in history]
    avg_reward = sum(rewards) / len(rewards)
    trend = _linear_slope(rewards)
    std = _stddev(rewards)
    breadth = _metric_breadth(history)

    consistency = 1.0 - _normalize(std, 0, 15)

    score = (
        0.35 * _normalize(avg_reward, 25, 60)
        + 0.25 * breadth
        + 0.20 * consistency
        + 0.20 * max(0.0, _normalize(trend, 0, 1.0))
    )
    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Task 3: Drought Year (Hard)
# ---------------------------------------------------------------------------

def grade_drought_year(history: list[dict[str, Any]]) -> float:
    """Hard task: maximize outcomes under drought with limited water budget.

    Score = 0.25 * normalize(avg_reward, 5, 30)
          + 0.25 * normalize(improvement_last7_vs_first7, 0, 10)
          + 0.20 * consistency
          + 0.15 * environmental_score_improvement
          + 0.15 * metric_breadth
    """
    if not history:
        return 0.0

    rewards = [h["reward_total"] for h in history]
    avg_reward = sum(rewards) / len(rewards)
    std = _stddev(rewards)

    # Improvement: last 7 vs first 7 avg reward
    first7 = rewards[:7] if len(rewards) >= 7 else rewards
    last7 = rewards[-7:] if len(rewards) >= 7 else rewards
    improvement = (sum(last7) / len(last7)) - (sum(first7) / len(first7))

    consistency = 1.0 - _normalize(std, 0, 15)
    breadth = _metric_breadth(history)

    # Environmental score improvement
    env_first = history[0].get("metrics", {}).get("environmental_score", 70)
    env_last = history[-1].get("metrics", {}).get("environmental_score", 70)
    env_improvement = _normalize(env_last - env_first, -10, 15)

    score = (
        0.25 * _normalize(avg_reward, 5, 30)
        + 0.25 * _normalize(improvement, 0, 10)
        + 0.20 * consistency
        + 0.15 * env_improvement
        + 0.15 * breadth
    )
    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Task 4: Supply Chain Disruption
# ---------------------------------------------------------------------------

def grade_supply_chain_disruption(history: list[dict[str, Any]]) -> float:
    """Supply chain task: keep outcomes stable under input constraints.

    Emphasizes: avg reward, nutrient stress improvement, breadth, and budget discipline.
    """
    if not history:
        return 0.0

    rewards = [h.get("reward_total", 0.0) for h in history]
    avg_reward = sum(rewards) / len(rewards)
    breadth = _metric_breadth(history)

    # Nutrient stress improvement (lower is better)
    ns_first = history[0].get("metrics", {}).get("nutrient_stress", 50)
    ns_last = history[-1].get("metrics", {}).get("nutrient_stress", 50)
    ns_improvement = (ns_first - ns_last)  # positive is good

    # Budget discipline (if present)
    bud = (history[-1].get("budget") or {})
    total = bud.get("total_usd", None)
    spent = bud.get("spent_usd", 0.0)
    if total in (None, 0):
        budget_score = 0.5
    else:
        budget_score = 1.0 - _normalize(float(spent) / float(total), 0.0, 1.0)

    score = (
        0.40 * _normalize(avg_reward, 20, 60)
        + 0.20 * _normalize(ns_improvement, 0, 20)
        + 0.25 * breadth
        + 0.15 * budget_score
    )
    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Task 5: Regulatory Shift
# ---------------------------------------------------------------------------

def grade_regulatory_shift(history: list[dict[str, Any]]) -> float:
    """Regulatory task: maintain pest control while improving environmental score."""
    if not history:
        return 0.0

    rewards = [h.get("reward_total", 0.0) for h in history]
    avg_reward = sum(rewards) / len(rewards)
    breadth = _metric_breadth(history)

    env_first = history[0].get("metrics", {}).get("environmental_score", 70)
    env_last = history[-1].get("metrics", {}).get("environmental_score", 70)
    env_improvement = (env_last - env_first)

    pp_first = history[0].get("metrics", {}).get("pest_pressure", 30)
    pp_last = history[-1].get("metrics", {}).get("pest_pressure", 30)
    pest_improvement = (pp_first - pp_last)  # positive is good

    score = (
        0.35 * _normalize(avg_reward, 20, 60)
        + 0.25 * _normalize(env_improvement, 0, 20)
        + 0.25 * _normalize(pest_improvement, 0, 20)
        + 0.15 * breadth
    )
    return round(max(0.0, min(1.0, score)), 4)
