# Crop Management OpenEnv — Round 1 Requirements (OpenEnv Hackathon)

This document is a **focused slice** of the full vision ([requirements-round2.md](requirements-round2.md)) scoped to the Hackathon Round 1 deliverables: an OpenEnv-compliant crop management environment.

---

## 1. Scope & Goal

Build an **Outcome-Based Crop Management Simulator Environment** where an RL/LLM agent manages a simulated farm field — measured by **crop/field metric changes** (crop health, growth rate, soil health, water stress, nutrient stress, pest pressure, crop quality, environmental score) — over a multi-day growing season. The agent takes actions (irrigation, fertilization, pest management) and observes **outcome deltas**, learning what action sequences produce the best crop improvements for each scenario. The environment exposes the OpenEnv interface (`step()`, `reset()`, `state()`) with Pydantic-typed models.

**Key design principle:** There are no fixed action-quality scores. The reward is computed entirely from scenario-weighted crop metric deltas. Each scenario has a **hidden field response model** (FieldModel) that the agent must learn through experience — the same action produces different outcomes under different soil, weather, and pest conditions.

**Key differentiator:** Delayed feedback. Fertilizer applied today may not affect nutrient stress for 2–4 days (synthetic) or 5–10 days (organic). Pest resistance to chemicals accumulates over weeks. Weather introduces irreducible stochasticity. Crop growth stages change metric sensitivity over the season.

**What Round 1 delivers:**
- OpenEnv-compliant environment (the simulator + outcome-based payoff function)
- 3 tasks with programmatic graders (easy → medium → hard)
- Baseline `inference.py` using an LLM agent via OpenAI API
- Dockerfile for deployment
- `openenv.yaml` metadata

**What Round 1 does NOT include** (deferred to Round 2+):
- Trained NN policy (DQN/PPO/SAC) — Round 1 agent is an LLM
- Multi-field / multi-crop / market price dynamics
- Satellite imagery or real-world data integration
- Production deployment, observability, privacy

---

## 2. OpenEnv Interface

The environment must implement the OpenEnv specification. All inputs and outputs are Pydantic models.

### 2.1 Core Methods

```python
class CropEnv:
    def reset(self, task_name: str) -> Observation:
        """Initialize a new episode for the given task. Returns initial observation."""

    def step(self, action: Action) -> StepResult:
        """Apply action, advance one day, return (observation, reward, done, info)."""

    def state(self) -> EnvState:
        """Return the full internal state (for debugging/grading). Not seen by agent."""

    def grade(self) -> float:
        """Compute the grader score (0.0–1.0) for the current task."""
```

### 2.2 Pydantic Models

```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

# --- Enums ---

class IrrigationLevel(str, Enum):
    NONE = "none"
    LIGHT = "light"        # 5mm
    MODERATE = "moderate"  # 15mm
    HEAVY = "heavy"        # 30mm

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

# --- Action ---

class Action(BaseModel):
    irrigation: IrrigationLevel
    fertilizer: FertilizerType
    pest_management: PestManagement

# --- Crop Metrics (the outcome variables) ---

class CropMetrics(BaseModel):
    crop_health: float = Field(ge=0, le=100)
    growth_rate: float = Field(ge=0, le=10)
    soil_health: float = Field(ge=0, le=100)
    water_stress: float = Field(ge=0, le=100)
    nutrient_stress: float = Field(ge=0, le=100)
    pest_pressure: float = Field(ge=0, le=100)
    crop_quality: float = Field(ge=0, le=100)
    environmental_score: float = Field(ge=0, le=100)

class CropMetricDeltas(BaseModel):
    crop_health: float
    growth_rate: float
    soil_health: float
    water_stress: float
    nutrient_stress: float
    pest_pressure: float
    crop_quality: float
    environmental_score: float

# --- Weather (visible to agent) ---

class WeatherObservation(BaseModel):
    temperature: float
    rainfall_mm: float = Field(ge=0)
    is_extreme_event: bool = False
    extreme_event_type: Optional[str] = None

# --- Observation ---

class Observation(BaseModel):
    day: int
    total_days: int
    growth_stage: GrowthStage
    metrics: CropMetrics
    deltas: CropMetricDeltas
    weather: WeatherObservation
    trends: Optional[OutcomeTrends] = None  # Available after day 7
    scenario_name: str
    soil_moisture: float
    water_used_total: float

# --- Reward ---

class RewardBreakdown(BaseModel):
    crop_health_reward: float
    growth_rate_reward: float
    soil_health_reward: float
    water_stress_reward: float
    nutrient_stress_reward: float
    pest_pressure_reward: float
    crop_quality_reward: float
    environmental_score_reward: float
    total: float  # 0-100

# --- Step Result ---

class StepResult(BaseModel):
    observation: Observation
    reward: RewardBreakdown
    done: bool
    info: dict
```

### 2.3 `openenv.yaml`

```yaml
tasks:
  - name: ideal_season
    description: "Easy: Optimize crop growth in favorable conditions over 60 days"
    difficulty: easy
  - name: variable_weather
    description: "Medium: Balance all 8 metrics under variable weather over 90 days"
    difficulty: medium
  - name: drought_year
    description: "Hard: Maximize outcomes under drought with limited water budget over 90 days"
    difficulty: hard

environment:
  name: crop-outcome
  description: >
    Outcome-based crop management simulator where rewards are driven by
    measured field/crop metric changes weighted by farming scenario.
    Features delayed feedback, pest resistance buildup, weather stochasticity,
    and crop growth stages with varying sensitivity.

entrypoint: python inference.py
```

---

## 3. Crop Metric Outcomes — The Reward Signal

For Round 1, the reward is **entirely outcome-based**. There are no fixed action-quality scores. The agent observes 8 crop/field metrics and is rewarded based on how its actions change those metrics, weighted by the farming scenario.

### 3.1 Metric Definitions

| Metric | Direction | Range | What It Measures |
|--------|-----------|-------|------------------|
| `crop_health` | Higher is better | 0–100 | Overall plant vigor |
| `growth_rate` | Higher is better | 0–10 cm/day | Daily growth rate |
| `soil_health` | Higher is better | 0–100 | Combined soil quality index |
| `water_stress` | Lower is better | 0–100 | Water stress level |
| `nutrient_stress` | Lower is better | 0–100 | Nutrient deficiency/excess |
| `pest_pressure` | Lower is better | 0–100 | Pest/disease threat |
| `crop_quality` | Higher is better | 0–100 | Expected harvest quality |
| `environmental_score` | Higher is better | 0–100 | Sustainability score |

### 3.2 Scenario-Specific Weighting

Each scenario defines different weights for each metric. All weights for a scenario sum to 1.0:

| Metric | Ideal Season | Variable Weather | Drought Year |
|--------|-------------|-----------------|-------------|
| `crop_health` | 0.20 | 0.15 | 0.20 |
| `growth_rate` | 0.25 | 0.15 | 0.10 |
| `soil_health` | 0.05 | 0.15 | 0.10 |
| `water_stress` | 0.05 | 0.15 | 0.25 |
| `nutrient_stress` | 0.10 | 0.10 | 0.10 |
| `pest_pressure` | 0.10 | 0.15 | 0.10 |
| `crop_quality` | 0.20 | 0.10 | 0.05 |
| `environmental_score` | 0.05 | 0.05 | 0.10 |

### 3.3 Actions (the Agent's Levers)

Irrigation, fertilizer, and pest management are levers the agent can pull — not reward categories:

| Action Dimension | Options | Count |
|-----------------|---------|-------|
| Irrigation | none, light (5mm), moderate (15mm), heavy (30mm) | 4 |
| Fertilizer | none, nitrogen, phosphorus, balanced, organic | 5 |
| Pest Management | none, scouting, biological, chemical_light, chemical_heavy | 5 |

Total: 4 × 5 × 5 = **100 action combinations** per time step.

### 3.4 Cross-Action Interactions (Hidden in Simulator)

These interactions are encoded in the simulator's field response model. They are **not visible to the agent** — the agent must discover them through observed outcomes:

| Interaction | Effect |
|---|---|
| Over-irrigation + high soil moisture | Waterlogging → water stress spike, soil degradation |
| Heavy irrigation in rain | Runoff → environmental score drop |
| Over-fertilization (synthetic) | Nutrient burn → nutrient stress increase |
| Organic fertilizer | Slow release (5-10 days), improves soil health |
| Chemical pesticides + repeated use | Pest resistance builds up (effectiveness drops to 5%) |
| Chemical pesticides in rain | Reduced effectiveness + environmental runoff |
| Biological pest control + healthy soil | Enhanced effectiveness |
| Stress during flowering/fruiting | Disproportionate crop quality damage |
| Heat wave | Water stress spike, crop health damage |
| Storm | Direct crop health damage (3-8 points) |

---

## 4. Payoff Function (Outcome-Based Reward)

### 4.1 Per-Step Reward

$$R_t = 50 + \sum_{i=1}^{8} \text{normalize}(\Delta_i) \times w_{i,\text{scenario}} \times 100$$

Clamped to $[0, 100]$. Baseline (no metric change) = 50. Improvements push above 50, declines push below 50.

### 4.2 Reward Computation

For each metric:

1. **Raw delta**: The change in the metric from the previous step (computed by the simulator)
2. **Sign normalization**: For "lower is better" metrics (`water_stress`, `nutrient_stress`, `pest_pressure`), the delta is negated so positive always = improving
3. **Scale normalization**: Divide by a `DELTA_SCALE` that defines what an "excellent" daily improvement looks like
4. **Clamp**: Normalized value clamped to [-2, +2] to prevent outlier noise from dominating
5. **Weight**: Multiply by the scenario-specific weight

**Delta scales (what constitutes "excellent" daily improvement):**

| Metric | Delta Scale | Meaning |
|--------|------------|---------|
| `crop_health` | 3.0 | +3/day is excellent |
| `growth_rate` | 0.5 cm/day | +0.5/day improvement is excellent |
| `soil_health` | 1.0 | +1/day is excellent (soil changes slowly) |
| `water_stress` | 3.0 | −3/day stress reduction is excellent |
| `nutrient_stress` | 3.0 | −3/day stress reduction is excellent |
| `pest_pressure` | 3.0 | −3/day pressure reduction is excellent |
| `crop_quality` | 1.5 | +1.5/day is excellent |
| `environmental_score` | 1.5 | +1.5/day is excellent |

---

## 5. Hidden Transition Model — FieldModel

Each scenario has a **FieldModel** — a set of ~14 hidden parameters that govern how actions translate to metric changes. The agent never sees these parameters; it only observes the resulting metric deltas and weather.

### 5.1 FieldModel Parameters

| Parameter | Description |
|-----------|-------------|
| `moisture_retention` | Fraction of soil moisture retained daily |
| `nutrient_absorption_rate` | Rate of synthetic fertilizer nutrient release |
| `organic_decomposition_rate` | Rate of organic fertilizer decomposition |
| `waterlogging_threshold` | Soil moisture % above which waterlogging occurs |
| `nutrient_burn_threshold` | Nutrient level above which burn occurs |
| `crop_water_sensitivity` | How strongly crops react to water stress |
| `crop_nutrient_sensitivity` | How strongly crops react to nutrient stress |
| `base_growth_rate` | Ideal growth rate under perfect conditions |
| `pest_growth_rate` | Natural daily pest population growth |
| `pest_chemical_resistance` | Baseline pest resistance to chemicals |
| `resistance_buildup_rate` | How fast resistance grows per chemical application |
| `rain_probability` | Daily probability of rainfall |
| `rain_intensity_mean` | Average rainfall when it rains (mm) |
| `temperature_mean` | Average daily temperature (°C) |

### 5.2 Delayed Feedback

This is the key distinguishing feature:

- **Synthetic fertilizer** (nitrogen, phosphorus, balanced): Peak nutrient release at day+2, fading by day+4
- **Organic fertilizer**: Slow release from day+3 through day+10, plus long-term soil health improvement
- **Pest resistance**: Each chemical application adds `resistance_buildup_rate` to accumulated resistance, capped at 95%

### 5.3 Growth Stages

The crop passes through 5 growth stages, each with different sensitivity to water, nutrients, and growth potential:

| Stage | Day Range (fraction) | Water Sensitivity | Nutrient Sensitivity | Growth Multiplier |
|-------|---------------------|-------------------|---------------------|-------------------|
| Germination | 0–15% | 0.8 | 0.6 | 0.3 |
| Vegetative | 15–40% | 1.0 | 1.2 | 1.0 |
| Flowering | 40–60% | 1.3 | 1.0 | 0.8 |
| Fruiting | 60–80% | 1.1 | 0.8 | 0.5 |
| Maturity | 80–100% | 0.6 | 0.4 | 0.2 |

### 5.4 Weather System

Weather introduces irreducible stochasticity (unlike the wellness env's compliance-based randomness):

- **Temperature**: Drawn from N(mean, 3.0)
- **Rainfall**: Occurs with probability `rain_probability`; intensity drawn from N(rain_intensity_mean, 4.0)
- **Extreme events** (5% daily chance):
  - **Heat wave**: 3-5 days of elevated temperature (+8-15°C), increases water stress and crop damage
  - **Storm**: 1-2 days of heavy rain (4× intensity), direct crop health damage (3-8 points)

---

## 6. Tasks & Graders

### 6.1 Task Overview

| Task | Scenario | Days | Difficulty | Key Challenge |
|------|----------|------|------------|---------------|
| `ideal_season` | Fertile soil, regular rain, ∞ water | 60 | Easy | Optimize growth rate and crop quality |
| `variable_weather` | Alternating wet/dry, pest surge mid-season | 90 | Medium | Balance all 8 metrics under variability |
| `drought_year` | Low rain, depleted soil, 800mm water budget | 90 | Hard | Maximize outcomes with constrained water |

### 6.2 Grader Formulas

#### Task 1: `ideal_season` (Easy)

$$\text{Score} = 0.6 \times \text{norm}(\overline{R}, 40, 75) + 0.2 \times \text{norm}(\Delta\text{growth\_rate}, 0, 3.0) + 0.2 \times \text{norm}(\text{trend}, -0.5, 1.0)$$

#### Task 2: `variable_weather` (Medium)

$$\text{Score} = 0.35 \times \text{norm}(\overline{R}, 40, 75) + 0.25 \times \text{breadth} + 0.20 \times \text{consistency} + 0.20 \times \max(0, \text{trend})$$

Where `breadth` = fraction of 8 metrics that improved over the episode.

#### Task 3: `drought_year` (Hard)

$$\text{Score} = 0.25 \times \text{norm}(\overline{R}, 35, 65) + 0.25 \times \text{norm}(\text{improvement}, 0, 15) + 0.20 \times \text{consistency} + 0.15 \times \text{env\_improvement} + 0.15 \times \text{breadth}$$

Where `improvement` = avg(last 7 rewards) − avg(first 7 rewards).

---

## 7. Inference Agent

### 7.1 LLM Agent

The baseline agent uses an LLM (GPT-4o-mini via OpenAI API) to select actions. The system prompt explains:
- The 8 crop metrics and their improvement directions
- The 3 action dimensions and their effects
- Delayed feedback dynamics (fertilizer delay, pest resistance)
- Growth stages and their varying sensitivity
- Weather events and their impacts
- Cross-action interactions (waterlogging, nutrient burn, chemical resistance)

### 7.2 Fallback Policy

If the LLM call fails or JSON parsing fails, a rule-based fallback policy activates:
- **Irrigation**: Based on soil moisture (high → none, low → heavy), water stress, and growth stage sensitivity
- **Fertilizer**: Based on nutrient stress (high → balanced/organic), growth stage, and soil moisture
- **Pest management**: Based on pest pressure (high → chemical/biological), soil health, and rainfall

### 7.3 stdout Format

```
[START] task=ideal_season
[STEP] day=1 | metrics: crop_health=62.3 growth_rate=2.1 ... | deltas: crop_health=+2.3 ... | weather: temp=24.5 rain=3.2mm | soil_moisture=55.2 | stage=vegetative | reward=58.4
...
[END] task=ideal_season | score=0.72 | avg_reward=61.3 | trend=0.45 | stddev=8.2 | water_used=342.5mm
```

---

## 8. Test Suite

76 tests across 4 files:

| File | Tests | Coverage |
|------|-------|----------|
| `test_env.py` | 32 | Models, scenarios, env interface, end-to-end episodes |
| `test_payoff.py` | 11 | Weight structure, delta scales, reward computation |
| `test_simulator.py` | 13 | Transition dynamics, delayed fertilizer, pest resistance, weather |
| `test_graders.py` | 13 | Empty history, score range, ordering, determinism |

---

## 9. Deliverables Checklist

- [x] `crop_env/env.py` — CropEnv class (reset/step/state/grade)
- [x] `crop_env/models.py` — All Pydantic models
- [x] `crop_env/simulator.py` — Hidden transition dynamics (FieldModel response)
- [x] `crop_env/payoff.py` — Outcome-based reward computation
- [x] `crop_env/graders.py` — Task graders
- [x] `crop_env/scenarios.py` — FieldModel configs, weather generation, growth stages
- [x] `crop_env/__init__.py` — Package exports
- [x] `inference.py` — LLM agent + fallback policy
- [x] `tests/` — 76 tests (4 files)
- [x] `openenv.yaml` — Task metadata
- [x] `Dockerfile` — Deployment container
- [x] `requirements.txt` — Dependencies
- [x] `README.md` — Documentation
