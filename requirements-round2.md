# Crop Management OpenEnv — Full Requirements

## 1. Vision

A crop management simulator that uses Reinforcement Learning to optimize farming decisions — measured by crop/field metrics like crop health, growth rate, soil health, water stress, nutrient stress, pest pressure, crop quality, and environmental score. The RL agent learns optimal action sequences (irrigation, fertilization, pest management) by observing how actions change field conditions, then recommends daily farming actions that maximize crop outcome improvements for each scenario's priorities.

### 1.1 Terminology

| Term | Definition |
|------|-----------|
| **Policy** | The learned decision function that maps states to recommended actions. |
| **Scenario** | A set of environmental conditions (soil type, weather pattern, pest dynamics) that define a farming challenge. Each scenario has hidden parameters. |
| **FieldModel** | The hidden response model for a scenario — ~14 parameters governing soil, crop, pest, and weather dynamics. |
| **Payoff function** | Deterministic scoring code that evaluates crop metric deltas and returns a numerical reward. |
| **Episode** | One complete run through a growing season (60-90 simulated days). |
| **Step** | One day within an episode. |
| **Growth stage** | Current crop development phase (germination → vegetative → flowering → fruiting → maturity). |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RL ENVIRONMENT                              │
│                                                                     │
│  State Space ──→ Agent ──→ Actions ──→ Simulator ──→ Next State     │
│  (metrics,        ↑       (irrigation,   (hidden         │          │
│   deltas,         │        fertilizer,   FieldModel      │          │
│   weather,        │        pest mgmt)    response)       │          │
│   trends,         │                        │             │          │
│   stage)          │                        │             │          │
│                   └──── Reward (from Payoff Function) ◄──┘          │
│                    (outcome-based: metric deltas × scenario weights)│
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Payoff Function

### 3.1 Structure

The payoff function is **outcome-based**: it computes reward from measured crop metric deltas, not from action quality. The scoring formula executes as deterministic code at runtime.

$$R = 50 + \sum_{i=1}^{8} \text{normalize}(\Delta_i) \times w_{i,\text{scenario}} \times 100 \quad \text{clamped to } [0, 100]$$

Where:
- $\Delta_i$ is the change in metric $i$ for this step (computed by the simulator)
- For "lower is better" metrics (water_stress, nutrient_stress, pest_pressure), the delta is negated so positive = improving
- Each delta is normalized by a DELTA_SCALE defining what "excellent daily improvement" looks like
- $w_{i,\text{scenario}}$ are scenario-specific weights (all sum to 1.0 per scenario)
- Baseline 50 = no change; >50 = metrics improved; <50 = metrics declined

### 3.2 Crop Metric Outcomes

The system measures 8 field/crop outcomes:

| Metric | Direction | Good Delta Scale | What It Captures |
|--------|-----------|-----------------|------------------|
| `crop_health` | Higher = better | +3/day | Overall plant vigor |
| `growth_rate` | Higher = better | +0.5 cm/day | Daily growth rate |
| `soil_health` | Higher = better | +1/day | Combined soil quality (changes slowly) |
| `water_stress` | Lower = better | −3/day | Water deficit or excess |
| `nutrient_stress` | Lower = better | −3/day | Nutrient deficiency or toxicity |
| `pest_pressure` | Lower = better | −3/day | Pest/disease threat level |
| `crop_quality` | Higher = better | +1.5/day | Expected harvest grade |
| `environmental_score` | Higher = better | +1.5/day | Sustainability and eco-impact |

### 3.3 Interaction Effects (embedded in simulator)

Cross-action interactions are captured in the **simulator's field transition model**, not as separate bonus terms. The payoff function only sees the resulting metric deltas:
- Over-irrigation causes waterlogging → water stress spikes → lower reward
- Synthetic fertilizer accumulation causes nutrient burn → nutrient stress increase → penalty
- Chemical pesticides lose effectiveness over time → pest pressure stops declining → stalled reward
- Organic fertilizer improves soil health long-term → positive soil_health deltas → higher reward
- Stress during flowering/fruiting damages quality disproportionately → large negative crop_quality deltas

### 3.4 Scenario-Dependent Weighting

Each scenario defines which metric improvements matter most:

| Scenario | Primary Metrics (weight ≥ 0.20) | Secondary Metrics |
|----------|--------------------------------|-------------------|
| Ideal Season | growth_rate (0.25), crop_health (0.20), crop_quality (0.20) | Others at 0.05–0.10 |
| Variable Weather | All balanced (0.10–0.15 each) | environmental_score (0.05) |
| Drought Year | water_stress (0.25), crop_health (0.20) | Others at 0.05–0.10 |

### 3.5 Weight Progression

| Phase | Weight Strategy |
|-------|----------------|
| V1 | Scenario-specific weights derived from agronomic prioritization (current) |
| V2 | Research-derived weights from crop science effect sizes |
| V3 | Region/crop-dependent fine-tuning from aggregated field data |
| V4 | Learned weights from farm-specific outcome tracking |

---

## 4. RL Environment

### 4.1 State Space

The state vector is composed of two layers:

**Layer 1 — Current snapshot (available from day 1):**
- Current crop metric values (8 values)
- Metric deltas since last step (8 values)
- Weather observation (temperature, rainfall, extreme events)
- Soil moisture level
- Cumulative water used
- Current growth stage (categorical)
- Scenario name (categorical)

**Layer 2 — Outcome trends (require 7+ days of history):**
- 7-day slopes for each metric (6 values: crop_health, growth_rate, soil_health, water_stress, nutrient_stress, pest_pressure)
- 7-day reward trend and reward consistency (2 values)

### 4.2 Action Space

| Dimension | Options | Count |
|-----------|---------|-------|
| Irrigation | none, light (5mm), moderate (15mm), heavy (30mm) | 4 |
| Fertilizer | none, nitrogen, phosphorus, balanced, organic | 5 |
| Pest management | none, scouting, biological, chemical_light, chemical_heavy | 5 |

Total combinations: 4 × 5 × 5 = **100**

### 4.3 Hidden FieldModel Parameters (~14)

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `moisture_retention` | Fraction of soil moisture retained daily | 0.55–0.85 |
| `nutrient_absorption_rate` | Synthetic fertilizer release rate | 0.3–0.7 |
| `organic_decomposition_rate` | Organic fertilizer decomposition rate | 0.2–0.45 |
| `waterlogging_threshold` | Moisture % causing waterlogging | 65–85 |
| `nutrient_burn_threshold` | Nutrient level causing burn | 110–135 |
| `crop_water_sensitivity` | Crop reaction to water stress | 0.6–1.0 |
| `crop_nutrient_sensitivity` | Crop reaction to nutrient stress | 0.5–0.9 |
| `base_growth_rate` | Ideal growth under perfect conditions | 2.5–5.0 |
| `pest_growth_rate` | Natural pest population growth | 0.5–1.5 |
| `pest_chemical_resistance` | Baseline pest resistance | 0.0–0.15 |
| `resistance_buildup_rate` | Resistance increase per chemical use | 0.02–0.05 |
| `rain_probability` | Daily rainfall probability | 0.08–0.40 |
| `rain_intensity_mean` | Average rainfall when it rains (mm) | 5–15 |
| `temperature_mean` | Average daily temperature (°C) | 22–30 |

### 4.4 Delayed Feedback

This is the key design differentiator from simple step-reward environments:

| Effect | Delay | Mechanism |
|--------|-------|-----------|
| Synthetic fertilizer | 2–4 days | Peak release at day+2, fades by day+4 |
| Organic fertilizer | 3–10 days | Slow decomposition from day+3 through day+10 |
| Pest resistance | Cumulative over weeks | Each chemical application adds to accumulated resistance |
| Soil health from organics | Days to weeks | Gradual improvement over many applications |

### 4.5 Growth Stages

| Stage | Day Fraction | Water Sensitivity | Nutrient Sensitivity | Growth Multiplier |
|-------|-------------|-------------------|---------------------|-------------------|
| Germination | 0–15% | 0.8 | 0.6 | 0.3 |
| Vegetative | 15–40% | 1.0 | 1.2 | 1.0 |
| Flowering | 40–60% | 1.3 | 1.0 | 0.8 |
| Fruiting | 60–80% | 1.1 | 0.8 | 0.5 |
| Maturity | 80–100% | 0.6 | 0.4 | 0.2 |

### 4.6 Weather System

Weather replaces compliance as the primary source of stochasticity:

- **Temperature**: Gaussian around field mean, σ = 3°C
- **Rainfall**: Bernoulli(rain_probability) × Gaussian(rain_intensity_mean, 4mm)
- **Extreme events** (5% daily chance):
  - **Heat wave**: 3-5 days, +8-15°C, increases water stress and crop damage
  - **Storm**: 1-2 days, 4× rainfall, direct crop health damage (3-8 points), chemical runoff

---

## 5. Scenarios

### 5.1 Ideal Season (Easy)

| Attribute | Value |
|-----------|-------|
| Duration | 60 days |
| Soil | Fertile (high retention, low resistance) |
| Weather | Regular rain (35% daily), temperate (25°C mean) |
| Water budget | Unlimited |
| Starting conditions | Healthy crops, moderate soil |
| Key challenge | Optimize growth rate and crop quality |

### 5.2 Variable Weather (Medium)

| Attribute | Value |
|-----------|-------|
| Duration | 90 days |
| Soil | Average (moderate retention) |
| Weather | Variable (25% rain), warmer (27°C mean) |
| Water budget | Unlimited |
| Starting conditions | Moderate across all metrics |
| Key challenges | Weather variability, pest surge mid-season, balanced metric management |

### 5.3 Drought Year (Hard)

| Attribute | Value |
|-----------|-------|
| Duration | 90 days |
| Soil | Depleted (low retention, poor starting health) |
| Weather | Arid (12% rain), hot (30°C mean) |
| Water budget | 800mm total |
| Starting conditions | Low crop health, high water stress, degraded soil |
| Key challenges | Water budget constraint, poor soil, high baseline stress |

---

## 6. Graders

### 6.1 Ideal Season

$$\text{Score} = 0.6 \times \text{norm}(\overline{R}, 40, 75) + 0.2 \times \text{norm}(\Delta\text{growth\_rate}, 0, 3.0) + 0.2 \times \text{norm}(\text{trend}, -0.5, 1.0)$$

Emphasizes average reward (60%), growth rate improvement (20%), and positive reward trend (20%).

### 6.2 Variable Weather

$$\text{Score} = 0.35 \times \text{norm}(\overline{R}, 40, 75) + 0.25 \times \text{breadth} + 0.20 \times \text{consistency} + 0.20 \times \max(0, \text{trend})$$

Balances average reward (35%), metric breadth (25% — how many of the 8 metrics improved), consistency (20% — inverse stddev), and trend (20%).

### 6.3 Drought Year

$$\text{Score} = 0.25 \times \text{norm}(\overline{R}, 35, 65) + 0.25 \times \text{norm}(\text{improvement}, 0, 15) + 0.20 \times \text{consistency} + 0.15 \times \text{env\_improvement} + 0.15 \times \text{breadth}$$

Emphasizes improvement over time (25% — last 7 vs first 7 rewards), consistency (20%), environmental sustainability (15%), and breadth (15%).

---

## 7. Agent Architecture

### 7.1 Round 1: LLM Agent

The baseline agent uses GPT-4o-mini via OpenAI API. The system prompt explains metrics, actions, delayed feedback, growth stages, and weather. The agent receives the full observation as structured text and returns a JSON action.

Fallback policy: rule-based reactive farming that responds to current soil moisture, stress levels, pest pressure, and growth stage.

### 7.2 Round 2: NN + LLM Hybrid

| Component | Role |
|-----------|------|
| NN policy (DQN/PPO/SAC) | Fast action selection trained on simulator episodes |
| LLM reasoning | Interprets weather forecasts, explains action rationale |
| Ensemble | NN proposes top-3, LLM selects/adjusts based on context |

### 7.3 Round 2: Training Strategy

- Pre-train NN on 10K+ episodes per scenario
- Use replay buffer with delayed-reward credit assignment
- Curriculum: ideal_season → variable_weather → drought_year
- Evaluation: grade() on held-out seeds

---

## 8. Future Extensions (Post-Hackathon)

### 8.1 Multi-Crop / Multi-Field

- Multiple crop types with different growth curves, sensitivities, and value
- Multi-field management: allocate limited resources across fields
- Crop rotation considerations across seasons

### 8.2 Market Dynamics

- Fluctuating crop prices based on quality, timing, and market conditions
- Economic reward component: maximize revenue vs. minimize costs
- Risk management: insurance decisions, hedging

### 8.3 Real-World Data Integration

- Satellite imagery for growth stage / health estimates
- IoT soil sensors for moisture and nutrient data
- Historical weather data for regional scenario generation
- Yield prediction models calibrated to real outcomes

### 8.4 Advanced Weather Modeling

- Multi-day weather forecasts (probabilistic, not deterministic)
- Seasonal patterns (monsoon, dry season, etc.)
- Climate change scenario modeling
- Spatial weather variation across fields

### 8.5 Sustainability Optimization

- Carbon footprint tracking per action
- Biodiversity impact scoring
- Water use efficiency metrics
- Regulatory compliance (pesticide limits, nutrient runoff caps)

### 8.6 Progressive Signal Enrichment

Not all fields have full sensor coverage. The environment should gracefully degrade when some metrics are unavailable:

**Data Tiers:**

| Tier | Available Metrics | Typical Source |
|------|-------------------|----------------|
| Tier 1 (basic) | crop_health, growth_rate, pest_pressure | Visual inspection |
| Tier 2 (standard) | + soil_health, water_stress, nutrient_stress | Soil sensors |
| Tier 3 (full) | + crop_quality, environmental_score | Lab analysis, sustainability audit |

**Implementation:**
- Each metric gets a binary availability mask
- Missing metrics receive zero weight in the reward function
- Available weights are re-normalized to sum to 1.0
- The agent sees NaN for unavailable metrics in the observation

This allows the same environment to handle scenarios ranging from basic visual-only farming to fully instrumented precision agriculture.

---

## 9. Testing Strategy

### 9.1 Unit Tests

| Area | Key Tests |
|------|-----------|
| Models | Pydantic validation, enum completeness, constraint enforcement |
| Simulator | Delayed effects, pest resistance, waterlogging, weather impact |
| Payoff | Weight structure, baseline reward, sign flipping, clamping |
| Graders | Empty history, score range, ordering (good > bad), determinism |
| Environment | Interface compliance (reset/step/state/grade), end-to-end episodes |

### 9.2 Property Tests

- Determinism: same seed → same episode trajectory
- Reward range: always in [0, 100]
- Grade range: always in [0.0, 1.0]
- Metric clamping: all metrics stay in valid ranges after any action sequence
- Water budget: drought_year never exceeds 800mm + one step margin

### 9.3 Integration Tests

- Full episode runs for all 3 scenarios
- Inference agent produces valid stdout format
- Different seeds produce different trajectories

---

## 10. Deployment

### 10.1 Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "inference.py"]
```

### 10.2 Dependencies

```
pydantic>=2.0
openai>=1.0
numpy>=1.24
pytest>=7.0
```

### 10.3 Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required for LLM agent |
| `OPENAI_MODEL` | Optional, defaults to `gpt-4o-mini` |
