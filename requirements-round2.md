# Crop Management OpenEnv — Round 2 Requirements

## 0. Hackathon Context

**Event**: OpenEnv Hackathon Round 2 (onsite, 25–26 April 2026)

**Selected Theme**: Theme #3.1 — World Modeling (Professional Tasks)
**Bonus Sub-theme**: Scaler AI Labs — Multi-App RL Environment for Enterprise Workflows

**Why this theme fits the existing codebase:**
- Partial observability is already built in via the hidden `FieldModel` (14 parameters the agent cannot see directly)
- Multi-step causal workflows with delayed effects are already implemented (fertilizer delay 2–10 days, pest resistance buildup)
- Real professional domain (agriculture) with measurable outcomes
- Least refactoring risk — extends what exists rather than replacing it
- Sub-theme bonus achieved by adding multi-tool API layer and economic dimension (budget, costs, profit)

**Pitch angle**: "We built an agricultural enterprise simulator where an LLM agent must call external tools (soil tests, weather forecasts, market price feeds), manage a seasonal budget, and adapt to mid-season disruptions — all under partial observability. The agent has to do real work, not exploit shortcuts."

---

**Judging Criteria Alignment**

| Criterion | Weight | How we address it |
|---|---|---|
| Environment Innovation | 40% | Tool-discovery layer, economic reward, enterprise scenarios with mid-season disruptions |
| Storytelling | 30% | Before/after reward curves (rule-based baseline → SFT → GRPO), clear 3-min demo flow |
| Showing Reward Improvement | 20% | SFT→GRPO training pipeline with logged reward curves per training step |
| Reward & Training Script | 10% | Direct `env.step()` reward signal into HF TRL `GRPOTrainer`, minimal Colab script |

**Mandatory Deliverables (from organizers):**
- [ ] OpenEnv-compliant environment (latest release)
- [ ] Minimal training script using Unsloth or HF TRL in Colab
- [ ] Mini-blog on HuggingFace or <2 min YouTube video
- [ ] Environment hosted on Hugging Face Spaces

---

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

## 7. New Scenarios (Round 2)

Two new enterprise-focused scenarios are added alongside the existing three. They target the Scaler AI Labs sub-theme by introducing mid-season disruptions that force adaptive replanning.

### 7.1 `supply_chain_disruption` (Hard)

| Attribute | Value |
|-----------|-------|
| Duration | 90 days |
| Soil | Average fertility |
| Weather | Normal (25% rain, 26°C) |
| Water budget | Unlimited |
| Disruption | Preferred fertilizer type (balanced) unavailable from day 30–60 |
| Key challenge | Agent must discover the unavailability via tool call, pivot to organic/nitrogen, manage delayed nutrient stress without its usual tool |

**Grader**: Weighted average reward (40%) + metric breadth (25%) + adaptation score — reward trend after disruption vs before (35%)

### 7.2 `regulatory_shift` (Hard)

| Attribute | Value |
|-----------|-------|
| Duration | 90 days |
| Soil | Good fertility |
| Weather | Variable (20% rain, 28°C mean) |
| Water budget | Unlimited |
| Disruption | Chemical pesticides (chemical_light, chemical_heavy) banned after day 30 |
| Key challenge | Agent must detect the regulation change via tool call, pivot to biological pest control without losing ground on pest_pressure |

**Grader**: Weighted average reward (40%) + compliance score — zero chemical use post-ban (30%) + pest_pressure trend after pivot (30%)

---

## 8. Tool Discovery Layer (Round 2 Addition)

This is the core differentiator for Theme #3.1. The agent must interact with external tool APIs to reveal hidden world state — mirroring real enterprise workflows where information has a cost.

### 8.1 New Tool Endpoints (`app.py`)

Three new `POST /tools/*` endpoints are added to the FastAPI server:

| Endpoint | What it reveals | Cost |
|---|---|---|
| `POST /tools/soil_test` | 3–4 of the 14 hidden `FieldModel` parameters (moisture_retention, waterlogging_threshold, nutrient_burn_threshold) | 1 time step deducted from episode + $50 budget |
| `POST /tools/weather_forecast` | Probabilistic 7-day rainfall outlook (mean ± std, probability of extreme event) | $30 budget |
| `POST /tools/market_prices` | Current commodity price per unit of crop_quality (changes weekly by scenario) | $10 budget |

### 8.2 Tool Call Mechanics

- Tool calls are a **distinct action type** — the agent either takes a farm action (irrigation/fertilizer/pest) OR calls a tool in a given step, not both
- Tool results are appended to the next step's observation as `tool_result: {...}`
- Calling a tool on a step where budget is exhausted returns `{"error": "budget_exhausted"}`
- The agent must reason about the ROI of each tool call vs. acting directly

### 8.3 New Models (`crop_env/models.py`)

```python
class ToolCallType(str, Enum):
    SOIL_TEST = "soil_test"
    WEATHER_FORECAST = "weather_forecast"
    MARKET_PRICES = "market_prices"

class ToolAction(BaseModel):
    tool: ToolCallType

class ToolResult(BaseModel):
    tool: ToolCallType
    result: dict
    cost: float

class BudgetState(BaseModel):
    season_budget: float          # Total budget for the episode
    spent: float                  # Amount spent so far
    remaining: float              # season_budget - spent
    intervention_costs: dict      # Per-action costs this episode
```

### 8.4 Changes to `crop_env/env.py`

- `step()` accepts either `Action` or `ToolAction`
- `BudgetState` tracked in episode state
- `state()` includes `budget` field
- Tool call results cached until next farm action step

---

## 9. Economic Dimension (Round 2 Addition)

Adding a profit component to the reward grounds the environment in enterprise reality and enables agents that learn resource efficiency — not just crop metric maximization.

### 9.1 Per-Action Costs (`crop_env/scenarios.py`)

| Action | Cost |
|---|---|
| Irrigation: light | $10 |
| Irrigation: moderate | $25 |
| Irrigation: heavy | $50 |
| Fertilizer: nitrogen | $30 |
| Fertilizer: phosphorus | $30 |
| Fertilizer: balanced | $45 |
| Fertilizer: organic | $20 |
| Pest: scouting | $5 |
| Pest: biological | $25 |
| Pest: chemical_light | $35 |
| Pest: chemical_heavy | $60 |

### 9.2 Revenue Function

At episode end, revenue is computed as:

$$\text{Revenue} = \text{crop\_quality}_{\text{final}} \times \text{yield\_estimate} \times \text{market\_price}$$

Where `yield_estimate` is derived from cumulative `growth_rate` across the episode, and `market_price` is the value returned by the `market_prices` tool (or scenario default if never called).

### 9.3 Economic Reward Component (`crop_env/payoff.py`)

The existing blended reward formula is extended:

$$R_{\text{total}} = 0.80 \times R_{\text{agronomy}} + 0.20 \times R_{\text{economic}}$$

Where:
- $R_{\text{agronomy}}$ = the existing blended delta + state quality reward (unchanged)
- $R_{\text{economic}}$ = normalized profit margin for the episode, computed at end and backfilled per step

This keeps the existing reward signal dominant while adding a profit signal that differentiates budget-efficient agents from wasteful ones.

### 9.4 Observation Update

`BudgetState` is appended to `Observation` so the agent sees remaining budget and spend rate at every step.

---

## 10. Updated Graders (`crop_env/graders.py`)

Existing graders for `ideal_season`, `variable_weather`, and `drought_year` are unchanged. New graders for round 2 scenarios:

| Scenario | Grader Components |
|---|---|
| `supply_chain_disruption` | avg_reward (40%) + metric_breadth (25%) + post-disruption reward trend (35%) |
| `regulatory_shift` | avg_reward (40%) + compliance_score — zero chemical post-ban (30%) + pest_pressure trend after pivot (30%) |

Updated `openenv.yaml` adds both new tasks with `difficulty: hard`.

---

## 11. Agent Redesign (`inference.py`)

### 11.1 Problems with the Round 1 Agent

| Problem | Impact |
|---|---|
| `SYSTEM_PROMPT` contains hardcoded conditional rules | LLM executes rules, doesn't reason — no better than the fallback |
| `build_user_message()` passes only `soil_moisture` and `rainfall` | 7/8 metrics, all deltas, growth stage, trends ignored |
| Temperature = 0 | Fully deterministic — no exploration during training rollouts |
| Fallback encodes identical logic to system prompt | Can't demonstrate LLM improvement over baseline |
| Reward never fed back to agent | No self-correction, no learning signal in context |

### 11.2 New Agent Design

**System prompt** (`SYSTEM_PROMPT`): Replace rule list with domain framing:
- Explain what each metric means and which direction is good
- Explain key tradeoffs (over-irrigation vs water stress, chemical resistance buildup, organic fertilizer delay)
- Explain growth stage sensitivity changes
- Explain tool call ROI (when is a soil test worth the step cost?)
- Instruct agent to reason briefly before outputting JSON (`"reasoning": "...", "action": {...}`)
- Remove all hardcoded conditionals

**Observation message** (`build_user_message()`): Include:
- All 8 current metric values + all 8 deltas (the signals the environment was designed around)
- Current growth stage
- 7-day trends (slope per metric) when available
- Last 3 step rewards (closes feedback loop)
- Budget remaining + spend rate
- Last tool result (if any)
- Current step / total days

**Temperature**: `0.5` during training rollouts (exploration), `0.0` for eval/grading

**Fallback policy**: Kept as rule-based reactive farming — clearly distinct from the LLM policy. Used only when JSON parse fails, not as a policy equivalent.

**Tool calling**: Agent outputs either a farm action JSON or a tool call JSON:
```json
{"action_type": "farm", "irrigation": "moderate", "fertilizer": "organic", "pest_management": "biological"}
{"action_type": "tool", "tool": "soil_test"}
```

---

## 12. Training Pipeline

### 12.1 Strategy: SFT → GRPO

Training proceeds in two stages. This mirrors the DeepSeek-R1 approach (cold-start SFT, then RL).

**Why two stages:**
- GRPO from scratch on a small LLM produces noisy, slow convergence because the model doesn't know the output format or domain
- SFT teaches format and basic domain policy quickly (30 min on free Colab)
- GRPO then pushes the model beyond the ceiling of the rule-based demonstrations

**Expected reward progression:**
| Checkpoint | ideal_season | variable_weather | drought_year |
|---|---|---|---|
| Rule-based baseline | ~0.45 | ~0.25 | ~0.44 |
| After SFT | ~0.48 | ~0.30 | ~0.45 |
| After GRPO | ~0.60 | ~0.50 | ~0.55 |

### 12.2 Stage 1: SFT Dataset Generation

**Script**: `generate_sft_data.py`

1. Run the rule-based fallback policy for 100 episodes across all 5 scenarios
2. Keep only steps where `reward > 60` (good decisions)
3. Format each as `(observation_text, action_json)` pair
4. Save as JSONL: `sft_data.jsonl` (~3,000–5,000 training examples)

**SFT training** (`train_sft.py`):
- Model: `Qwen/Qwen2.5-1.5B-Instruct` (fits free Colab T4 with 4-bit quantization)
- Trainer: `trl.SFTTrainer`
- LoRA: rank=16, alpha=32, target_modules=["q_proj","v_proj"]
- Epochs: 2–3
- Runtime: ~25 min on free Colab T4

### 12.3 Stage 2: GRPO Fine-tuning

**Script**: `train_grpo.py` (the Colab-runnable script for submission)

```
For each training step:
  1. Sample a random scenario and reset the environment
  2. For N=8 steps in the episode, generate G=4 candidate actions from the LLM
  3. Run each candidate through the environment (forked state)
  4. Score each with env.step() → reward
  5. GRPO update: push model toward above-average candidates
  6. Log reward per training step → reward curve
```

**Configuration:**
- Base model: SFT checkpoint from Stage 1
- Trainer: `trl.GRPOTrainer`
- Unsloth 4-bit quantization for memory efficiency
- Training steps: 300–500 (sufficient for visible improvement on free Colab)
- Batch size: 4 (G=4 candidates per step)
- Learning rate: 5e-5

**Reward signal**: Direct `result.reward.total` from `env.step()` — no reshaping. The environment's blended formula is the GRPO objective.

**Output**: LoRA adapter saved to `./grpo_adapter/`, reward curve logged to `rewards.csv`

### 12.4 Inference with Trained Model (`inference.py`)

At eval time, `inference.py` loads the GRPO adapter on top of the SFT Qwen2.5-1.5B checkpoint:

```python
MODEL_NAME = os.environ.get("MODEL_NAME", "grpo_adapter")  # local path or HF repo
```

The same `run_task()` loop is used — no changes needed. The trained model plays the environment and its scores are compared against the rule-based baseline.

---

## 13. File-by-File Change Summary

| File | Change Type | What Changes |
|---|---|---|
| `crop_env/models.py` | Extend | Add `ToolCallType`, `ToolAction`, `ToolResult`, `BudgetState`; add `tool_result` and `budget` fields to `Observation` |
| `crop_env/env.py` | Extend | `step()` accepts `Action \| ToolAction`; track `BudgetState`; cache tool results; include budget in `state()` |
| `crop_env/scenarios.py` | Extend | Add per-action costs to each scenario; add `supply_chain_disruption` and `regulatory_shift` scenario configs |
| `crop_env/simulator.py` | Extend | Handle disruption flags (fertilizer unavailability, chemical ban) based on scenario and current day |
| `crop_env/payoff.py` | Extend | Add economic reward component (20% weight); compute per-step cost deduction |
| `crop_env/graders.py` | Extend | Add graders for `supply_chain_disruption` and `regulatory_shift` |
| `app.py` | Extend | Add `POST /tools/soil_test`, `POST /tools/weather_forecast`, `POST /tools/market_prices` endpoints |
| `inference.py` | Rewrite | New domain-reasoning system prompt; rich observation message; tool call support; temperature control; distinct fallback |
| `openenv.yaml` | Extend | Add 2 new tasks |
| `generate_sft_data.py` | New | Run rule-based policy, filter high-reward steps, save JSONL dataset |
| `train_sft.py` | New | HF TRL SFTTrainer on Qwen2.5-1.5B-Instruct with LoRA |
| `train_grpo.py` | New | HF TRL GRPOTrainer with Unsloth; env.step() as reward; reward curve logging |
| `Dockerfile` | Update | Port 7860, HF Spaces SDK: docker compatibility |
| `requirements.txt` | Update | Add `trl`, `unsloth`, `datasets` |

---

## 14. Testing Strategy

### 14.1 Existing Tests (must not break)

All 76 existing tests in `tests/` must pass after every phase of changes:

| File | Tests | What's covered |
|------|-------|----------------|
| `test_env.py` | 32 | Interface compliance, end-to-end episodes |
| `test_payoff.py` | 11 | Reward computation |
| `test_simulator.py` | 13 | Transition dynamics |
| `test_graders.py` | 13 | Grader scoring |

### 14.2 New Tests Required

| Test | File | What it validates |
|---|---|---|
| `test_tool_endpoints` | `tests/test_tools.py` | Each tool endpoint returns correct schema, deducts correct budget |
| `test_budget_tracking` | `tests/test_tools.py` | Budget exhaustion returns error, never goes negative |
| `test_disruption_scenarios` | `tests/test_env.py` | supply_chain_disruption blocks balanced fertilizer post-day-30; regulatory_shift blocks chemical post-day-30 |
| `test_new_graders` | `tests/test_graders.py` | New graders return [0,1], good agent > bad agent |
| `test_economic_reward` | `tests/test_payoff.py` | Economic component adds to total, high-spend episodes score lower |

### 14.3 Reward Improvement Verification

Before submission, run:

```bash
# Baseline (rule-based)
python run_inference.py --agent=fallback

# After SFT
python run_inference.py --agent=sft

# After GRPO
python run_inference.py --agent=grpo
```

Scores should strictly increase across the three checkpoints on at least 2 of 5 scenarios.

---

## 15. Deployment

### 15.1 HF Spaces (mandatory)

Update `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

`README.md` front-matter for HF Spaces:
```yaml
---
title: Crop Outcome OpenEnv
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
---
```

### 15.2 Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `HF_TOKEN` | HuggingFace token for model inference | Yes |
| `API_BASE_URL` | OpenAI-compatible base URL | Yes |
| `MODEL_NAME` | Model to use (local path or HF repo ID) | Yes |
| `SEED` | Episode seed for reproducibility | No (default: 42) |

---

## 16. Implementation Order

Work in this sequence to minimize risk of breaking existing functionality:

| Phase | Files | Depends on | Validates with |
|---|---|---|---|
| **0 — Agent redesign** | `inference.py` | Nothing | Manual run_inference.py; LLM score > baseline |
| **1a — New models** | `crop_env/models.py` | Nothing | Existing tests still pass |
| **1b — Budget + tool env** | `crop_env/env.py`, `app.py` | Phase 1a | `test_tools.py` |
| **1c — Costs in scenarios** | `crop_env/scenarios.py` | Phase 1a | `test_payoff.py` |
| **2 — Economic reward** | `crop_env/payoff.py` | Phase 1c | `test_payoff.py` |
| **3 — New scenarios** | `crop_env/scenarios.py`, `crop_env/simulator.py` | Phase 1c | `test_disruption_scenarios` |
| **4 — New graders** | `crop_env/graders.py`, `openenv.yaml` | Phase 3 | `test_graders.py` |
| **5 — SFT data + training** | `generate_sft_data.py`, `train_sft.py` | Phase 0 | Reward curve vs baseline |
| **6 — GRPO training** | `train_grpo.py` | Phase 5 | Reward curve vs SFT |
| **7 — Deployment** | `Dockerfile`, `README.md`, `requirements.txt` | All phases | HF Space /health endpoint |
