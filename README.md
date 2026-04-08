# Crop Management OpenEnv — Outcome-Based Simulator

An OpenEnv-compliant crop management simulator where rewards are driven by
**measured field/crop metric changes** (outcomes), not predetermined
action-quality scores. An LLM agent manages a simulated farm field over a
multi-day growing season, choosing daily irrigation, fertilization, and pest
management actions while observing how those actions change 8 crop/field
metrics under hidden soil, weather, and pest dynamics.

## Architecture

```
inference.py          LLM agent (prescriptive decision-tree prompt)
app.py                FastAPI server for OpenEnv validation (port 7860)
openenv.yaml          Task metadata for OpenEnv tooling
crop_env/
  env.py              CropEnv — OpenEnv interface (reset/step/state/grade)
  models.py           Pydantic models (Action, CropMetrics, Observation, etc.)
  simulator.py        Hidden transition dynamics (FieldModel response)
  payoff.py           Blended reward: 70% metric deltas + 30% state quality
  graders.py          Task graders (multi-criteria, 0.0–1.0 scores)
  scenarios.py        FieldModel configs, weather generation, growth stages
tests/
  test_env.py         Environment interface + end-to-end tests
  test_payoff.py      Reward computation tests
  test_simulator.py   Transition dynamics + delayed effects tests
  test_graders.py     Grader tests (76 tests total)
```

## Outcome Metrics (8)

| Metric | Range | Direction | What It Captures |
|--------|-------|-----------|------------------|
| Crop Health | 0–100 | Higher is better | Overall plant vigor |
| Growth Rate | 0–10 cm/day | Higher is better | Daily growth rate |
| Soil Health | 0–100 | Higher is better | Combined soil quality (changes slowly) |
| Water Stress | 0–100 | Lower is better | Water deficit or excess |
| Nutrient Stress | 0–100 | Lower is better | Nutrient deficiency or toxicity |
| Pest Pressure | 0–100 | Lower is better | Pest/disease threat level |
| Crop Quality | 0–100 | Higher is better | Expected harvest grade |
| Environmental Score | 0–100 | Higher is better | Sustainability and eco-impact |

## Action Space (100 combinations)

| Dimension | Options |
|-----------|---------|
| Irrigation | none, light (5mm), moderate (15mm), heavy (30mm) |
| Fertilizer | none, nitrogen, phosphorus, balanced, organic |
| Pest Management | none, scouting, biological, chemical_light, chemical_heavy |

## Tasks

| Task | Days | Difficulty | Key Challenge |
|------|------|------------|--------------|
| ideal_season | 60 | Easy | Optimize growth rate and crop quality in favorable conditions |
| variable_weather | 90 | Medium | Balance all 8 metrics under weather variability + pest surges |
| drought_year | 90 | Hard | Maximize outcomes with 800mm water budget + depleted soil |

## Reward Function

The reward is **blended** from two components:

$$R = 0.7 \times R_{\text{delta}} + 0.3 \times R_{\text{state}}$$

- **Delta component (70%)**: Rewards metric improvements. Each metric delta is
  normalized by a scale representing "excellent daily change," weighted by
  scenario-specific priorities, and centered at 50 (no change = 50).
- **State quality component (30%)**: Rewards maintaining good absolute metric
  values. Each metric is normalized to [0, 1] and weighted by the same scenario
  weights. This prevents the agent from being penalized for stability once
  metrics are already high.

### Scenario Weights

| Metric | ideal_season | variable_weather | drought_year |
|--------|-------------|-----------------|-------------|
| crop_health | 0.20 | 0.15 | 0.20 |
| growth_rate | 0.25 | 0.15 | 0.10 |
| soil_health | 0.05 | 0.15 | 0.10 |
| water_stress | 0.05 | 0.15 | 0.25 |
| nutrient_stress | 0.10 | 0.10 | 0.10 |
| pest_pressure | 0.10 | 0.15 | 0.10 |
| crop_quality | 0.20 | 0.10 | 0.05 |
| environmental_score | 0.05 | 0.05 | 0.10 |

## Graders

Each task has a multi-criteria grader that produces a 0.0–1.0 score:

**ideal_season**: 60% avg reward + 20% growth_rate improvement + 20% reward trend

**variable_weather**: 35% avg reward + 25% metric breadth (fraction of 8 metrics
that improved) + 20% consistency (inverse stddev) + 20% positive trend

**drought_year**: 25% avg reward + 25% last-7-vs-first-7 improvement + 20%
consistency + 15% environmental_score improvement + 15% metric breadth

## Key Design Features

- **Hidden dynamics**: Each scenario has ~14 hidden FieldModel parameters (moisture retention, pest growth rate, etc.) the agent must learn through experience
- **Delayed feedback**: Synthetic fertilizer peaks at day+2, organic decomposes over days 3–10
- **Pest resistance**: Chemical pesticides lose effectiveness with repeated use (cumulative resistance buildup)
- **Growth stages**: Germination → Vegetative → Flowering → Fruiting → Maturity, each with different water/nutrient sensitivity and growth multipliers
- **Weather stochasticity**: Temperature/rainfall variation, 5% daily chance of extreme events (heat waves, storms)
- **Cross-action interactions**: Waterlogging from over-irrigation, nutrient burn from over-fertilization, chemical runoff during storms

## Agent

The baseline agent uses an LLM (gpt-4o-mini) with a prescriptive decision-tree
prompt. The system prompt encodes exact conditional rules for irrigation,
fertilization, and pest management based on soil moisture, rainfall, growth
stage, and scenario. Temperature is set to 0 for deterministic output. A
rule-based fallback policy handles cases where the LLM is unavailable.

## Baseline Scores

| Task | Score | Avg Reward | Reward Stddev |
|------|-------|-----------|---------------|
| ideal_season | 0.4516 | 63.42 | 10.34 |
| variable_weather | 0.2489 | 36.33 | 13.61 |
| drought_year | 0.4388 | 14.12 | 7.05 |

Scores demonstrate clear difficulty progression: ideal > drought > variable,
consistent with the task design.

## Running

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
python inference.py
```

Or with HuggingFace-hosted models:

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=your_token
python inference.py
```

## Testing

```bash
pytest tests/ -v    # 76 tests
```

## Deployment

```bash
docker build -t crop-outcome .
docker run -p 7860:7860 crop-outcome
```

Verify the Space is live:

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "ideal_season"}'
```
