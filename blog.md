# Smart Harvest: Training an Agent to Make Farm Decisions Under Uncertainty

## The problem we’re solving

Real farmers make high-stakes decisions every day: how much to irrigate, which fertilizer to apply, and how to respond to pests. Those choices are rarely “one-size-fits-all”—they depend on weather, soil conditions, crop growth stage, resource limits, and real-world constraints like input shortages or new regulations.

When decisions are wrong or inefficient, the outcomes are costly:

- Lower crop yield and quality  
- Profit loss (spending too much for too little return)  
- Wasted resources (water, fertilizer, pesticide)  
- Long-term soil and environmental damage  

Our goal was to build a realistic training ground where an AI agent can learn these tradeoffs the same way a farmer does: by acting, observing consequences, and improving over time.

---

## Our solution: an outcome-based RL environment (OpenEnv)

We built an OpenEnv-compliant crop management environment where an agent manages a simulated field over a multi-day season. Instead of scoring actions with fixed “good/bad” labels, the environment rewards the agent based on measurable outcomes—how the field and crop metrics change after decisions are applied.

---

## What the agent observes (state)

Each day, the agent receives an observation that summarizes the farm’s condition, including 8 key metrics:

- Crop health  
- Growth rate  
- Soil health  
- Water stress  
- Nutrient stress  
- Pest pressure  
- Crop quality  
- Environmental score  

It also sees context like day, total season length, growth stage, and weather (temperature/rainfall), plus recent deltas (what changed since yesterday). After enough days, it can also use 7-day trends (slopes) to understand whether the farm is improving or drifting into trouble.

---

## What the agent can do (actions)

Each step represents a “day” where the agent chooses farm actions across three dimensions:

- **Irrigation:** none / light / moderate / heavy  
- **Fertilizer:** none / nitrogen / phosphorus / balanced / organic  
- **Pest management:** none / scouting / biological / chemical (light/heavy)  

That’s a practical, farmer-like action space: simple categories, but with meaningful consequences.

---

## A Deterministic Environment

Because the simulator uses a seeded pseudo-random generator, running the same scenario with the same seed produces the same weather, dynamics, and outcomes—so training and evaluation are reproducible.

We designed the simulator so the agent can’t just memorize a single policy. The environment includes:

- Hidden dynamics (soil/water/pest parameters the agent must infer indirectly)  
- Weather stochasticity (daily variability + occasional extreme events)  
- Growth stages (different sensitivity during germination vs. fruiting, etc.)  
- Delayed effects (some fertilizers pay off days later, not immediately)  
- Cross-action interactions (e.g., over-irrigation + storms can hurt environmental score)  

This pushes the agent to learn strategies that are robust, not brittle.

---

## Scenarios (tasks): the same farm, different realities

To make learning meaningful, the agent is evaluated under multiple scenarios (tasks) with different pressures—like real seasons where conditions change.

Examples include:

- Ideal season: easier conditions; optimize growth and quality.  
- Variable weather: more volatility; manage pests and stress while keeping performance stable.  
- Drought year: limited water budget; learn water-efficient strategies.  
- Supply chain disruption: a key fertilizer becomes unavailable mid-season.  
- Regulatory shift: chemical pesticides become unavailable after a cutoff date.  

To mirror real decision support, the environment also supports ToolActions—budgeted information calls (they cost money, and results appear in the next observation). This forces the agent to learn when extra information is actually worth paying for.

---

## Supply chain disruption: learning to choose alternatives

In the supply_chain_disruption scenario, the agent encounters a realistic constraint: balanced fertilizer is unavailable during a time window. The agent has to:

- Recognize the constraint (attempting to use balanced fertilizer becomes invalid)  
- Adapt by choosing an alternative (e.g., nitrogen/phosphorus/organic) based on current nutrient stress, soil health, and recent trends  
- Potentially use a tool call (like a soil test) to reduce uncertainty before spending on an imperfect substitute  
- Stay within budget while still protecting long-term yield and quality  

This turns “fertilizer choice” into a planning problem under scarcity, not just a static lookup table.

---

## Regulatory shift: learning safer pest strategies

In regulatory_shift, chemical pesticides become banned after a certain day. The agent learns to:

- Avoid chemical-heavy actions after the cutoff  
- Shift toward scouting and biological controls  
- Manage pest pressure proactively earlier in the season so late-season bans don’t cause a collapse  
- Balance environmental score and crop quality, not just suppress pests at all costs  

This scenario teaches policy adaptation: the same pest level can require different actions depending on the regulatory regime.

---

## What the agent learned (the takeaway)

Across tasks, the agent learns three core lessons that look a lot like real farm management:

### Decisions must be state-dependent

Irrigation, fertilizer, and pest control only make sense relative to current stress levels, soil moisture, weather, and growth stage.

---

### Short-term fixes can cause long-term damage

Aggressive actions can boost one metric while harming others (e.g., reducing pests but damaging environmental score, or pushing growth while degrading soil health).

---

### Constraints change the “optimal” policy

When resources are limited (drought water budget), supplies disappear (fertilizer shortage), or rules change (pesticide bans), the agent must learn a new best response—often by using tools strategically and relying more on trends and delayed-effect reasoning.