"""Push trained DQN models to Hugging Face Hub."""

import os
import json
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi

# Hugging Face settings
HF_REPO = "Thulasii/Crop-UI"
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Read from environment variable

# Models to push
MODELS_DIR = Path("models")
MODELS = {
    "dqn_model.pth": "Single-task DQN (Ideal Season)",
    "dqn_multitask.pth": "Multi-task DQN (All scenarios)",
    "dqn_ideal_season.pth": "Task-specific: Ideal Season",
    "dqn_variable_weather.pth": "Task-specific: Variable Weather",
    "dqn_drought_year.pth": "Task-specific: Drought Year",
    "dqn_supply_chain_disruption.pth": "Task-specific: Supply Chain Disruption",
    "dqn_regulatory_shift.pth": "Task-specific: Regulatory Shift",
}


def create_model_card(model_name: str, description: str) -> str:
    """Create a model card for the model."""
    
    stats_file = MODELS_DIR / model_name.replace(".pth", "_stats.json")
    stats_content = ""
    
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)
        stats_content = f"""
## Training Statistics
- Episodes: {stats.get('episodes', stats.get('total_episodes', 'N/A'))}
- Average Reward: {stats.get('avg_reward', 'N/A')}
- Final Epsilon: {stats.get('final_epsilon', 'N/A')}
"""
    
    card = f"""---
library_name: transformers
license: mit
tags:
  - reinforcement-learning
  - dqn
  - crop-management
  - agriculture
---

# {model_name}

{description}

## Model Details
- **Framework**: PyTorch
- **Algorithm**: Deep Q-Network (DQN)
- **Environment**: CropEnv (Outcome-Based Crop Management Simulator)
- **Architecture**: Linear network (34 -> 128 -> 128 -> 100 actions)

## Usage

```python
import torch
from train_dqn import SimpleLinearDQN, observation_to_tensor, discrete_to_action

# Load model
model = SimpleLinearDQN(state_size=34, action_size=100)
model.load_state_dict(torch.load('{model_name}'))
model.eval()

# Use for inference
with torch.no_grad():
    state_tensor = observation_to_tensor(obs).unsqueeze(0)
    q_values = model(state_tensor)
    action_idx = q_values.argmax(dim=1).item()
    action = discrete_to_action(action_idx)
```

{stats_content}

## Tasks Supported
- ideal_season: Easy (rewards 20-80)
- variable_weather: Medium (rewards 10-60)
- drought_year: Hard (rewards 5-40)
- supply_chain_disruption: Medium (rewards 10-50)
- regulatory_shift: Hard (rewards 10-50)

## License
MIT

## Citation
```bibtex
@misc{{crop_dqn_{model_name.split('.')[0]},
  title={{Deep Q-Network for Crop Management}},
  author={{Thulasii}},
  year={{2026}},
  howpublished={{Hugging Face Hub}},
  url={{https://huggingface.co/{HF_REPO}}}
}}
```
"""
    return card


def push_models_to_hf(
    repo_id: str = HF_REPO,
    token: str = HF_TOKEN,
    models: Optional[dict] = None,
):
    """Push all trained models to Hugging Face Hub."""
    
    if models is None:
        models = MODELS
    
    if not token:
        print("❌ Error: HF_TOKEN environment variable not set!")
        print("Set it with: export HF_TOKEN='your_token_here'")
        return
    
    print(f"\n{'='*60}")
    print(f"🚀 Pushing Models to Hugging Face Hub")
    print(f"{'='*60}")
    print(f"Repository: {repo_id}")
    print(f"Models: {len(models)}")
    
    api = HfApi(token=token)
    
    # Create repo if it doesn't exist
    try:
        repo_url = api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
        )
        print(f"✅ Repository ready: {repo_url}")
    except Exception as e:
        print(f"⚠️  Repository check: {e}")
    
    # Push each model
    for model_file, description in models.items():
        model_path = MODELS_DIR / model_file
        
        if not model_path.exists():
            print(f"⏭️  Skipping {model_file} (not found)")
            continue
        
        print(f"\n📤 Pushing {model_file}...")
        
        try:
            # Upload model file
            api.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=f"models/{model_file}",
                repo_id=repo_id,
                token=token,
                repo_type="model",
            )
            print(f"  ✅ Model file uploaded")
            
            # Upload stats file if exists
            stats_path = model_path.with_suffix(".json")
            if stats_path.exists():
                api.upload_file(
                    path_or_fileobj=str(stats_path),
                    path_in_repo=f"models/{stats_path.name}",
                    repo_id=repo_id,
                    token=token,
                    repo_type="model",
                )
                print(f"  ✅ Stats file uploaded")
            
            # Create and upload model card
            card_content = create_model_card(model_file, description)
            card_path = f"models/{model_file.replace('.pth', '_README.md')}"
            
            api.upload_file(
                path_or_fileobj=card_content.encode(),
                path_in_repo=card_path,
                repo_id=repo_id,
                token=token,
                repo_type="model",
            )
            print(f"  ✅ Model card uploaded")
            
        except Exception as e:
            print(f"  ❌ Error uploading {model_file}: {e}")
            continue
    
    # Create main README
    main_readme = f"""# Crop Management DQN Models

Deep Q-Network models trained on the CropEnv simulator for crop management optimization.

## Available Models

### General Purpose
- **dqn_model.pth** - Single-task DQN trained on ideal_season
- **dqn_multitask.pth** - Multi-task DQN trained on all scenarios

### Task-Specific (Specialized)
- **dqn_ideal_season.pth** - Optimized for ideal_season scenario
- **dqn_variable_weather.pth** - Optimized for variable_weather scenario
- **dqn_drought_year.pth** - Optimized for drought_year scenario
- **dqn_supply_chain_disruption.pth** - Optimized for supply_chain_disruption scenario
- **dqn_regulatory_shift.pth** - Optimized for regulatory_shift scenario

## Quick Start

```python
import torch
from train_dqn import SimpleLinearDQN

# Load a model
model = SimpleLinearDQN(state_size=34, action_size=100)
model.load_state_dict(torch.load('models/dqn_multitask.pth'))
model.eval()
```

## Architecture

- **Input State**: 34-dimensional observation vector
- **Hidden Layers**: 2 dense layers (128 units each) with ReLU
- **Output**: 100 possible actions (4 irrigation × 5 fertilizer × 5 pest management)
- **Framework**: PyTorch

## Action Space

- **Irrigation**: none, light, moderate, heavy (4 options)
- **Fertilizer**: none, nitrogen, phosphorus, balanced, organic (5 options)
- **Pest Management**: none, scouting, biological, chemical_light, chemical_heavy (5 options)

**Total Actions**: 4 × 5 × 5 = 100

## Task Difficulties

- ⭐ **Easy**: ideal_season
- ⭐⭐ **Medium**: variable_weather, supply_chain_disruption
- ⭐⭐⭐ **Hard**: drought_year, regulatory_shift

## Training Details

- **Algorithm**: Deep Q-Learning with Experience Replay
- **Network**: 2-layer fully connected network
- **Exploration**: Epsilon-greedy (ε starts at 1.0, decays to 0.05)
- **Discount Factor (γ)**: 0.99
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Replay Buffer**: 10,000 transitions

## Performance

All models are normalized to 0-100 scale for fair comparison across tasks.

## License

MIT

## Citation

```bibtex
@misc{{crop_dqn_models,
  title={{Deep Q-Network Models for Crop Management}},
  author={{Thulasii}},
  year={{2026}},
  howpublished={{Hugging Face Hub}},
  url={{https://huggingface.co/{repo_id}}}
}}
```

## Contact

For questions or issues, please open an issue on GitHub.
"""
    
    print(f"\n📝 Creating main README...")
    try:
        api.upload_file(
            path_or_fileobj=main_readme.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
            repo_type="model",
        )
        print(f"  ✅ Main README uploaded")
    except Exception as e:
        print(f"  ❌ Error uploading README: {e}")
    
    print(f"\n{'='*60}")
    print(f"✨ Upload Complete!")
    print(f"{'='*60}")
    print(f"🔗 View your models: https://huggingface.co/{repo_id}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Push DQN models to Hugging Face Hub")
    parser.add_argument("--repo", default=HF_REPO, help="Hugging Face repo ID")
    parser.add_argument("--token", default=HF_TOKEN, help="Hugging Face token")
    parser.add_argument("--all", action="store_true", default=True, help="Push all models")
    parser.add_argument("--multitask-only", action="store_true", help="Push only multitask model")
    parser.add_argument("--single-only", action="store_true", help="Push only single-task model")
    
    args = parser.parse_args()
    
    # Determine which models to push
    models_to_push = MODELS.copy()
    
    if args.multitask_only:
        models_to_push = {
            "dqn_multitask.pth": MODELS.get("dqn_multitask.pth"),
            "dqn_model.pth": MODELS.get("dqn_model.pth"),
        }
    elif args.single_only:
        models_to_push = {
            "dqn_model.pth": MODELS.get("dqn_model.pth"),
        }
    
    push_models_to_hf(
        repo_id=args.repo,
        token=args.token,
        models=models_to_push,
    )
