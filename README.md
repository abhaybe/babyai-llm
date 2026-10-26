# Practical LLM-Guided RL: Help-When-Stuck and Automated Reward Shaping

Two low-cost ways to use an LLM inside reinforcement learning without replacing the RL agent:

1. **Help-when-stuck:** when the agent goes 50 steps without reward, an LLM suggests a single action (at most 3 times per episode). It fires on fewer than 2% of steps.
2. **Automated reward shaping:** at the start of training, an LLM writes a reward-shaping function from a task description and the environment API. The code is checked (AST parse, blocked imports, sandboxed test call) and then runs every step with no further API calls.

The agent is a standard PPO policy (Stable-Baselines3) on sparse-reward [MiniGrid](https://github.com/Farama-Foundation/Minigrid) tasks.

📄 **Full write-up:** [report (PDF)](report.pdf)

## Results

Success rate after 300K environment steps (50 evaluation episodes, deterministic policy):

| Environment | PPO baseline | + Help-when-stuck | + Reward-shaping code | Combined |
|---|---|---|---|---|
| Empty-8x8 | 100% (150K steps to converge) | 100% (120K steps) | 100% | 100% |
| DoorKey-8x8 | 5% | 82% | 86% | **90%** |
| MultiRoom-N6 | 0% | 0% | 28% | **30%** |

- The two methods are complementary: help-when-stuck works on DoorKey, where stuck states are easy to interpret (facing a locked door, key in view), but fails on MultiRoom; generated reward shaping supplies the dense signal MultiRoom needs.
- LLM cost is about $0.05–0.10 per training run with `gpt-4o-mini`.

## Repository layout

```
configs/                 # one YAML per experimental condition (DoorKey by default)
  baseline_doorkey.yaml  # PPO only
  help_mode.yaml         # + help-when-stuck
  code_gen.yaml          # + LLM-generated reward shaping
  combined.yaml          # both
src/
  train.py               # training entry point
  eval.py                # evaluate saved models
  llm_help_mode.py       # help-when-stuck wrapper
  llm_code_gen.py        # reward-shaping code generation + validation
  llm_wrapper.py         # LLM client and caching
  env_wrapper.py         # MiniGrid environment construction
  minigrid_extractor.py  # small CNN feature extractor for 7x7 observations
```

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...   # needed for help-mode and code-gen configs only
```

## Usage

Train one condition (from the repo root):

```bash
python src/train.py --config configs/combined.yaml
```

Outputs (checkpoints, best model, TensorBoard logs) go to `output/<exp_name>_<timestamp>/`. To run another environment, change `env_name` in the config (for example `MiniGrid-MultiRoom-N6-v0`).

Evaluate a trained model:

```bash
python src/eval.py --model output/<run>/best_model/best_model --config configs/combined.yaml --n_episodes 50
python src/eval.py --list    # list available models
```

## Limitations and next steps

- Results are from single training runs; multiple seeds are needed to separate close conditions (82% vs. 86% vs. 90%).
- The shaped reward is added directly rather than in potential-based form (F = γΦ(s′) − Φ(s)), so it can in principle change the optimal policy; switching to potential-based shaping is the natural fix.
- Help-when-stuck overrides the policy's action, so those transitions are off-policy for PPO.

Course project for MIT 6.7920 (Deep Reinforcement Learning), Fall 2025.
