# Stable Error-seeking Exploration

This repository contains code to reproduce all results of the paper

**Learning to Explore in Diverse Reward Settings via Temporal-Difference-Error Maximization**

# Installation

Clone the repository and install all requirements from the `requirements.txt`. We recommend using a virtual environment.

```bash
git clone https://github.com/Sebastian-Griesbach/SEE.git
cd SEE
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

# Usage

Specific results can be repeated by executing `run_experiment.py`.

```bash
python see/run_experiment.py
```

Or directly in your IDE.

To recreate any of the experiments in the paper, open `run_experiment.py`, scroll down to the bottom and modify:

```python
    if __name__ == "__main__":
        setup_and_run_experiment(
            environment_name="Pendulum-v1",
            reward_setting="adverse",
            method="SAC+SEE",
        )
```

You can choose between:

- **environment_name**:

  - `"Pendulum-v1"`
  - `"LocalOptimumCar-v0"`
  - `"HalfCheetah-v5"`
  - `"Ant-v5"`
  - `"Hopper-v5"`
  - `"Swimmer-v5"`
  - `"FetchPickAndPlace-v4"`
  - `"LargePointMaze-v3"`

- **reward_setting**:

  - `"dense"`
  - `"sparse"`
  - `"adverse"`

- **method**:
  - `"SAC"`
  - `"TD3"`
  - `"SAC+SEE"`
  - `"TD3+SEE"`
  - `"SAC+SEE w/o maximum update"`
  - `"SAC+SEE w/o conditioning"`
  - `"SAC+SEE w/o mixing"`
  - `"TD3+SEE w/o maximum update"`
  - `"TD3+SEE w/o conditioning"`
  - `"TD3+SEE w/o mixing"`

All other settings will be automatically set to match the settings of the experiments in the paper.
The code is optimized for readability and is fully documented with docstrings.

[wandb](https://github.com/wandb/wandb) is used for logging which requires an account to use. If you want to change that, there are three occurrences of wandb calls in `run_experiment.py` which you can replace with an alternative. It is not used outside of this file.

Most environments are included in the `environments` folder. But the [Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics) environments are based on a [fork](https://github.com/Sebastian-Griesbach/Gymnasium-Robotics) of that repository, which alters the reward functions of the relevant environments.

SAC+SEE and TD3+SEE are implemented using the [Athlete API](https://github.com/Sebastian-Griesbach/Athlete). For the baseline SAC and TD3 the Athlete implementations are used directly.
