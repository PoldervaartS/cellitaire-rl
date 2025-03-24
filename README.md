# cellitaire-rl
Reinforcement Learning Environment for the game Cellitaire

### This will not be following the Gymnasium spec exactly because I don't like it.
- The goal is to solve/best-do cellitaire, not make it a benchmark environment. So custom reward shaping and other configuration options are here.



## Setup

- Conda env the requirements and then pip install the local files

```
conda env create -f cellitaire-rl-environment.yaml
conda activate cellitaire-rl
pip install -e .
```