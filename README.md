# cellitaire-rl
Bot capable of solving a custom made Cellitaire Environment
The goal of this project is to show and explore Reinforcement Learning Algorithms and their applications around a [game called Cellitaire](link). Where you play Solitaire but cards can only be placed by a format determined by a cellular automatae. 

In exploring these algorithms we have created a bot capable of a **X%** winrate over Y games

Here you can see our bot performing after training
[VIDEO OF 100% winrate bot]


Here you can see the bot at initialization acting effectively randomly
[video of brand-new agent 0 training steps. Effectively the random agent]

We achieved these results after X training steps
[SHOW TRAINING GRAPHS HERE. Which will be loss, avg reward, and some hyperparam sweeps maybe (showing regularization vs no regularization crashing)]

The Succesful formula
PPO - Regularization - Scaling Foundation Reward


### This will not be following the Gymnasium spec exactly.
- our goal is to create an environment to succeed in as opposed to establishing another Gym environment to benchmark algorithms
    - Making Rewards editable and configurable as opposed to always the same rewards for an environment to allow reward shaping and other configuration.


## Setup

- Conda env the requirements and then pip install the local files

```
conda env create -f cellitaire-rl-environment.yaml
conda activate cellitaire-rl
pip install -e .
```