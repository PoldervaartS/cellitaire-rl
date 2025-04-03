# cellitaire-rl
Bot capable of solving a custom made Cellitaire Environment
The goal of this project is to show and explore Reinforcement Learning Algorithms and their applications around a [game called Cellitaire](https://marcosd.itch.io/cellitaire) made by Marcos Donnantuoni. Cellitaire is similar to Solitaire with the exception that in lieu of tableaus, cards are put in play through the rules of cellular automatae.
In exploring these algorithms we have created a bot capable of a **X%** winrate over Y games

Here you can see a sample of our current best performing AI playing the game in our test environment
https://github.com/user-attachments/assets/45e9ddd2-56ad-4036-9811-23bf3e68ddea


Here you can see the bot at initialization acting effectively randomly
[video of brand-new agent 0 training steps. Effectively the random agent]

We achieved these results after ~38000 training steps
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
