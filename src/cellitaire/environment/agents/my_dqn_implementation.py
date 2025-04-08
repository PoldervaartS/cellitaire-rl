from cellitaire.environment.agents.agent import Agent
from cellitaire.environment.cellitaire_env import CellitaireEnv


class MyDQNImplementation(Agent):
    def __init__(self, cellitaire_env: CellitaireEnv, arch_config: dict):
        super().__init__(cellitaire_env)
