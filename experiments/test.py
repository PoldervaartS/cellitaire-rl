from cellitaire.environment.agents.PPOAgent import Agent
from cellitaire.environment.cellitaire_env import CellitaireEnv
from cellitaire.environment.rewards.reward import *
from cellitaire.environment.rewards.foundation_rewards import *
import numpy as np

from process import *



normalize_reward = False
save_model = True
if __name__ == '__main__':
    board_rows = 7
    board_cols = 12
    num_reserved = 6
    test_reward = CombinedReward([
        #PlacedCardInFoundationReward(weight=6),
        WinReward(weight=100, rows=board_rows, cols=board_cols),
        #ConstantReward(weight=0.5),
        ScalingPlacedCardInFoundationReward(weight=1, rows=board_rows, cols=board_cols),
        #PlayedLegalMoveReward(weight=1, rows=board_rows, cols=board_cols, num_reserved = num_reserved),
        #PeriodicPlacedCardInFoundationReward(weight=4, reward_period=3),
        #CreatedMovesReward(weight=1, num_reserved=num_reserved, foundation_count_dropoff=30)
    ])
    env = CellitaireEnv(test_reward, rows=board_rows, cols=board_cols, num_reserved=num_reserved, max_moves=1200, max_illegal_moves=1200)
    env.action_space.n = board_rows * board_cols
    #env.render()

    #agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
    #                alpha=alpha, n_epochs=n_epochs, 
    #                input_dims=env.observation_space.shape)

    N = 10000
    batch_size = 2000
    n_epochs = 5
    alpha = 4e-6
    gamma = 0.99
    gae_lambda = 0.95
    checkpoint_dir = 'tmp/testing_parallel_2'
    agent = Agent(n_actions=env.action_space.n, 
        input_dims=(board_rows * board_cols * 4 + 6,), batch_size=batch_size, fc1_actor=2048, fc2_actor=2048, fc1_critic=4096, fc2_critic=4096,
                    alpha=alpha, n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda, checkpoint_dir=checkpoint_dir)

    trainer = AgentTrainer(agent, env, checkpoint_dir=checkpoint_dir, steps_to_post=N, batch_size=batch_size, collector_processes=3, normalize_reward=False, performance_games_to_sim=100, steps_between_performance_checks=3)
    trainer.start_training()