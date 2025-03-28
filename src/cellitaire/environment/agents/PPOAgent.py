import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.distributions.categorical import Categorical

from cellitaire.environment.agents.PPOMemory import PPOMemory
from cellitaire.environment.cellitaire_env import CellitaireEnv

class ActorNetwork(nn.Module):
    def __init__(
        self, 
        n_actions, 
        input_dims, 
        alpha,
        chkpt_dir='tmp/ppo', 
        num_embeddings=53, 
        embedding_dim=30, 
        embeddings_in_state=85,
        num_hidden_layers=1,
        hidden_dim=256
    ):
        super(ActorNetwork, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.embeddings_in_state = embeddings_in_state

        # Create embedding layer only if embeddings_in_state > 0.
        if embeddings_in_state > 0:
            self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim).to(self.device)
            input_layer_dim = input_dims[0] - embeddings_in_state + (embeddings_in_state * embedding_dim)
        else:
            self.embedding_layer = None
            input_layer_dim = input_dims[0]

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(input_layer_dim, hidden_dim),
            nn.ReLU()
        )
        
        for _ in range(num_hidden_layers):
            self.actor.append(nn.Linear(hidden_dim, hidden_dim))
            self.actor.append(nn.LayerNorm((hidden_dim)))
        
        self.actor.append(nn.Linear(hidden_dim, n_actions))
        self.actor.append(nn.Softmax(dim=-1))
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)
        self.n_actions = n_actions

    def forward(self, state):
        if self.embeddings_in_state > 0:
            embeddings = self.embedding_layer(state[:, :self.embeddings_in_state].int()).view(state.shape[0], 1, -1)
            state = state.view(state.shape[0], 1, -1)[:, :, self.embeddings_in_state:]

            
            # Concatenate the remaining state features with the flattened embeddings.
            new_state = torch.cat((state, embeddings), dim=2)
        else:
            new_state = state.view(state.shape[0], 1, -1)
        
        # Pass the processed state through the actor network.
        logits = self.actor(new_state)
        return logits

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(
        self, 
        input_dims, 
        alpha, 
        num_hidden_layers=1, 
        hidden_dim=256, 
        chkpt_dir='tmp/ppo'
    ):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')

        self.critic = nn.Sequential(
            nn.Linear(*input_dims, hidden_dim),
            nn.ReLU()
        )
        
        for _ in range(num_hidden_layers):
            self.critic.append(nn.Linear(hidden_dim, hidden_dim))
            self.critic.append(nn.LayerNorm((hidden_dim)))


        self.critic.append(nn.Linear(hidden_dim, 1))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class PPOAgent:
    def __init__(
        self, 
        n_actions, 
        input_dims, 
        gamma=0.99, 
        alpha=0.0003, 
        gae_lambda=0.95,
        policy_clip=0.2, 
        batch_size=64, 
        n_epochs=10,
        num_hidden_layers_actor=1,
        hidden_dim_actor=256,
        num_hidden_layers_critic=1,
        hidden_dim_critic=256,
        embeddings_in_state_actor=85,
        embedding_dim_actor=30
    ):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.n_actions = n_actions

        self.actor = ActorNetwork(
            n_actions, 
            input_dims, 
            alpha, 
            num_hidden_layers=num_hidden_layers_actor, 
            hidden_dim=hidden_dim_actor,
            embeddings_in_state=embeddings_in_state_actor,
            embedding_dim=embedding_dim_actor
        )
        self.critic = CriticNetwork(
            input_dims, 
            alpha,
            num_hidden_layers=num_hidden_layers_critic,
            hidden_dim=hidden_dim_critic
        )
        actor_param_count = sum(p.numel() for p in self.actor.parameters())
        critic_param_count = sum(p.numel() for p in self.critic.parameters())
        print(f'NUM - PARAMS {actor_param_count + critic_param_count}')
        self.memory = PPOMemory(batch_size)
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        #print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        #print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    '''
    def choose_action(self, observation, legal_actions):
        state = torch.tensor(np.array([observation]), dtype=torch.float).to(self.actor.device)
        
        dist = self.actor(state)
        legal_actions = torch.tensor(np.array(legal_actions)).to(self.actor.device)
        dist = Categorical(dist)
        value = self.critic(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value
    '''
    
    '''
    def choose_action(self, observation, legal_actions):
        # Convert observation to tensor and send it to the actor's device.
        state = torch.tensor(np.array([observation]), dtype=torch.float).to(self.actor.device)

        # Get logits from the actor; assumed shape: (batch, 1, k)
        logits = self.actor(state)

        # Convert legal_actions to a tensor on the same device.
        legal_actions = torch.tensor(np.array(legal_actions)).to(self.actor.device)

        # Mask logits so that only legal actions remain.
        # If logits has shape (n, 1, k), we index into the third dimension.
        masked_logits = logits[:, 0, legal_actions]  # shape: (n, len(legal_actions))

        # Build a Categorical using the masked logits.
        dist = Categorical(logits=masked_logits)

        # Get the value estimate from the critic.
        value = self.critic(state)

        # Sample an action index from the distribution.
        # This gives a relative index in the filtered (legal) logits.
        relative_index = dist.sample()  # shape: (n,)

        # Map the relative index back to the original action space.
        action = legal_actions[relative_index]

        # Compute the log probability for the sampled action.
        probs = torch.squeeze(dist.log_prob(relative_index)).item()

        # Squeeze and convert to Python scalars.
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value
    '''
    
    def choose_action(self, observation, legal_actions=None):
        state = torch.tensor(np.array([observation]), dtype=torch.float).to(self.actor.device)

        # Get logits from the actor; assumed shape: (n, 1, k)
        logits = self.actor(state)

        # If legal_actions is provided, mask the logits to include only those actions.
        if legal_actions is not None:
            legal_actions = torch.tensor(np.array(legal_actions)).to(self.actor.device)
            masked_logits = logits[:, 0, legal_actions]  # shape: (n, len(legal_actions))
            dist = Categorical(logits=masked_logits)
        else:
            # Use all logits if no legal actions mask is provided.
            full_logits = logits.squeeze(1)  # shape: (n, k)
            dist = Categorical(logits=full_logits)

        value = self.critic(state)

        # Sample an action index from the distribution.
        sampled_index = dist.sample()  # shape: (n,)

        # Map back to the original action if a mask was applied.
        if legal_actions is not None:
            action = legal_actions[sampled_index]
        else:
            action = sampled_index

        # Get the log probability of the sampled action.
        log_prob = torch.squeeze(dist.log_prob(sampled_index)).item()

        # Squeeze and convert to Python scalars.
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, log_prob, value


    def learn(self):
        device = self.actor.device  # Assuming this is a CUDA device
        for _ in range(self.n_epochs):
            # Retrieve batch data
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
                self.memory.generate_batches()
            
            # Convert arrays to torch tensors on GPU
            rewards = torch.tensor(reward_arr, dtype=torch.float32, device=device)
            values = torch.tensor(vals_arr, dtype=torch.float32, device=device)
            dones = torch.tensor(dones_arr, dtype=torch.float32, device=device)
            
            # Compute deltas for GAE: delta_t = r_t + gamma * V(t+1) * (1-done) - V(t)
            deltas = rewards[:-1] + self.gamma * values[1:] * (1 - dones[:-1]) - values[:-1]
            
            # Compute advantage vector using the vectorized discounted cumulative sum.
            advantage = torch.zeros_like(rewards, device=device)
            advantage[:-1] = discount_cumsum(deltas, self.gamma * self.gae_lambda)
            
            # Loop over minibatches
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float, device=device)
                old_probs = torch.tensor(old_prob_arr[batch], device=device)
                actions = torch.tensor(action_arr[batch], device=device)
    
                # Forward pass through the actor and critic networks
                dist = self.actor(states)
                dist = Categorical(dist)
                critic_value = self.critic(states).squeeze()
                
                # Calculate probability ratio and losses
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
    
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value).pow(2).mean()
    
                total_loss = actor_loss + 0.5 * critic_loss
                
                # Update the networks
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
    
        self.memory.clear_memory()      

    def play(self, env:CellitaireEnv):
        """
        Just plays the game continously without learning. For Visualization
        """       
        while True:
            observation, reward, done, truncated, info = env.reset()
            observation = env.get_state()
            done = False
            score = 0
            episode_steps = 0
            while (not done) and (not truncated):
                # Once text-render is working? print(env.render())
                legal_actions = env.get_legal_actions_as_int()
                action, prob, val = self.choose_action(observation, legal_actions)
                observation_, reward, done, truncated, info = env.step(action)
                episode_steps += 1
                score += reward
                observation = observation_
            print(f'Episode ended. | {episode_steps} | cumulative score {score}')

    def train(self, env:CellitaireEnv, learn_steps:int):
        score_history = []
        moves_history = []
        cards_saved_history = []
        best_score = -10000
        n_steps = 0
        learn_iters = 0
        episode_count = 0
        while True:
            observation, reward, done, truncated, info = env.reset()
            observation = env.get_state()
            done = False
            score = 0
            while (not done) and (not truncated):
                legal_actions = env.get_legal_actions_as_int()
                action, prob, val = self.choose_action(observation, legal_actions)
                observation_, reward, done, truncated, info = env.step(action)
                print(observation_)
                print(f'Legal Actions: {env.get_legal_actions()}')
                n_steps += 1
                score += reward
                self.remember(observation, action, prob, val, reward, done)
                if n_steps % learn_steps == 0:
                    self.learn()
                    learn_iters += 1
                observation = observation_
            score_history.append(score)
            moves_history.append(env.num_moves)
            cards_saved_history.append(env.game.foundation.total_cards())
            avg_score = np.mean(score_history[-100:])
            avg_moves = np.mean(moves_history[-100:])
            avg_cards_saved = np.mean(cards_saved_history[-100:])
            episode_count += 1

            if avg_score > best_score:
                best_score = avg_score
                #self.save_models()
                episodes_without_best = 0
                print(f'episode {episode_count:>5} | score {score:>6.1f} | recent avg {avg_score:>6.1f} | avg moves {avg_moves:>5.1f} | avg saved {avg_cards_saved:>4.1f} | learning steps {learn_iters:6} | done {done} *') 
            else:
                episodes_without_best += 1
                
            if episodes_without_best % 100 == 0 and episodes_without_best > 0:
                print(f'episode {episode_count:>5} | score {score:>6.1f} | recent avg {avg_score:>6.1f} | avg moves {avg_moves:>5.1f} | avg saved {avg_cards_saved:>4.1f} | learning steps {learn_iters:6} | done {done}') 
            

def discount_cumsum(x, discount):
    """
    Compute discounted cumulative sums of vector x with discount factor.
    For each index t, returns sum_{l=0}^{T-t-1} discount^l * x[t+l].
    """
    T = x.size(0)
    # Create a vector of discount factors
    discount_factors = discount ** torch.arange(T, device=x.device, dtype=x.dtype)
    # Multiply elementwise and compute cumulative sums in reverse order
    x_discounted = x * discount_factors
    reversed_x = torch.flip(x_discounted, dims=[0])
    cumsum_reversed = torch.cumsum(reversed_x, dim=0)
    discounted_cumsum = torch.flip(cumsum_reversed, dims=[0])
    # Divide by discount factors to get the proper values
    return discounted_cumsum / discount_factors


