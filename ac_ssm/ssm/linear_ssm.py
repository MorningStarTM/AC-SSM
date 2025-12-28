import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.distributions import Categorical
from ac_ssm.utils.logger import logger
import numpy as np
from ac_ssm.utils.utils import save_episode_rewards




class ACSSM(nn.Module):
    """
    Latent linear SSM + discrete-action actor.

    Dynamics:
        x_{t+1} = A x_t + B * onehot(a_t)

    Policy:
        logits_t = C x_t + D o_t
        a_t ~ Categorical(logits=logits_t)

    Notes:
      - o_t is the observation (B, obs_dim)
      - x_t is latent state (B, state_dim)
      - action is integer in [0, n_actions-1]
    """
    def __init__(self, obs_dim: int, state_dim: int, n_actions: int, use_obs_in_policy: bool = True, lr:float=3e-4):
        super().__init__()
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.use_obs_in_policy = use_obs_in_policy

        # Optional encoder: map observation -> latent x_t
        self.encoder = nn.Sequential(
                nn.Linear(obs_dim, 256), nn.ReLU(),
                nn.Linear(256, state_dim)
            )

        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),                 # ← normalize hidden to tame scales
            nn.Linear(256, 1)
        )

        # A: (state_dim, state_dim)
        self.A = nn.Parameter(0.01 * torch.randn(state_dim, state_dim))

        # B: (state_dim, n_actions)  (because we feed one-hot(action))
        self.B = nn.Parameter(0.01 * torch.randn(state_dim, n_actions))

        # C: (n_actions, state_dim)  maps x_t -> logits
        self.C = nn.Parameter(0.01 * torch.randn(n_actions, state_dim))

        # D: (n_actions, obs_dim) maps o_t -> logits 
        self.D = nn.Parameter(0.01 * torch.randn(n_actions, obs_dim))

        # If you also want a next-observation predictor from x_{t+1}
        self.obs_decoder = nn.Linear(state_dim, obs_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logprobs, self.state_values, self.rewards = [], [], []

    def init_state(self, o0: torch.Tensor) -> torch.Tensor:
        """Initialize latent state from first observation."""
        return self.encoder(o0)

    def policy_logits(self, x_t: torch.Tensor, o_t: torch.Tensor) -> torch.Tensor:
        """
        logits = C x_t + D o_t
        x_t: (B, state_dim)
        o_t: (B, obs_dim)
        returns: (B, n_actions)
        """
        logits_from_x = x_t @ self.C.T
        if self.use_obs_in_policy:
            logits_from_o = o_t @ self.D.T
            return logits_from_x + logits_from_o
        return logits_from_x

    def act(self, x_t: torch.Tensor, o_t: torch.Tensor, deterministic: bool = False):
        """Sample (or argmax) discrete integer action."""
        logits = self.policy_logits(x_t, o_t)
        dist = Categorical(logits=logits)
        if deterministic:
            a = torch.argmax(logits, dim=-1)
        else:
            a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp, dist

    def step_dynamics(self, x_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """
        x_{t+1} = A x_t + B onehot(a_t)
        x_t: (B, state_dim)
        a_t: (B,) int64
        returns x_{t+1}: (B, state_dim)
        """
        a_onehot = F.one_hot(a_t, num_classes=self.n_actions).float()  # (B, n_actions)
        x_tp1 = x_t @ self.A.T + a_onehot @ self.B.T
        return x_tp1

    def predict_next_obs(self, x_tp1: torch.Tensor) -> torch.Tensor:
        """Optional: decode predicted next observation from latent."""
        return self.obs_decoder(x_tp1)
    
    def encode(self, next_obs: torch.Tensor) -> torch.Tensor:
        """Encode observation into latent state."""
        return self.encoder(next_obs)
    
    def compute_intrinsic_reward(self, x_t_1: torch.Tensor, x_t:torch.Tensor) -> torch.Tensor:
        x_t_1_encoded = self.encoder(x_t_1)
        loss = F.mse_loss(x_t, x_t_1_encoded, reduction='none').mean(dim=-1)
        return loss

    def forward(self, o_t: torch.Tensor, x_t: torch.Tensor = None, deterministic: bool = False):
        """
        Convenience forward: given o_t (and optionally x_t), produce action.
        If x_t is None, it will be initialized from o_t.
        """
        
        if o_t is not torch.Tensor:
            o_t = torch.tensor(o_t, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        if x_t is None:
            x_t = self.init_state(o_t)

        a_t, logp, dist = self.act(x_t, o_t, deterministic=deterministic)
        x_tp1 = self.step_dynamics(x_t, a_t)

        self.logprobs.append(logp.squeeze(-1))
        self.state_values.append(self.critic(o_t).squeeze(-1))


        return {
            "x_t": x_t,
            "action": a_t,
            "logp": logp,
            "dist": dist,
            "x_tp1": x_tp1,
        }


    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]


    def calculateLoss(self, gamma=0.99, value_coef=0.5, entropy_coef=0.01):
        returns = []
        g = 0.0
        for r in reversed(self.rewards):
            g = r + gamma * g
            returns.insert(0, g)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
    
        # stabilize return normalization
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
    
        values = torch.stack(self.state_values).to(self.device).squeeze(-1)
        logprobs = torch.stack(self.logprobs).to(self.device)
    
        advantages = returns - values.detach()
        # advantage normalization helps a LOT with small/medium LR
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    
        policy_loss = -(logprobs * advantages).mean()
        value_loss  = F.smooth_l1_loss(values, returns)
    
        # crude entropy from logprobs; better is dist.entropy() if you also stored dist params
        entropy = -(logprobs.exp() * logprobs).mean()
    
        return policy_loss + value_coef * value_loss - entropy_coef * entropy
    

    def save_checkpoint(self, path: str):
        """Save model + optimizer to a single .pt file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "model": self.state_dict(),
                "optim": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str, device="cpu", strict: bool = True):
        """Load model + optimizer from a single .pt file."""
        ckpt = torch.load(path, map_location=device)

        self.load_state_dict(ckpt["model"], strict=strict)
        self.to(device)

        self.optimizer.load_state_dict(ckpt["optim"])
        # move optimizer state tensors to device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)




class SSMTrainer:
    def __init__(self, env, converter, config):
        self.env = env
        self.config = config
        self.converter = converter
        self.obs_dim = config['obs_dim']
        self.state_dim = config['state_dim']
        self.n_actions = config['n_actions']
        self.lr = config['lr']
        self.danger = 0.9
        self.thermal_limit = self.env._thermal_limit_a

        self.agent = ACSSM(self.obs_dim, self.state_dim, self.n_actions, use_obs_in_policy=True, lr=self.lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent.to(self.device)


        self.optimizer = self.agent.optimizer

        self.step_counter = 0
        self.episode_rewards = []
        self.episode_steps = [] 
        self.loss = []
        self.ac_losses = []
        self.obs_losses = []
        self.action_list = []

        os.makedirs("ssm\\episode_reward", exist_ok=True)
        os.makedirs("ssm\\episode_length", exist_ok=True)


    def is_safe(self, obs):
        
        for ratio, limit in zip(obs.rho, self.thermal_limit):
            # Seperate big line and small line
            if (limit < 400.00 and ratio >= self.danger - 0.05) or ratio >= self.danger:
                return False
        return True
    
    
    def train(self):
        running_reward = 0
        for i_episode in range(0, self.config['episodes']):
            obs = self.env.reset()
            done = False
            episode_total_reward = 0

            loss_obs = 0.0
            x_t = None
            for t in range(self.config['max_ep_len']):

                is_safe = self.is_safe(obs)
                if not is_safe:
                    output = self.agent(obs.to_vect(), x_t=x_t)
                    x_t = output["x_tp1"].detach()
                    self.action_list.append(output['action'].cpu().numpy()[0])
                    grid_action = self.converter.act(output['action'].cpu().numpy()[0])
                else:
                    grid_action = self.env.action_space({})
                    self.action_list.append(58)


                obs_, reward, done, _ = self.env.step(grid_action)

                next_obs = torch.tensor(obs_.to_vect(), dtype=torch.float32).unsqueeze(0).to(self.device)
                encoded_next_obs = self.agent.encode(next_obs)

                

                if not is_safe:
                    # ToDo: monitor action and do_actions
                    loss_obs += F.mse_loss(output['x_tp1'], encoded_next_obs).mean()
                    self.agent.rewards.append(reward)

                episode_total_reward += reward
                obs = obs_

                if done:
                    break

            #logger.info(f"Episode {i_episode} reward: {episode_total_reward}")  
            self.episode_rewards.append(episode_total_reward)  
            # Updating the policy :
            self.optimizer.zero_grad()
            ac_loss = self.agent.calculateLoss(self.config['gamma'])
            self.ac_losses.append(ac_loss.item())
            self.obs_losses.append(loss_obs.item())
            loss = ac_loss + loss_obs
            self.loss.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
            self.optimizer.step()        
            self.agent.clearMemory()
            loss_obs = 0.0

            # saving the model if episodes > 999 OR avg reward > 200 
            if i_episode != 0 and i_episode % 1000 == 0:
                self.agent.save_checkpoint(path=self.config['save_path'])    
           
            
            if i_episode % 20 == 0:
                running_reward = running_reward/20
                logger.info('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, episode_total_reward))
                running_reward = 0
            
            survival_steps = t + 1          # because t is 0-indexed
            self.episode_steps.append(survival_steps)

        save_episode_rewards(self.episode_rewards, save_dir="ssm\\episode_reward", filename="actor_critic_reward.npy")
        logger.info(f"reward saved at ssm\\episode_reward")
        
        np.save("ssm\\episode_length\\actor_critic_steps.npy", np.array(self.episode_steps, dtype=int))
        np.save("ssm\\episode_reward\\actor_critic_loss.npy", np.array(self.loss, dtype=np.float32))
        np.save("ssm\\episode_reward\\actor_critic_actions.npy", np.array(self.action_list, dtype=np.float32))

        



        