import torch 
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import grid2op
from ac_ssm.utils.converter import ActionConverter
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNSandBoxScore
import os
from ac_ssm.utils.logger import logger
import torch.optim as optim
from collections import defaultdict
import os
import numpy as np
from ac_ssm.utils.utils import save_episode_rewards



class ActorCriticUP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        self.affine = nn.Sequential(nn.Linear(self.config['input_dim'], 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 256),
                                    nn.ReLU(),
                                    nn.LayerNorm(256),
                                    nn.ReLU())


        self.action_layer = nn.Linear(256, self.config.get('action_dim', 58))
        self.value_layer  = nn.Linear(256, 1)

        self.logprobs, self.state_values, self.rewards = [], [], []

        # Optional: safer initializations
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _sanitize(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        x.clamp_(-1e6, 1e6)
        return x


    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---------- ADD THIS ----------
    def _human_readable(self, n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        elif n >= 1_000:
            return f"{n / 1_000:.2f}K"
        else:
            return str(n)

    def param_counts(model: nn.Module):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable = total - trainable

        logger.info(f"Total params:        {total/1e6:.3f} M")
        logger.info(f"Trainable params:    {trainable/1e6:.3f} M")
        logger.info(f"Non-trainable params:{non_trainable/1e6:.3f} M")
    
    def forward(self, state_np):
        x = torch.from_numpy(state_np).float().to(self.value_layer.weight.device)
        x = self._sanitize(x)

        if x.dim() == 1:                      # ensure [B, F]
            x = x.unsqueeze(0)

        h = self.affine(x)                       # includes LayerNorm + ReLU
        h = torch.nan_to_num(h)                  # belt & suspenders

        logits = self.action_layer(h)
        logits = torch.nan_to_num(logits)        # if any NaN slipped through
        logits = logits - logits.max(dim=-1, keepdim=True).values           # stable softmax
        probs  = torch.softmax(logits, dim=-1)

        # final guard
        if not torch.isfinite(probs).all():
            # Zero-out non-finites and renormalize as an emergency fallback
            probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
            s = probs.sum()
            probs = (probs + 1e-12) / (s + 1e-12)

        dist   = Categorical(probs=probs)
        action = dist.sample()

        self.logprobs.append(dist.log_prob(action).squeeze(-1))
        self.state_values.append(self.value_layer(h).squeeze(-1))

        return action.item()
    

    def calculateLoss(self, gamma=0.99, value_coef=0.5, entropy_coef=0.01):
        # discounted returns
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

    
    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]

    def save_checkpoint(self, optimizer:optim=None, filename="actor_critic_checkpoint.pth"):
        """Save model + optimizer for exact training resumption."""
        os.makedirs("models", exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        save_path = os.path.join("models", filename)
        torch.save(checkpoint, save_path)
        print(f"[SAVE] Checkpoint saved to {save_path}")


    def load_checkpoint(self, folder_name=None, filename="actor_critic_checkpoint.pth", optimizer:optim=None, load_optimizer=True):
        """Load model + optimizer state."""
        if folder_name is not None:
            file_path = os.path.join(folder_name, filename)
        else:
            file_path = os.path.join("models", filename)
        if not os.path.exists(file_path):
            print(f"[LOAD] No checkpoint found at {file_path}")
            return False

        checkpoint = torch.load(file_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[LOAD] Checkpoint loaded from {file_path}")
        return True
    




class ActorCriticTrainer:
    def __init__(self, env, actor_config):
        self.actor_config = actor_config
        self.env = env
        self.danger = 0.9
        self.thermal_limit = self.env._thermal_limit_a
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agent = ActorCriticUP(actor_config)
        
        self.agent.param_counts()

        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=actor_config['learning_rate'])

        if actor_config['load_model']:
            self.agent.load_checkpoint(optimizer=self.optimizer, folder_name=actor_config['actor_checkpoint_path'], filename=actor_config['checkpoint_path'])
            logger.info(f"ActorCritic Model Loaded from {actor_config['actor_checkpoint_path']}/{actor_config['checkpoint_path']}")

        self.converter = ActionConverter(self.env)

        self.episode_rewards = []
        self.episode_lenths = []
        self.episode_reasons = []
        self.agent_actions = []
        self.actor_loss = []
        self.episode_path = self.actor_config['episode_path']
        os.makedirs(self.episode_path, exist_ok=True)
        logger.info(f"Episode path : {self.episode_path}")



    def is_safe(self, obs):
        
        for ratio, limit in zip(obs.rho, self.thermal_limit):
            # Seperate big line and small line
            if (limit < 400.00 and ratio >= self.danger - 0.05) or ratio >= self.danger:
                return False
        return True
    
    
    def train(self):
        running_reward = 0
        actions = []
        for i_episode in range(0, self.actor_config['episodes']):
            obs = self.env.reset()
            done = False
            episode_total_reward = 0

            for t in range(self.actor_config['max_ep_len']):
                is_safe = self.is_safe(obs)

                if not is_safe:
                    action = self.agent(obs.to_vect())
                    actions.append(action)
                    grid_action = self.converter.act(action)
                else:
                    grid_action = self.env.action_space({})
                obs_, reward, done, _ = self.env.step(grid_action)

                if not is_safe:
                    self.agent.rewards.append(reward)

                episode_total_reward += reward
                obs = obs_

                if done:
                    break

            
            self.episode_rewards.append(episode_total_reward)  
            self.episode_lenths.append(t + 1)
            # Updating the policy :
            self.optimizer.zero_grad()
            loss = self.agent.calculateLoss(self.actor_config['gamma'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
            self.optimizer.step()        
            self.agent.clearMemory()

            # saving the model if episodes > 999 OR avg reward > 200 
            if i_episode != 0 and i_episode % 1000 == 0:
                self.agent.save_checkpoint(optimizer=self.optimizer, filename="actor_critic.pt")    
           
            
            if i_episode % 20 == 0:
                running_reward = running_reward/20
                logger.info('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, episode_total_reward))
                running_reward = 0
            
            survival_steps = t + 1          # because t is 0-indexed
            self.episode_lenths.append(survival_steps)

        save_episode_rewards(self.episode_rewards, save_dir="ICM\\episode_reward", filename="actor_critic_reward.npy")
        np.save(os.path.join(self.episode_path, "actor_critic_lengths.npy"),
                np.array(self.episode_lenths, dtype=np.int32)) 
        np.save(os.path.join(self.episode_path, "actor_critic_actions.npy"), np.array(actions, dtype=np.int32))
        np.save(os.path.join(self.episode_path, "actor_critic_loss.npy"), np.array(self.actor_loss, dtype=np.float32))
        logger.info(f"reward saved at ICM\\episode_reward")
        os.makedirs("ICM\\episode_reward", exist_ok=True)
        np.save("ICM\\episode_reward\\actor_critic_steps.npy", np.array(self.episode_lenths, dtype=int))