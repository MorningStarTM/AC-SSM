import os
import torch
import grid2op
from lightsim2grid import LightSimBackend
from ac_ssm.ssm.linear_ssm import ACSSM
from ac_ssm.utils.logger import logger
from ac_ssm.ssm.linear_ssm import SSMTrainer
from grid2op.Reward.l2RPNSandBoxScore import L2RPNSandBoxScore
from ac_ssm.utils.converter import ActionConverter
from grid2op.Action import TopologyChangeAction, TopologySetAction



env_name = "rte_case5_example"
env = grid2op.make(env_name, test=True, reward_class=L2RPNSandBoxScore)


config = {
    'obs_dim':192,
    'state_dim':64,
    'n_actions':58,
    'lr':1e-4,
    'episodes':10000,
    'max_ep_len':2016,
    'gamma':0.99,
    'save_path':f'ssm/checkpoints/{env_name}_actor_critic.pt',

}
converter = ActionConverter(env)
trainer = SSMTrainer(env, converter, config)

trainer.train()