"""Custom gym environments used by the GNN-based RL agent."""

from gym.envs.registration import register

register(
    id='Redis-v0',
    entry_point='gnn_rl_env.envs:Redis',
)
