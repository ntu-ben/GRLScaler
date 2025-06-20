"""Custom gym environments used by the GNN-based RL agent."""

from gymnasium.envs.registration import register

# Re-export environments from the consolidated ``gnn_rl.envs`` package so
# existing entry points remain valid.
from gnn_rl.envs import Redis, K8sEnv

register(
    id="Redis-v0",
    entry_point="gnn_rl.envs:Redis",
)

register(
    id="K8sEnv-v0",
    entry_point="gnn_rl.envs:K8sEnv",
)
