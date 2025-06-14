"""GNN-based RL package."""

try:
    from gym.envs.registration import register
except Exception:  # pragma: no cover - gym is optional for some tests
    register = lambda *a, **kw: None

try:
    from .envs import Redis, OnlineBoutique
    register(
        id='Redis-v0',
        entry_point='gnn_rl.envs:Redis',
    )
except Exception:  # pragma: no cover - optional
    Redis = OnlineBoutique = None

__all__ = [
    'Redis',
    'OnlineBoutique',
]
