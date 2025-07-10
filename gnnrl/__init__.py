"""GNNRL: Graph Neural Network Reinforcement Learning for Kubernetes Autoscaling."""

__version__ = "1.0.0"

# Re-export main components from core module
from .core import (
    HeteroGraphEncoder,
    HeteroGAT,
    GCNEncoder,
    TGNEncoder,
    GNNPPOPolicy,
    Redis,
    OnlineBoutique,
    K8sEnv
)

__all__ = [
    "HeteroGraphEncoder",
    "HeteroGAT",
    "GCNEncoder",
    "TGNEncoder",
    "GNNPPOPolicy",
    "Redis",
    "OnlineBoutique",
    "K8sEnv"
]