import argparse
from datetime import datetime
import os
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
import torch
from torch_geometric.utils import dense_to_sparse

from gnnrl.core.envs import Redis, OnlineBoutique
from gnnrl.core.agents.ppo_gnn import GNNPPOPolicy
from gnnrl.core.common.feature_builder import build_hetero_data

# Load environment variables so URLs/K8s settings come from `.env`
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)
except Exception:  # pragma: no cover - optional dependency
    if ENV_PATH.exists():
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                os.environ.setdefault(k, v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tgn", choices=["gat", "gcn", "tgn", "dysat"], help="GNN encoder type")
    parser.add_argument("--steps", type=int, default=1_000_00, help="Training steps")
    parser.add_argument("--k8s", action="store_true", help="Interact with a live Kubernetes cluster")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to dataset CSV file")
    parser.add_argument(
        "--use-case",
        default="redis",
        choices=["redis", "online_boutique"],
        help="Select training environment",
    )
    args = parser.parse_args()

    if args.use_case == "redis":
        env = Redis(k8s=args.k8s, use_graph=True, dataset_path=args.dataset_path)
    else:
        env = OnlineBoutique(k8s=args.k8s, use_graph=True, dataset_path=args.dataset_path)
    sample = env.reset()
    if isinstance(sample, tuple):
        sample = sample[0]

    # Convert raw observation to DataFrames for HeteroData
    svc_df = pd.DataFrame(sample["node_features"])  # service-level metrics
    edge_index, edge_attr = dense_to_sparse(torch.tensor(sample["adjacency"]))
    edge_df = pd.DataFrame({
        "src": edge_index[0].numpy(),
        "dst": edge_index[1].numpy(),
    })
    if edge_attr.numel() > 0:
        edge_df["weight"] = edge_attr.numpy()

    # Node (cluster-level) features are not provided
    node_df = pd.DataFrame()

    metadata = build_hetero_data(svc_df, node_df, edge_df).metadata()

    # Use same log directory structure as gym_hpa
    scenario = 'real' if args.k8s else 'simulated'
    goal = 'latency'  # default goal for this script
    log_dir = f"../../results/{args.use_case}/{scenario}/{goal}/"

    policy_kwargs = dict(metadata=metadata, model=args.model)
    model = PPO(GNNPPOPolicy, env, policy_kwargs=policy_kwargs, tensorboard_log=log_dir)
    
    # Create tb_log_name following gym_hpa convention with gnn suffix
    tb_log_name = f"ppo_{args.model}_env_{env.name}_goal_{goal}_k8s_{args.k8s}_totalSteps_{args.steps}_run"
    model.learn(args.steps, tb_log_name=tb_log_name)


if __name__ == "__main__":
    main()
