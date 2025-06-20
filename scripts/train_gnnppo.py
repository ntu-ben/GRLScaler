import argparse
from datetime import datetime

import pandas as pd
from stable_baselines3 import PPO
import torch
from torch_geometric.utils import dense_to_sparse

from gnn_rl.envs import Redis, OnlineBoutique
from gnn_rl.agents.ppo_gnn import GNNPPOPolicy
from gnn_rl.common.feature_builder import build_hetero_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gat", choices=["gat", "gcn", "dysat"], help="GNN encoder type")
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

    log_dir = f"runs/gnnppo/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    policy_kwargs = dict(metadata=metadata, model=args.model)
    model = PPO(GNNPPOPolicy, env, policy_kwargs=policy_kwargs, tensorboard_log=log_dir)
    model.learn(args.steps)


if __name__ == "__main__":
    main()
