import argparse
from datetime import datetime

import pandas as pd
from stable_baselines3 import PPO

from gnn_rl.envs import Redis
from gnn_rl.agents.ppo_gnn import GNNPPOPolicy
from gnn_rl.common.feature_builder import build_hetero_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gat", choices=["gat", "gcn", "dysat"], help="GNN encoder type")
    parser.add_argument("--steps", type=int, default=1_000_00, help="Training steps")
    args = parser.parse_args()

    env = Redis(use_graph=True)
    sample = env.reset()
    svc_df = pd.DataFrame(sample.get("svc_df", []))
    node_df = pd.DataFrame(sample.get("node_df", []))
    edge_df = pd.DataFrame(sample.get("edge_df", []))
    metadata = build_hetero_data(svc_df, node_df, edge_df).metadata()

    log_dir = f"runs/gnnppo/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    policy_kwargs = dict(metadata=metadata, model=args.model)
    model = PPO(GNNPPOPolicy, env, policy_kwargs=policy_kwargs, tensorboard_log=log_dir)
    model.learn(args.steps)


if __name__ == "__main__":
    main()
