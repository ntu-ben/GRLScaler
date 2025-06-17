import argparse
import json
from pathlib import Path
from queue import Queue

import pandas as pd
from stable_baselines3 import PPO

from gnn_rl.agents.ppo_gnn import GNNPPOPolicy
from gnn_rl.common.feature_builder import build_hetero_data
from gnn_rl.envs import K8sEnv


BASELINES = [
    ("mlp_ppo", None, "ppo"),
    ("gat_ppo", "gat", "ppo"),
    ("gcn_ppo", "gcn", "ppo"),
    ("gat_sac", "gat", "sac"),
]


def create_env(fixtures=True):
    if fixtures:
        q = Queue()
        edges = json.load(open("tests/fixtures/edges.json"))
        metrics = json.load(open("tests/fixtures/metrics.json"))
        q.put({"edges": edges, "metrics": metrics})
    else:
        q = Queue()
    return K8sEnv(q)


def run(cfg_name, encoder, algo, seed, steps):
    env = create_env()
    obs = env.reset()
    metadata = build_hetero_data(
        obs["graph"]["svc_df"],
        obs["graph"]["node_df"],
        obs["graph"]["edge_df"],
    ).metadata()
    policy_kwargs = {"metadata": metadata, "model": encoder} if encoder else {}
    if algo == "ppo":
        model = PPO(
            GNNPPOPolicy if encoder else "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            seed=seed,
            n_steps=64,
            batch_size=64,
        )
    else:
        # Placeholder for discrete SAC or other algos
        model = PPO(
            GNNPPOPolicy if encoder else "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            seed=seed,
            n_steps=64,
            batch_size=64,
        )
    model.learn(steps)
    # Simple evaluation
    total_reward = 0.0
    for _ in range(5):
        obs, _ = env.reset(), {}
        action, _ = model.predict(obs)
        _, reward, *_ = env.step(action)
        total_reward += reward
    return {
        "cfg": cfg_name,
        "seed": seed,
        "reward": total_reward / 5,
        "slo_vio": 0.0,
        "caf": 0.0,
        "slack": 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    Path(args.output).mkdir(exist_ok=True)
    all_results = []
    for cfg_name, enc, algo in BASELINES:
        for seed in range(args.seeds):
            res = run(cfg_name, enc, algo, seed, args.steps)
            all_results.append(res)
    df = pd.DataFrame(all_results)
    df.to_csv(Path(args.output) / "metrics.csv", index=False)

    summary = df.groupby("cfg")["reward"].mean().reset_index()
    summary.to_csv(Path(args.output) / "summary.csv", index=False)


if __name__ == "__main__":
    main()
