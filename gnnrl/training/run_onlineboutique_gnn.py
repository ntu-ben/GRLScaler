#!/usr/bin/env python3
"""
Simple script to run OnlineBoutique with k8s cluster mode and GNN enabled
This script provides an easy way to experiment with GNN-based autoscaling.

Usage:
    python run_onlineboutique_gnn.py [--k8s] [--steps STEPS] [--goal GOAL]

Options:
    --k8s           Enable k8s cluster mode (default: simulation mode)
    --steps STEPS   Number of training steps (default: 1000)
    --goal GOAL     Optimization goal: 'latency' or 'cost' (default: 'latency')
"""

import sys
import argparse
sys.path.append('.')

from stable_baselines3 import PPO
from gnnrl.core.envs import OnlineBoutique
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Run OnlineBoutique GNN experiment')
    parser.add_argument('--k8s', action='store_true', help='Enable k8s cluster mode')
    parser.add_argument('--steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--goal', default='latency', choices=['latency', 'cost'], 
                       help='Optimization goal')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Display experiment configuration
    logger.info("=== GNN OnlineBoutique Experiment ===")
    logger.info(f"K8s mode: {'Enabled' if args.k8s else 'Simulation'}")
    logger.info(f"Training steps: {args.steps}")
    logger.info(f"Optimization goal: {args.goal}")
    
    # Check environment setup
    if args.k8s:
        logger.info("Checking K8s connectivity...")
        if not os.getenv('KUBE_HOST'):
            logger.warning("KUBE_HOST not set in environment. Make sure kubectl proxy is running.")
    
    # Create OnlineBoutique environment with k8s and graph features enabled
    logger.info("Creating OnlineBoutique environment with use_graph=True")
    env = OnlineBoutique(k8s=args.k8s, use_graph=True, goal_reward=args.goal)
    
    logger.info("Environment created successfully")
    logger.info(f"Action space: {env.action_space}")
    logger.info(f"Observation space: {env.observation_space}")
    
    # Test environment reset
    obs = env.reset()
    logger.info(f"Environment reset successful")
    logger.info(f"Observation type: {type(obs)}")
    logger.info(f"Observation keys: {list(obs.keys())}")
    logger.info(f"Node features shape: {obs['node_features'].shape}")
    logger.info(f"Adjacency matrix shape: {obs['adjacency'].shape}")
    
    # Create PPO model with MultiInputPolicy for dict observation space
    logger.info("Creating PPO model...")
    model = PPO("MultiInputPolicy", env, verbose=1, n_steps=128, batch_size=64)
    
    # Note: For now using standard MLP policy. To use GNN policy, you would need:
    # from gnn_rl.agents.ppo_gnn import GNNPPOPolicy
    # But that requires additional metadata setup
    
    logger.info(f"Starting training for {args.steps} steps...")
    model.learn(total_timesteps=args.steps, progress_bar=True)
    
    logger.info("Training completed successfully!")
    
    # Save the model
    model_name = f"onlineboutique_{'k8s' if args.k8s else 'sim'}_gnn_{args.goal}_{args.steps}steps"
    model.save(model_name)
    logger.info(f"Model saved as '{model_name}'")
    
    # Test the trained model
    logger.info("Testing trained model...")
    obs = env.reset()
    for i in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        logger.info(f"Step {i+1}: Action={action}, Reward={reward:.4f}")
        if done:
            obs = env.reset()
            logger.info("Episode finished, resetting environment")
    
    logger.info("Test completed successfully!")

if __name__ == "__main__":
    main()