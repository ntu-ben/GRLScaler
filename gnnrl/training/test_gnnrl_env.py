#!/usr/bin/env python3
"""
Quick test script to verify GNNRL environment and model compatibility.
This script tests the fixed flat_feats bug and validates the environment.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import logging
import numpy as np
from stable_baselines3.common.env_checker import check_env

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_environment_creation():
    """Test OnlineBoutique environment creation with GNN mode."""
    logger.info("Testing OnlineBoutique environment creation...")
    
    try:
        from gnnrl.core.envs import OnlineBoutique
        
        # Test simulation mode with graph
        env = OnlineBoutique(
            k8s=False,
            goal_reward="latency", 
            use_graph=True,
            dataset_path=None
        )
        
        logger.info("✓ Environment created successfully")
        
        # Test reset
        obs = env.reset()
        logger.info(f"✓ Environment reset successful")
        logger.info(f"Observation keys: {obs.keys()}")
        
        if 'graph' in obs:
            logger.info(f"Graph keys: {obs['graph'].keys()}")
            logger.info(f"svc_df shape: {obs['graph']['svc_df'].shape}")
            logger.info(f"node_df shape: {obs['graph']['node_df'].shape}")
            logger.info(f"edge_df shape: {obs['graph']['edge_df'].shape}")
        
        if 'flat_feats' in obs:
            logger.info(f"✓ flat_feats found with shape: {obs['flat_feats'].shape}")
        else:
            logger.error("❌ flat_feats missing from observation!")
            return False
        
        # Test environment checker
        check_env(env)
        logger.info("✓ Environment validation passed")
        
        env.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment test failed: {e}")
        return False

def test_gnn_policy():
    """Test GNNPPOPolicy creation and forward pass."""
    logger.info("Testing GNNPPOPolicy...")
    
    try:
        from gnnrl.core.envs import OnlineBoutique
        from gnnrl.core.agents.ppo_gnn import GNNPPOPolicy
        from gymnasium import spaces
        import torch
        
        # Create environment to get observation/action spaces  
        dataset_path = str(Path(__file__).parent.parent / "data/datasets/real/onlineboutique/v1/online_boutique_gym_observation.csv")
        env = OnlineBoutique(k8s=False, use_graph=True, dataset_path=dataset_path)
        obs = env.reset()
        
        # Create dummy metadata
        metadata = (
            ['svc', 'node'],  # node types
            [('svc', 'calls', 'svc')]  # edge types
        )
        
        # Create policy
        policy = GNNPPOPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda x: 3e-4,
            metadata=metadata,
            model='gat',
            embed_dim=16
        )
        
        logger.info("✓ GNNPPOPolicy created successfully")
        
        # Test forward pass
        obs_tensor = {}
        for key, value in obs.items():
            if key == 'graph':
                obs_tensor[key] = {}
                for subkey, subvalue in value.items():
                    obs_tensor[key][subkey] = torch.tensor(subvalue, dtype=torch.float32)
            else:
                obs_tensor[key] = torch.tensor(value, dtype=torch.float32)
        
        # Add batch dimension
        for key in obs_tensor:
            if key == 'graph':
                for subkey in obs_tensor[key]:
                    if obs_tensor[key][subkey].dim() == 1:
                        obs_tensor[key][subkey] = obs_tensor[key][subkey].unsqueeze(0)
                    elif obs_tensor[key][subkey].dim() == 2:
                        obs_tensor[key][subkey] = obs_tensor[key][subkey].unsqueeze(0)
            else:
                if obs_tensor[key].dim() == 1:
                    obs_tensor[key] = obs_tensor[key].unsqueeze(0)
        
        logits, values = policy.forward(obs_tensor)
        logger.info(f"✓ Forward pass successful - logits shape: {logits.shape}, values shape: {values.shape}")
        
        env.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ GNN policy test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all tests."""
    logger.info("🧪 Starting GNNRL environment and model tests...")
    logger.info("="*50)
    
    # Test environment
    env_ok = test_environment_creation()
    
    logger.info("="*50)
    
    # Test GNN policy
    policy_ok = test_gnn_policy()
    
    logger.info("="*50)
    
    if env_ok and policy_ok:
        logger.info("🎉 All tests passed! GNNRL is ready to use.")
        return 0
    else:
        logger.error("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())