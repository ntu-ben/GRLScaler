#!/usr/bin/env python3
"""
Debug script for GNNRL training issues
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import logging
from stable_baselines3 import A2C
from gnnrl.core.envs import OnlineBoutique
from gnnrl.core.agents.ppo_gnn import GNNPPOPolicy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_training():
    """Debug GNNRL training step by step"""
    
    # Step 1: Create environment
    logger.info("Creating environment...")
    try:
        env = OnlineBoutique(
            k8s=False,
            goal_reward="latency",
            use_graph=True,
            dataset_path=None
        )
        logger.info("✓ Environment created")
    except Exception as e:
        logger.error(f"❌ Environment creation failed: {e}")
        return
    
    # Step 2: Test environment reset
    logger.info("Testing environment reset...")
    try:
        obs, info = env.reset()
        logger.info(f"✓ Reset successful, obs keys: {obs.keys() if isinstance(obs, dict) else type(obs)}")
    except Exception as e:
        logger.error(f"❌ Reset failed: {e}")
        return
    
    # Step 3: Test environment step
    logger.info("Testing environment step...")
    try:
        # Create a valid action (all zeros for "do nothing")
        action = [0] * env.action_space.nvec.shape[0]
        logger.info(f"Action shape: {len(action)}, Action space: {env.action_space}")
        
        step_result = env.step(action)
        logger.info(f"✓ Step successful, returned {len(step_result)} values")
        logger.info(f"Step result types: {[type(x) for x in step_result]}")
    except Exception as e:
        logger.error(f"❌ Step failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Create model
    logger.info("Creating GNNRL model...")
    try:
        metadata = (
            ['svc', 'node'],  # node types
            [('svc', 'calls', 'svc')]  # edge types
        )
        
        policy_kwargs = {
            'metadata': metadata,
            'model': 'tgn',
            'embed_dim': 16,
        }
        
        model = A2C(
            GNNPPOPolicy,
            env=env,
            learning_rate=3e-4,
            n_steps=5,
            verbose=1,
            policy_kwargs=policy_kwargs,
        )
        logger.info("✓ Model created")
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Test training
    logger.info("Testing training...")
    try:
        model.learn(total_timesteps=5)
        logger.info("✓ Training successful")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    env.close()
    logger.info("🎉 All tests passed!")

if __name__ == "__main__":
    debug_training()