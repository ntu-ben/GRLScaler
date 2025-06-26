#!/usr/bin/env python3
"""
GNNRL Experiment Runner for OnlineBoutique Microservices
=======================================================

This script runs Graph Neural Network Reinforcement Learning experiments
on the OnlineBoutique microservices in Kubernetes.

Features:
- Support for both simulation and live K8s cluster modes
- Multiple GNN model types (GAT, GCN)
- Configurable training parameters
- Automatic environment setup and validation
- Integration with Prometheus metrics and Kiali service graph

Usage:
    # Basic simulation mode
    python run_gnnrl_experiment.py

    # Live K8s cluster mode
    python run_gnnrl_experiment.py --k8s --steps 5000

    # Custom model and goal
    python run_gnnrl_experiment.py --k8s --model gat --goal cost --steps 10000

Requirements:
- OnlineBoutique deployed in 'onlineboutique' namespace
- Prometheus and Kiali running in 'istio-system' namespace
- kubectl proxy running on port 8001 (for K8s mode)
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback

from gnnrl.core.envs import OnlineBoutique
from stable_baselines3.common.policies import ActorCriticPolicy

class DetailedLoggingCallback(BaseCallback):
    """Custom callback for detailed training logging"""
    
    def __init__(self, verbose=0):
        super(DetailedLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log every 100 steps with detailed metrics
        if self.n_calls % 100 == 0:
            # Get current environment info
            infos = self.locals.get('infos', [{}])
            env = self.training_env.envs[0] if hasattr(self.training_env, 'envs') else self.training_env
            
            # Extract metrics from environment
            if hasattr(env, 'deploymentList') and env.deploymentList:
                total_pods = sum(d.num_pods for d in env.deploymentList)
                avg_latency = getattr(env.deploymentList[9], 'latency', 0) if len(env.deploymentList) > 9 else 0
                total_replicas = sum(d.desired_replicas for d in env.deploymentList)
            else:
                total_pods = avg_latency = total_replicas = 0
            
            # Log detailed training metrics
            logger.info(f"Step {self.n_calls}: "
                       f"Total_Pods={total_pods}, "
                       f"Avg_Latency={avg_latency:.2f}, "
                       f"Total_Replicas={total_replicas}, "
                       f"Episode_Reward={self.locals.get('rewards', [0])[-1] if 'rewards' in self.locals else 0}")
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout"""
        logger.info(f"Rollout end at step {self.n_calls} - collecting experiences for policy update")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default environment variables
DEFAULT_ENV_VARS = {
    'KUBE_HOST': 'http://localhost:8001',
    'PROMETHEUS_URL': 'http://localhost:9090',
    'KIALI_URL': 'http://localhost:20001/kiali',
    'NAMESPACE_ONLINEBOUTIQUE': 'onlineboutique',
}

def load_env_vars():
    """Load environment variables from .env file if it exists."""
    env_file = Path(__file__).parent.parent.parent / '.env'
    
    # Set defaults first
    for key, value in DEFAULT_ENV_VARS.items():
        if key not in os.environ:
            os.environ[key] = value
            
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            logger.info(f"Loaded environment variables from {env_file}")
        except ImportError:
            # Manual parsing if dotenv not available
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
            logger.info(f"Manually loaded environment variables from {env_file}")
    else:
        logger.info("No .env file found, using defaults")

def validate_k8s_environment():
    """Validate that the K8s environment is properly set up."""
    import subprocess
    
    try:
        # Check if onlineboutique namespace exists and has pods
        result = subprocess.run(
            ['kubectl', 'get', 'pods', '-n', 'onlineboutique', '--no-headers'],
            capture_output=True, text=True, check=True
        )
        pods = result.stdout.strip().split('\n')
        running_pods = [p for p in pods if 'Running' in p]
        
        if len(running_pods) < 10:  # OnlineBoutique has 11 services
            logger.warning(f"Only {len(running_pods)} pods running in onlineboutique namespace")
            return False
            
        logger.info(f"✓ OnlineBoutique environment ready: {len(running_pods)} pods running")
        
        # Check if Prometheus and Kiali are running
        result = subprocess.run(
            ['kubectl', 'get', 'pods', '-n', 'istio-system', '--no-headers'],
            capture_output=True, text=True, check=True
        )
        istio_pods = result.stdout
        
        if 'prometheus' in istio_pods and 'kiali' in istio_pods:
            logger.info("✓ Prometheus and Kiali are running")
        else:
            logger.warning("⚠ Prometheus or Kiali may not be running")
            
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to validate K8s environment: {e}")
        return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run GNNRL experiments on OnlineBoutique microservices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--k8s', action='store_true',
        help='Enable live K8s cluster mode (default: simulation mode)'
    )
    
    parser.add_argument(
        '--steps', type=int, default=5000,
        help='Number of training steps (default: 5000)'
    )
    
    parser.add_argument(
        '--model', choices=['gat', 'gcn'], default='gat',
        help='GNN model type (default: gat)'
    )
    
    parser.add_argument(
        '--goal', choices=['latency', 'cost'], default='latency',
        help='Optimization goal (default: latency)'
    )
    
    parser.add_argument(
        '--embed-dim', type=int, default=32,
        help='GNN embedding dimension (default: 32)'
    )
    
    parser.add_argument(
        '--alg', choices=['ppo', 'a2c'], default='ppo',
        help='RL Algorithm (default: ppo)'
    )
    
    parser.add_argument(
        '--env-step-interval', type=float, default=15.0,
        help='Environment step interval in seconds (default: 15.0)'
    )
    
    parser.add_argument(
        '--dataset-path', type=str,
        default='gnnrl/data/datasets/real/onlineboutique/v1/online_boutique_gym_observation.csv',
        help='Dataset path for simulation mode'
    )
    
    parser.add_argument(
        '--save-freq', type=int, default=1000,
        help='Model checkpoint save frequency (default: 1000)'
    )
    
    parser.add_argument(
        '--log-dir', type=str, default='logs/gnnrl',
        help='Directory for training logs (default: logs/gnnrl)'
    )
    
    parser.add_argument(
        '--tensorboard-log', type=str, default='logs/tensorboard',
        help='Directory for tensorboard logs (default: logs/tensorboard)'
    )
    
    parser.add_argument(
        '--no-tensorboard', action='store_true',
        help='Disable tensorboard logging'
    )
    
    return parser.parse_args()

def create_environment(args):
    """Create and validate the training environment."""
    logger.info("Creating OnlineBoutique environment...")
    
    # Determine dataset path
    if not args.k8s and not os.path.exists(args.dataset_path):
        # Try relative to project root
        dataset_path = Path(__file__).parent.parent.parent / args.dataset_path
        if dataset_path.exists():
            args.dataset_path = str(dataset_path)
        else:
            logger.error(f"Dataset not found: {args.dataset_path}")
            return None
    
    try:
        env = OnlineBoutique(
            k8s=args.k8s,
            goal_reward=args.goal,
            use_graph=True,  # Enable GNN mode
            dataset_path=args.dataset_path if not args.k8s else None,
            waiting_period=args.env_step_interval
        )
        
        logger.info("✓ Environment created successfully")
        
        # Validate environment
        logger.info("Validating environment...")
        check_env(env)
        logger.info("✓ Environment validation passed")
        
        return env
        
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        return None

def create_model(env, args):
    """Create the GNNRL model."""
    logger.info(f"Creating GNNRL model with {args.model.upper()} encoder...")
    
    try:
        # Get metadata for GNN (node types and edge types)
        # OnlineBoutique has service nodes and cluster nodes
        metadata = (
            ['svc', 'node'],  # node types
            [('svc', 'calls', 'svc')]  # edge types - only service-to-service calls
        )
        
        # Setup tensorboard logging - use absolute path
        scenario = 'real' if args.k8s else 'simulated'
        tensorboard_log = None
        if not args.no_tensorboard:
            # 使用絕對路徑，基於專案根目錄
            results_dir = Path(__file__).parent.parent.parent / "results" / "online_boutique" / scenario / args.goal
            results_dir.mkdir(parents=True, exist_ok=True)
            tensorboard_log = str(results_dir)
            logger.info(f"Tensorboard logging enabled: {tensorboard_log}")
        else:
            logger.info("Tensorboard logging disabled")
        
        # Create RL model with MultiInput policy for Dict observation space
        if args.alg == 'a2c':
            model = A2C(
                policy="MultiInputPolicy",
                env=env,
                learning_rate=3e-4,
                n_steps=5,
                gamma=0.99,
                gae_lambda=1.0,
                ent_coef=0.01,
                vf_coef=0.25,
                max_grad_norm=0.5,
                verbose=1,
                tensorboard_log=tensorboard_log,
                policy_kwargs={}
            )
        else:  # default to PPO
            model = PPO(
                policy="MultiInputPolicy",
                env=env,
                learning_rate=3e-4,
                n_steps=256,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=1,
                tensorboard_log=tensorboard_log,
                policy_kwargs={}
            )
        
        logger.info("✓ GNNRL model created successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        return None

def run_experiment(args):
    """Run the main GNNRL experiment."""
    logger.info("="*60)
    logger.info("🚀 Starting GNNRL Experiment")
    logger.info("="*60)
    logger.info(f"Algorithm: {args.alg.upper()}")
    logger.info(f"Mode: {'Live K8s Cluster' if args.k8s else 'Simulation'}")
    logger.info(f"Model: {args.model.upper()}")
    logger.info(f"Goal: {args.goal}")
    logger.info(f"Steps: {args.steps:,}")
    logger.info(f"Embedding Dimension: {args.embed_dim}")
    logger.info("="*60)
    
    # Load environment variables
    load_env_vars()
    
    # Validate K8s environment if needed
    if args.k8s:
        logger.info("Validating K8s environment...")
        if not validate_k8s_environment():
            logger.error("❌ K8s environment validation failed")
            return False
        logger.info("✓ K8s environment validation passed")
    
    # Create environment
    env = create_environment(args)
    if env is None:
        return False
    
    # Create model
    model = create_model(env, args)
    if model is None:
        return False
    
    # Setup callbacks
    callbacks = []
    
    # Detailed logging callback
    detailed_logging = DetailedLoggingCallback(verbose=1)
    callbacks.append(detailed_logging)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.log_dir,
        name_prefix=f"gnnrl_{args.model}_{args.goal}"
    )
    callbacks.append(checkpoint_callback)
    
    callback_list = CallbackList(callbacks)
    
    # Run training
    logger.info(f"🎯 Starting training for {args.steps:,} steps...")
    start_time = time.time()
    
    try:
        # Create tensorboard log name - follow gym_hpa naming convention with gnn suffix
        env_name = "online_boutique_gym"
        tb_log_name = f"{args.alg}_{args.model}_env_{env_name}_goal_{args.goal}_k8s_{args.k8s}_totalSteps_{args.steps}"
        
        model.learn(
            total_timesteps=args.steps,
            callback=callback_list,
            tb_log_name=tb_log_name,
            progress_bar=False  # Disable progress bar to avoid tqdm dependency
        )
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Save final model
        model_filename = f"gnnrl_{args.model}_{args.goal}_k8s_{args.k8s}_steps_{args.steps}.zip"
        model.save(model_filename)
        logger.info(f"📁 Model saved as: {model_filename}")
        
        # Print summary
        logger.info("="*60)
        logger.info("📊 Experiment Summary")
        logger.info("="*60)
        logger.info(f"Training steps: {args.steps:,}")
        logger.info(f"Training time: {training_time:.2f} seconds")
        logger.info(f"Steps per second: {args.steps/training_time:.2f}")
        logger.info(f"Final model: {model_filename}")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False
    
    finally:
        env.close()

def main():
    """Main entry point."""
    try:
        args = parse_args()
        success = run_experiment(args)
        
        if success:
            logger.info("🎉 Experiment completed successfully!")
            return 0
        else:
            logger.error("💥 Experiment failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("🛑 Experiment interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())