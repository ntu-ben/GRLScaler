# GNNRL (Graph Neural Network Reinforcement Learning) Module

This directory contains the unified GNNRL implementation for Kubernetes autoscaling.

## Directory Structure

```
gnnrl/
├── core/                    # Core GNN-RL implementation
│   ├── agents/             # RL agents (PPO with GNN)
│   ├── envs/               # Environment implementations
│   ├── models/             # Neural network models
│   ├── common/             # Common utilities
│   ├── run/                # Execution scripts
│   └── util/               # General utilities
├── environments/           # Additional environment packages
├── training/               # Training and execution scripts
├── testing/                # Test files and fixtures
├── data/                   # Data files and datasets
├── docs/                   # Documentation
└── benchmarks/             # Benchmarking and analysis tools
```

## Key Components

- **Core Module**: Main GNN-RL implementation with policy, agents, and environments
- **Training Scripts**: Scripts for training and running GNNRL models
- **Testing**: Comprehensive test suite for all components
- **Data**: Graph definitions, datasets, and observations
- **Benchmarks**: Performance analysis and comparison tools

## Usage

Training scripts are located in the `training/` directory:
- `train_gnnppo.py` - Train GNN-PPO models
- `run_onlineboutique_gnn.py` - Run Online Boutique experiments
- `rl_batch_loadtest.py` - Batch load testing

## Testing

Run tests from the `testing/` directory to verify functionality.