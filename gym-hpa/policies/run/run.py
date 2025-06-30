import logging
import argparse
import sys
import os
from pathlib import Path

# Add the parent directory to Python path for gym_hpa imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3 import A2C
from sb3_contrib import RecurrentPPO

from gym_hpa.envs import Redis, OnlineBoutique
from stable_baselines3.common.callbacks import CheckpointCallback

# Logging
from policies.util.util import test_model

# Use unified logs directory for log files
log_dir = Path(__file__).parent.parent.parent.parent / "logs" / "gym-hpa"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=str(log_dir / 'run.log'), filemode='w', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

parser = argparse.ArgumentParser(description='Run ILP!')
parser.add_argument('--alg', default='ppo', help='The algorithm: ["ppo", "recurrent_ppo", "a2c"]')
parser.add_argument('--k8s', default=False, action="store_true", help='K8s mode')
parser.add_argument('--use_case', default='redis', help='Apps: ["redis", "online_boutique"]')
parser.add_argument('--goal', default='cost', help='Reward Goal: ["cost", "latency"]')

parser.add_argument('--training', default=False, action="store_true", help='Training mode')
parser.add_argument('--testing', default=False, action="store_true", help='Testing mode')
parser.add_argument('--loading', default=False, action="store_true", help='Loading mode')
parser.add_argument('--load_path', default='logs/models/test.zip', help='Loading path, ex: logs/models/test.zip')
parser.add_argument('--test_path', default='logs/models/test.zip', help='Testing path, ex: logs/models/test.zip')

parser.add_argument('--steps', default=500, help='The steps for saving.')
parser.add_argument('--total_steps', default=5000, help='The total number of steps.')

args = parser.parse_args()


def get_model(alg, env, tensorboard_log):
    model = 0
    if alg == 'ppo':
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, n_steps=500)
    elif alg == 'recurrent_ppo':
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
    elif alg == 'a2c':
        model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)  # , n_steps=steps
    else:
        logging.info('Invalid algorithm!')

    return model


def get_load_model(alg, tensorboard_log, load_path):
    # Remove .zip extension if present, as stable_baselines3 adds it automatically
    if load_path.endswith('.zip'):
        load_path = load_path[:-4]
    
    if alg == 'ppo':
        return PPO.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log, n_steps=500)
    elif alg == 'recurrent_ppo':
        return RecurrentPPO.load(load_path, reset_num_timesteps=False, verbose=1,
                                 tensorboard_log=tensorboard_log)  # n_steps=steps
    elif alg == 'a2c':
        return A2C.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log)
    else:
        logging.info('Invalid algorithm!')


def get_env(use_case, k8s, goal):
    env = 0
    if use_case == 'redis':
        env = Redis(k8s=k8s, goal_reward=goal)
    elif use_case == 'online_boutique':
        env = OnlineBoutique(k8s=k8s, goal_reward=goal)
    else:
        logging.error('Invalid use_case!')
        raise ValueError('Invalid use_case!')

    return env


def main():
    # Import and initialize Environment
    logging.info(args)

    alg = args.alg
    k8s = args.k8s
    use_case = args.use_case
    goal = args.goal

    loading = args.loading
    load_path = args.load_path
    training = args.training
    testing = args.testing
    test_path = args.test_path

    steps = int(args.steps)
    total_steps = int(args.total_steps)

    env = get_env(use_case, k8s, goal)

    scenario = ''
    if k8s:
        scenario = 'real'
    else:
        scenario = 'simulated'

    # 使用絕對路徑，基於專案根目錄 - 統一到 logs 目錄
    from pathlib import Path
    results_dir = Path(__file__).parent.parent.parent.parent / "logs" / "gym-hpa" / "tensorboard" / use_case / scenario / goal
    results_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log = str(results_dir) + "/"

    name = alg + "_env_" + env.name + "_goal_" + goal + "_k8s_" + str(k8s) + "_totalSteps_" + str(total_steps)

    # callback - use unified logs directory
    unified_log_path = Path(__file__).parent.parent.parent.parent / "logs" / "gym-hpa" / name
    unified_log_path.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = CheckpointCallback(save_freq=steps, save_path=str(unified_log_path), name_prefix=name)

    if training:
        if loading:  # resume training
            model = get_load_model(alg, tensorboard_log, load_path)
            model.set_env(env)
            model.learn(total_timesteps=total_steps, tb_log_name=name + "_run", callback=checkpoint_callback)
        else:
            model = get_model(alg, env, tensorboard_log)
            model.learn(total_timesteps=total_steps, tb_log_name=name + "_run", callback=checkpoint_callback)

        # Save model to unified models directory
        model_dir = Path(__file__).parent.parent.parent.parent / "logs" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{name}.zip"
        model.save(str(model_path))

    if testing:
        model = get_load_model(alg, tensorboard_log, test_path)
        test_model(model, env, n_episodes=100, n_steps=110, smoothing_window=5, fig_name=name + "_test_reward.png")


if __name__ == "__main__":
    main()
