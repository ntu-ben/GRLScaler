import logging
import argparse
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from sb3_contrib import RecurrentPPO
import torch 

from gnnrl.core.envs import Redis, OnlineBoutique
from stable_baselines3.common.callbacks import CheckpointCallback

# Logging
from gnnrl.core.util.util import test_model

logging.basicConfig(filename='run.log', filemode='w', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

parser = argparse.ArgumentParser(description='Run ILP!')
parser.add_argument('--alg', default='ppo', help='The algorithm: ["ppo", "recurrent_ppo", "a2c"]')
parser.add_argument('--k8s', default=False, action="store_true", help='K8s mode')
parser.add_argument('--use_case', default='redis', help='Apps: ["redis", "online_boutique"]')
parser.add_argument('--goal', default='cost', help='Reward Goal: ["cost", "latency"]')
parser.add_argument('--gnn_mode', action='store_true', help='Enable GNN based policy')

parser.add_argument('--training', default=False, action="store_true", help='Training mode')
parser.add_argument('--testing', default=False, action="store_true", help='Testing mode')
parser.add_argument('--loading', default=False, action="store_true", help='Loading mode')
parser.add_argument('--load_path', default='logs/model/test.zip', help='Loading path, ex: logs/model/test.zip')
parser.add_argument('--test_path', default='logs/model/test.zip', help='Testing path, ex: logs/model/test.zip')

parser.add_argument('--steps', default=500, help='The steps for saving.')
parser.add_argument('--total_steps', default=5000, help='The total number of steps.')
# --- 新增參數：--device --------------------------
parser.add_argument(
    '--device',
    default='auto',
    choices=['auto', 'cpu', 'cuda', 'mps'],
    help='PyTorch device. "auto" = mps > cuda > cpu'
)
args = parser.parse_args()

def resolve_device(arg: str) -> torch.device:
    """auto → 優先 mps，其次 cuda，再 fallback cpu"""
    if arg == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    return torch.device(arg)

DEVICE = resolve_device(args.device)

def get_model(alg, env, tensorboard_log, use_gnn=False):
    model = 0
    policy = "MlpPolicy"
    if use_gnn:
        from gnn_rl.gnn_policy import GNNActorCriticPolicy
        policy = GNNActorCriticPolicy
    if alg == 'ppo':
        model = PPO(policy, env, verbose=1, tensorboard_log=tensorboard_log, n_steps=500, device=DEVICE)
    elif alg == 'recurrent_ppo':
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=tensorboard_log, device=DEVICE)
    elif alg == 'a2c':
        model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, device=DEVICE)  # , n_steps=steps
    else:
        logging.info('Invalid algorithm!')

    return model


def get_load_model(alg, tensorboard_log, load_path, use_gnn=False):
    if alg == 'ppo':
        policy = "MlpPolicy"
        if use_gnn:
            from gnn_rl.gnn_policy import GNNActorCriticPolicy
            policy = GNNActorCriticPolicy
        return PPO.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log, n_steps=500, device=DEVICE, custom_objects={"policy_class": policy})
    elif alg == 'recurrent_ppo':
        return RecurrentPPO.load(load_path, reset_num_timesteps=False, verbose=1,
                                 tensorboard_log=tensorboard_log, device=DEVICE)  # n_steps=steps
    elif alg == 'a2c':
        return A2C.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log, device=DEVICE)
    else:
        logging.info('Invalid algorithm!')


def get_env(use_case, k8s, goal, use_graph=False):
    env = 0
    if use_case == 'redis':
        env = Redis(k8s=k8s, goal_reward=goal, use_graph=use_graph)
    elif use_case == 'online_boutique':
        env = OnlineBoutique(k8s=k8s, goal_reward=goal, use_graph=use_graph)
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
    use_gnn = args.gnn_mode

    loading = args.loading
    load_path = args.load_path
    training = args.training
    testing = args.testing
    test_path = args.test_path

    steps = int(args.steps)
    total_steps = int(args.total_steps)

    env = get_env(use_case, k8s, goal, use_graph=use_gnn)

    scenario = ''
    if k8s:
        scenario = 'real'
    else:
        scenario = 'simulated'

    tensorboard_log = "../../results/" + use_case + "/" + scenario + "/" + goal + "/"

    gnn_suffix = "_gnn" if use_gnn else ""
    name = alg + gnn_suffix + "_env_" + env.name + "_goal_" + goal + "_k8s_" + str(k8s) + "_totalSteps_" + str(total_steps)

    # callback
    checkpoint_callback = CheckpointCallback(save_freq=steps, save_path="logs/" + name, name_prefix=name)

    if training:
        if loading:  # resume training
            model = get_load_model(alg, tensorboard_log, load_path, use_gnn)
            model.set_env(env)
            model.learn(total_timesteps=total_steps, tb_log_name=name + "_run", callback=checkpoint_callback)
        else:
            model = get_model(alg, env, tensorboard_log, use_gnn)
            model.learn(total_timesteps=total_steps, tb_log_name=name + "_run", callback=checkpoint_callback)

        model.save(name)

    if testing:
        model = get_load_model(alg, tensorboard_log, test_path, use_gnn)
        test_model(model, env, n_episodes=100, n_steps=110, smoothing_window=5, fig_name=name + "_test_reward.png")


if __name__ == "__main__":
    main()
