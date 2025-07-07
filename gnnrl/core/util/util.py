import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path


def test_model(model, env, n_episodes, n_steps, smoothing_window, fig_name):
    episode_rewards = []
    reward_sum = 0
    obs = env.reset()

    print("------------Testing -----------------")

    for e in range(n_episodes):
        for _ in range(n_steps):
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            if done:
                episode_rewards.append(reward_sum)
                print("Episode {} | Total reward: {} |".format(e, str(reward_sum)))
                reward_sum = 0
                obs = env.reset()
                break

    env.close()

    # Free memory
    del model, env

    # Plot the episode reward over time
    plt.figure()
    rewards_smoothed = pd.Series(episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    
    # 確保圖片保存到 logs/gnnrl/charts/ 目錄
    charts_dir = Path("logs/gnnrl/charts")
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # 從 fig_name 中提取文件名
    fig_path = Path(fig_name)
    chart_file = charts_dir / fig_path.name
    
    plt.savefig(chart_file, dpi=250, bbox_inches='tight')
