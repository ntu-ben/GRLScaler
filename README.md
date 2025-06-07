# GRLScaler

本專案由 **國立台灣大學** 研究團隊維護，用於研究在 Kubernetes 平台上透過強化學習實現的自動擴縮 (Autoscaler)。專案部分程式碼取自 [gym-hpa](https://github.com/jpedro1992/gym-hpa) ，並在此基礎上新增圖神經網路相關功能與資料集。

## 與 gym-hpa 的主要差異

* 新增支援 **Graph Neural Network (GNN)** 的觀測空間與 `GNNActorCriticPolicy`，可在多服務拓樸下學習資源調度策略。
* 提供真實叢集測試所需的額外資料集與自動化腳本，方便在本地或遠端 Kubernetes 叢集上重現實驗。


The original [gym-hpa](https://github.com/jpedro1992/gym-hpa) provides a custom [OpenAI Gym](https://gym.openai.com/)
environment for the training of Reinforcement Learning (RL) agents in Kubernetes (K8s) clusters.
本專案在其基礎上擴充圖神經網路觀測空間與相關演算法，以探索更複雜服務拓樸下的自動擴縮行為。


## How does it work?

Two environments exist based on the [Redis Cluster](https://github.com/bitnami/charts/tree/master/bitnami/redis-cluster) and [Online Boutique](https://github.com/GoogleCloudPlatform/microservices-demo) applications. 

Both RL environments have been designed: actions, observations, reward function. 

Please check the [run.py](gnn_rl/run/run.py) file to understand how to run the framework.

To run in the real cluster mode, set your Kubernetes API token in the environment variable `K8S_TOKEN`.
You can also copy `.envTemplate` to `.env` and fill in your token so that `gym_hpa/envs/deployment.py` can read it automatically.

### Running

To run the code, go to the folder `gnn_rl/run` and run:

```bash
python run.py
```

Additional arguments can be passed while running run.py. Please check here [run.py](gnn_rl/run/run.py).

### Service graph via Linkerd

The environments obtain service topology from the Linkerd `viz` extension rather
than Jaeger. Make sure `linkerd-viz` is installed and accessible. By default the
library queries `http://metrics-api.linkerd-viz.svc.cluster.local:8085/api/edges`
(configurable via the `LINKERD_VIZ_API_URL` environment variable) to build the adjacency matrix used by
the GNN observation space.

### Deploying Online Boutique with Linkerd

When deploying the Online Boutique demo into your cluster, make sure each
deployment is injected with the Linkerd proxy. If you already applied the
manifest, re-apply it with:

```bash
kubectl -n onlineboutique get deploy -o yaml \
  | linkerd inject - \
  | kubectl apply -f -
```

This step ensures that service edges and metrics are visible through
`linkerd-viz`, allowing the environments in this repository to extract the
service graph correctly.

## Repository layout

```
gnn_rl/        # RL policies and training script
gym_hpa/       # Custom Gym environments
loadtest/      # Locust scenarios and remote agent for load testing
k8s_hpa/       # HPA baseline scripts and Kubernetes manifests
```

The orchestration script `rl_batch_loadtest.py` launches RL training and
triggers Locust scenarios. Set the `M1_HOST` environment variable to the URL of
the remote agent if running distributed tests, otherwise Locust runs locally.

Both `rl_batch_loadtest.py` and `loadtest/locust_agent.py` emit detailed
debug logs. Check `logs/<run-tag>/batch.log` on the runner side and the console
output of the agent to diagnose connectivity issues.



