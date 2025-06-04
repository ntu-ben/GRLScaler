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

Please check the [run.py](policies/run/run.py) file to understand how to run the framework. 

To run in the real cluster mode, you should add the token to your cluster [here](gym_hpa/envs/deployment.py)

### Running

To run the code, go to the folder `policies/run` and run:

```bash
python run.py
```

Additional arguments can be passed while running run.py. Please check here [run.py](policies/run/run.py). 

## Team

* [Jose Santos](https://scholar.google.com/citations?hl=en&user=57EIYWcAAAAJ)

* [Tim Wauters](https://scholar.google.com/citations?hl=en&user=Kvxp9iYAAAAJ)

* [Bruno Volckaert](https://scholar.google.com/citations?hl=en&user=NIILGOMAAAAJ)

* [Filip de Turck](https://scholar.google.com/citations?hl=en&user=-HXXnmEAAAAJ)

## Contact

If you want to contribute, please contact:

Lead developer: [Jose Santos](https://github.com/jpedro1992/)

For questions or support, please use GitHub's issue system.

## License

Copyright (c) 2020 Ghent University and IMEC vzw.

Address: IDLab, Ghent University, iGent Toren, Technologiepark-Zwijnaarde 126 B-9052 Gent, Belgium


