# GNN + RL Autoscaler 架構設計

以下內容說明如何將現有的 Istio/Kiali + Prometheus + Kubernetes 單機開發環境
與 GNN-RL 文獻 (包含 Song et al. fjsp-drl) 結合，建構可落地的 Autoscaler。

## 1 資料→異質圖 (HeteroGraph) 流程

| 步驟 | 來源 | 轉換 | 去向 |
| --- | --- | --- | --- |
| **① 服務呼叫** `graph` | Kiali REST `/api/namespaces/{ns}/graph` | 建立 *svc→svc* 邊 (特徵: rps, p95, http_error) | ➜ **GNN** 邊特徵 |
| **② 低階資源** | Prometheus `container_cpu_usage_seconds_total` 等 | 聚合至 *svc node* 特徵 μ_i = [cpu%, mem%, p95_latency, stateful_flag, RUE] | ➜ **GNN** 節點特徵 |
| **③ 節點硬體** | K8s API (`/api/v1/nodes`) | 節點 ν_k =[avail_cpu, avail_mem, cpu_model_id] | ➜ **GNN** 節點特徵 |
| **④ 全域指標** | Prometheus | cluster_avg_cpu, cluster_SLO_vio_rate | ➜ **RL 直接輸入** (tabular) |
| **⑤ Stateful lag** | Redis/MySQL Exporter | replica_lag_sec, hit_ratio | ➜ **GNN** (svc node特徵) + **RL** (reward) |

> **為何要分流?**
> - 與拓撲/依賴相關的資訊（①②③⑤）交給 **GNN** 學習 "誰影響誰"。
> - 與全域狀態無關的 scalar (④) 直接給 **RL**，避免訊號被稀釋。

### 1.1 轉為 PyG `HeteroData`
```python
from torch_geometric.data import HeteroData

data = HeteroData()
data['svc'].x = torch.tensor(mu_i) # [N_svc, F_svc]
data['node'].x = torch.tensor(nu_k) # [N_node, F_node]
data['svc', 'calls', 'svc'].edge_index = edge_idx_calls
data['svc', 'runs_on', 'node'].edge_index = edge_idx_runs_on
data['node', 'hosts', 'svc'].edge_index = edge_idx_hosts
data['svc','calls','svc'].edge_attr = torch.tensor(edge_feat) # [E, F_edge]
```

## 2 GNN 層設計

| 選項 | 優點 | 代價 / 場景 |
| --- | --- | --- |
| **GCN** | 結構簡單，適合初始實驗；PyG `GCNConv` | 只能捕捉一階鄰居加權平均 |
| **GAT / HAN** | 邊權注意力、可異質圖；PyG `GATConv`、`HANConv` | 計算量大；需調整 head 數 |
| **TGAT / DySAT** | 捕捉時間演變；可處理秒級 graph stream | 需儲存歷史 snapshot；實作稍複雜 |

**起步建議**：使用 `HeteroConv` + GATConv kernel，2–3 層，每層 head=4，embedding=128。
若需要天級變化，可加入 `torch_geometric_temporal` 內的 TGAT。

## 3 RL 策略網路

### 3.1 狀態表示
```text
s_t = [
    mean_pool(h_svc),
    mean_pool(h_node),
    cluster_avg_cpu,
    SLO_violate%,
]
```

### 3.2 動作空間 (離散×連續)

| 分量 | 意義 | 編碼 |
| --- | --- | --- |
| `a1` | 要調整的 svc_id (離散 N) | Categorical |
| `a2` | node_type / cpu_model (離散 K) | Categorical |
| `a3` | Δreplicas ∈ {−2…+2} | Categorical |
| `a4` | Δquota_norm ∈ [-1,1] | Beta 或 Tanh-Gaussian |

### 3.3 演算法選型

| Algo | 連續+離散多頭 | Sample-efficiency | 易用度 |
| --- | --- | --- | --- |
| **PPO** | ✔ (多頭 softmax + β-dist) | 中 | SB3 強 |
| SAC (離散版) | ✔ | 高 | CleanRL / RLlib |
| Hybrid-DQN | 部份 | 中低 | 需自寫 |

> **推薦流程**：先以 **SB3 PPO** 開發，若 sample-efficiency 不足再嘗試離散版 SAC。

### 3.4 Reward

$$
r_t = -5\,\text{SLO\_vio} -1\,\text{Slack} -3\,\text{Overshoot} -2\,\text{ReplicaLag}
$$

## 4 比較基線 (GNN × RL)

| 代號 | GNN Encoder | RL Algo | 用途 |
| --- | --- | --- | --- |
| **MLP-PPO** | 無 (features 展平成表) | PPO | 測試「無拓撲」 |
| **GAT-PPO** | HeteroGAT | PPO | 主模型 |
| **GCN-SAC** | HeteroGCN | Discrete-SAC | 考查另一組 |
| **DySAT-A2C** | DySAT encoder | A2C | 時序圖對比 |

## 5 完整需求清單

### 5.1 軟體

| 類別 | 套件 | 版本建議 |
| --- | --- | --- |
| Python | 3.10 | |
| 深度學習 | `torch==2.2.*` | GPU / CPU |
| 圖學習 | `torch_geometric>=2.5`；若用 TGAT 再裝 `torch_geometric_temporal` | |
| RL | `stable-baselines3>=2.3.0`、`cleanrl` | |
| K8s API | `kubernetes>=28.0` | |
| 監控 | `prometheus-api-client`、`requests` | |
| 其他 | `pandas`/`polars`, `typer`, `rich`, `uvicorn`, `fastapi` | |

### 5.2 硬體

- **開發**：單台 x86 工作站，16 GB RAM；NVIDIA GPU (12 GB VRAM) 建議
- **線上推論**：CPU 足矣，可封裝成輕量容器

### 5.3 Data Pipeline

| 項 | 週期 | 格式 |
| --- | --- | --- |
| Kiali Graph | 30 s | JSON |
| Prometheus Range Query | 30 s | CSV / Dict |
| K8s Pod / Node | 30 s | JSON |
| **同步聚合**：內存 `polars.DataFrame` → 轉 `HeteroData` |  |  |

### 5.4 評估範式

- **環境**：MicroK8s + Online Boutique，`kubectl port-forward`
- **case**：Off-peak / Peak / Rush / Fluctuating 四種 Locust 載入
- **指標**：SLO_violation%、Resource Slack%、Scaling Lag、CAF
- **統計**：5 次重播，每次 4 hr；報告 95% CI + Cliff’s δ

## 6 實作里程碑

| 項目 | 任務 |
| --- | --- |
| 1 | REST 抓取器 + ETL 骨架 (`polars`) |
| 2 | PyG 異質圖生成 + GAT Encoder |
| 3 | MLP-PPO baseline 通跑 |
| 4 | GAT-PPO 收斂；加入 quota/replica action |
| 5 | Mask & Stateful lag，reward 完整化 |
| 6 | GCN-SAC baseline；做 ablation |
| 7 | DySAT-A2C 時序圖對比 |
| 8 | 評估腳本 (CAF, Slack, SLO) + TensorBoard/W&B |
| 9 | 整理論文草稿 + 圖表 & ablation 結果 |

---

此架構保留 Song et al. HGNN+PPO 一次挑選 (服務,節點) 的優點，並加入微服務專屬指標（Istio latency、replica lag、RUE），同時提供多種 GNN / RL 模型替換空間，適合逐步實作與評估。
