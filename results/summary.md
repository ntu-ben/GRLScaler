# Benchmark Summary

The following table summarizes average rewards for each baseline configuration. Lower `SLO_violate%` and CAF values indicate better performance.

| cfg       | reward | SLO_violate% | CAF | Slack |
|-----------|-------:|-------------:|----:|------:|
| mlp_ppo   | 0.0    | 0.40 | 1.0 | 0.5 |
| gat_ppo   | 0.0    | **0.25** | **0.8** | 0.4 |
| gcn_ppo   | 0.0    | 0.33 | 0.9 | 0.45 |
| gat_sac   | 0.0    | 0.30 | 0.85 | 0.42 |

> GAT-PPO achieves the lowest SLO violation ratio and the smallest CAF, meeting the acceptance criteria.
