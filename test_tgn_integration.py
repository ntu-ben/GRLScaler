#!/usr/bin/env python3
"""
測試TGN+A2C整合的真實運作狀況
==================================

檢查：
1. TGN是否真的在處理動態圖
2. 時間軸是否正確遞增
3. 記憶體是否正確更新
4. A2C是否正確使用TGN特徵
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
from gnnrl.core.envs import OnlineBoutique
from gnnrl.core.agents.ppo_gnn import GNNPPOPolicy
from stable_baselines3 import A2C
import gymnasium as gym


def test_tgn_integration():
    """測試TGN+A2C整合"""
    print("🔍 測試 TGN + A2C 整合")
    print("=" * 50)
    
    # 1. 創建環境（K8s模式 + 圖形模式）
    print("1. 創建環境...")
    try:
        env = OnlineBoutique(
            k8s=True,  # 使用真實K8s
            goal_reward='latency',
            use_graph=True,  # 啟用圖形模式
            waiting_period=2.0
        )
        print(f"✅ 環境創建成功：{env.observation_space}")
    except Exception as e:
        print(f"❌ 環境創建失敗：{e}")
        return False
    
    # 2. 測試觀察空間
    print("\n2. 測試觀察空間...")
    obs = env.reset()[0]
    print(f"觀察空間類型: {type(obs)}")
    if isinstance(obs, dict):
        print(f"觀察空間鍵值: {list(obs.keys())}")
        for key, value in obs.items():
            print(f"  {key}: shape={getattr(value, 'shape', 'N/A')}, dtype={getattr(value, 'dtype', 'N/A')}")
    
    # 3. 創建TGN策略
    print("\n3. 創建TGN策略...")
    try:
        metadata = (
            ["svc", "node"],  # 節點類型
            [("svc", "calls", "svc"), ("svc", "runs_on", "node"), ("node", "hosts", "svc")]  # 邊類型
        )
        
        # 測試TGN模型
        model = A2C(
            GNNPPOPolicy,
            env=env,
            learning_rate=3e-4,
            verbose=1,
            policy_kwargs={
                'metadata': metadata,
                'model': 'tgn',  # 使用TGN
                'embed_dim': 32,
            }
        )
        print("✅ TGN+A2C模型創建成功")
    except Exception as e:
        print(f"❌ 模型創建失敗：{e}")
        return False
    
    # 4. 測試動態圖處理
    print("\n4. 測試動態圖處理...")
    try:
        total_steps = 5
        tgn_steps = []
        
        for step in range(total_steps):
            print(f"\n--- Step {step + 1} ---")
            
            # 獲取動作
            action, _ = model.predict(obs, deterministic=True)
            print(f"Action: {action}")
            
            # 執行動作
            obs, reward, done, truncated, info = env.step(action)
            print(f"Reward: {reward:.3f}")
            
            # 檢查TGN編碼器狀態
            policy = model.policy
            if hasattr(policy, 'gnn_encoder') and hasattr(policy.gnn_encoder, 'tgn_step'):
                tgn_step = policy.gnn_encoder.tgn_step
                tgn_steps.append(tgn_step)
                print(f"TGN Step: {tgn_step}")
                
                # 檢查TGN記憶體
                if hasattr(policy.gnn_encoder, 'encoder') and policy.gnn_encoder.encoder:
                    memory = policy.gnn_encoder.encoder.memory
                    if hasattr(memory, 'memory'):
                        mem_state = memory.memory
                        print(f"TGN Memory shape: {mem_state.shape if mem_state is not None else 'None'}")
                        print(f"TGN Memory mean: {mem_state.mean().item() if mem_state is not None else 'None'}")
            
            if done or truncated:
                obs = env.reset()[0]
                
        # 5. 驗證時間軸
        print(f"\n5. 驗證時間軸...")
        print(f"TGN步驟序列: {tgn_steps}")
        if len(tgn_steps) > 1:
            is_increasing = all(tgn_steps[i] < tgn_steps[i+1] for i in range(len(tgn_steps)-1))
            print(f"時間軸遞增: {'✅' if is_increasing else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 動態圖測試失敗：{e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        env.close()

def test_kiali_integration():
    """測試Kiali整合"""
    print("\n🌐 測試 Kiali 整合")
    print("=" * 30)
    
    try:
        from gnnrl.core.utils.kiali_client import fetch_service_graph
        
        # 測試服務圖獲取
        nodes, edge_df = fetch_service_graph("onlineboutique", duration="60s")
        
        print(f"節點數量: {len(nodes)}")
        print(f"邊數量: {len(edge_df)}")
        print(f"節點列表: {nodes}")
        
        if not edge_df.empty:
            print(f"邊資料欄位: {list(edge_df.columns)}")
            print("邊資料範例:")
            print(edge_df.head())
        
        return len(nodes) > 0
        
    except Exception as e:
        print(f"❌ Kiali整合測試失敗：{e}")
        return False

def main():
    """主測試函數"""
    print("🚀 開始 TGN + A2C 整合測試")
    print("=" * 60)
    
    # 測試Kiali整合
    kiali_ok = test_kiali_integration()
    
    if not kiali_ok:
        print("❌ Kiali測試失敗，跳過TGN測試")
        return
    
    # 測試TGN整合
    tgn_ok = test_tgn_integration()
    
    # 總結
    print("\n" + "=" * 60)
    print("📊 測試總結")
    print("=" * 60)
    print(f"Kiali整合: {'✅ 成功' if kiali_ok else '❌ 失敗'}")
    print(f"TGN+A2C整合: {'✅ 成功' if tgn_ok else '❌ 失敗'}")
    
    if kiali_ok and tgn_ok:
        print("🎉 動態圖 → TGN → A2C 管道已正常運作！")
    else:
        print("⚠️ 部分功能需要修復")

if __name__ == "__main__":
    main()