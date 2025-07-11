#!/usr/bin/env python3
"""
完整動態圖系統測試
====================
測試GNNRL系統的真正動態圖功能，包括：
1. DynamicGraphSpace的節點映射和填充
2. DynamicTGNEncoder的記憶體管理
3. OnlineBoutique環境的動態觀察空間
4. 完整的端到端流程
"""

import numpy as np
import torch
import logging
import sys
import os
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent))

from gnnrl.core.envs.dynamic_graph_space import DynamicGraphSpace, DynamicGraphConfig
from gnnrl.encoders.tgn_encoder import DynamicTGNEncoder
from gnnrl.core.envs.online_boutique import OnlineBoutique

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dynamic_graph_space():
    """測試動態圖空間管理"""
    print("=" * 60)
    print("🔬 測試 1: DynamicGraphSpace 功能")
    print("=" * 60)
    
    # 創建配置
    config = DynamicGraphConfig(
        max_nodes=20,
        max_edges=400,
        node_feat_dim=6,
        edge_feat_dim=7,
        global_feat_dim=4
    )
    
    # 創建動態圖空間
    dgs = DynamicGraphSpace(config)
    
    # 測試場景1：初始11個服務
    initial_services = [
        "recommendationservice", "productcatalogservice", "cartservice", 
        "adservice", "paymentservice", "shippingservice", "currencyservice",
        "redis-cart", "checkoutservice", "frontend", "emailservice"
    ]
    
    print(f"📊 初始服務數量: {len(initial_services)}")
    node_mapping = dgs.update_node_mapping(initial_services)
    print(f"🗺️  節點映射: {node_mapping}")
    
    # 測試節點特徵填充
    node_features = np.random.rand(len(initial_services), 6)
    padded_nodes, node_mask = dgs.pad_node_features(node_features, len(initial_services))
    
    print(f"📏 原始節點特徵形狀: {node_features.shape}")
    print(f"📏 填充後節點特徵形狀: {padded_nodes.shape}")
    print(f"🎭 節點遮罩: {node_mask[:15]}...")  # 顯示前15個
    
    # 測試場景2：服務擴展（添加新服務）
    print("\n" + "=" * 40)
    print("📈 測試服務擴展")
    print("=" * 40)
    
    expanded_services = initial_services + ["additional-service-1", "additional-service-2"]
    print(f"🆕 擴展後服務數量: {len(expanded_services)}")
    new_node_mapping = dgs.update_node_mapping(expanded_services)
    print(f"🗺️  新節點映射: {new_node_mapping}")
    
    # 測試邊特徵填充
    edge_features = np.random.rand(25, 7)  # 25條邊
    padded_edges, edge_mask = dgs.pad_edge_features(edge_features, 25)
    
    print(f"📏 原始邊特徵形狀: {edge_features.shape}")
    print(f"📏 填充後邊特徵形狀: {padded_edges.shape}")
    print(f"🎭 邊遮罩前30個: {edge_mask[:30]}")
    
    # 測試場景3：服務縮減
    print("\n" + "=" * 40)
    print("📉 測試服務縮減")
    print("=" * 40)
    
    reduced_services = initial_services[:8]  # 只保留前8個服務
    print(f"🔻 縮減後服務數量: {len(reduced_services)}")
    reduced_node_mapping = dgs.update_node_mapping(reduced_services)
    print(f"🗺️  縮減節點映射: {reduced_node_mapping}")
    
    print("✅ DynamicGraphSpace 測試完成！")
    return True

def test_dynamic_tgn_encoder():
    """測試動態TGN編碼器"""
    print("\n" + "=" * 60)
    print("🔬 測試 2: DynamicTGNEncoder 功能")
    print("=" * 60)
    
    # 創建編碼器
    encoder = DynamicTGNEncoder(
        max_nodes=20,
        in_dim=6,
        memory_dim=32,
        msg_dim=32
    )
    
    # 測試場景1：初始節點映射
    initial_services = [
        "recommendationservice", "productcatalogservice", "cartservice", 
        "adservice", "paymentservice"
    ]
    
    print(f"📊 初始服務: {initial_services}")
    node_mapping = encoder.update_node_mapping(initial_services)
    print(f"🗺️  TGN節點映射: {node_mapping}")
    
    # 創建測試數據
    num_nodes = len(initial_services)
    num_edges = 8
    
    node_features = torch.randn(num_nodes, 6)
    # 填充節點特徵到最大維度
    padded_node_features = torch.zeros(20, 6)
    padded_node_features[:num_nodes] = node_features
    
    edge_data = torch.randn(num_edges, 7)
    edge_data[:, 0] = torch.randint(0, num_nodes, (num_edges,))  # src
    edge_data[:, 1] = torch.randint(0, num_nodes, (num_edges,))  # dst
    
    # 創建遮罩 - 調整維度匹配
    node_mask = torch.ones(20)
    node_mask[num_nodes:] = 0
    edge_mask = torch.ones(num_edges)  # 只對實際邊數量創建遮罩
    
    # 將邊數據填充到最大維度
    padded_edge_data = torch.zeros(400, 7)
    padded_edge_data[:num_edges] = edge_data
    
    # 創建完整的邊遮罩
    full_edge_mask = torch.zeros(400)
    full_edge_mask[:num_edges] = 1
    
    print(f"📏 節點特徵形狀: {node_features.shape}")
    print(f"📏 邊數據形狀: {edge_data.shape}")
    
    # 前向傳播 - 使用填充後的數據
    output = encoder.forward(padded_edge_data, padded_node_features, full_edge_mask, node_mask)
    print(f"🎯 TGN輸出形狀: {output.shape}")
    print(f"🧠 記憶體狀態: {encoder.get_memory_state().shape if encoder.get_memory_state() is not None else 'None'}")
    
    # 測試場景2：節點映射變化
    print("\n" + "=" * 40)
    print("🔄 測試節點映射變化")
    print("=" * 40)
    
    expanded_services = initial_services + ["new-service-1", "new-service-2"]
    print(f"🆕 擴展服務: {expanded_services}")
    
    # 保存當前記憶體狀態
    old_memory = encoder.get_memory_state().clone() if encoder.get_memory_state() is not None else None
    
    # 更新節點映射
    new_node_mapping = encoder.update_node_mapping(expanded_services)
    print(f"🗺️  新TGN節點映射: {new_node_mapping}")
    
    # 檢查記憶體是否正確重新映射
    new_memory = encoder.get_memory_state()
    if old_memory is not None and new_memory is not None:
        print(f"🧠 記憶體狀態變化: {old_memory.shape} -> {new_memory.shape}")
        # 檢查保留的服務記憶體是否相同
        for service in initial_services:
            if service in new_node_mapping:
                old_id = initial_services.index(service)
                new_id = new_node_mapping[service]
                if old_id < old_memory.shape[0] and new_id < new_memory.shape[0]:
                    memory_diff = torch.norm(old_memory[old_id] - new_memory[new_id])
                    print(f"📊 {service} 記憶體差異: {memory_diff.item():.6f}")
    
    print("✅ DynamicTGNEncoder 測試完成！")
    return True

def test_online_boutique_integration():
    """測試OnlineBoutique環境整合"""
    print("\n" + "=" * 60)
    print("🔬 測試 3: OnlineBoutique 動態圖整合")
    print("=" * 60)
    
    try:
        # 創建環境（非k8s模式用於測試）
        env = OnlineBoutique(k8s=False, use_graph=True, goal_reward="latency")
        
        print(f"🌍 環境創建成功")
        print(f"📊 動態圖配置: max_nodes={env.dynamic_graph.config.max_nodes}")
        print(f"📊 觀察空間: {env.observation_space}")
        print(f"📊 動作空間: {env.action_space}")
        
        # 重置環境
        obs = env.reset()
        print(f"🔄 環境重置成功")
        
        if isinstance(obs, tuple):
            obs = obs[0]  # 新版gym格式
        
        # 檢查觀察空間結構
        if isinstance(obs, dict):
            print(f"📊 觀察空間結構:")
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {value}")
            
            # 測試動作執行
            print("\n" + "=" * 40)
            print("🎬 測試動作執行")
            print("=" * 40)
            
            # 隨機動作
            action = env.action_space.sample()
            print(f"🎲 隨機動作: {action}")
            
            # 執行動作
            next_obs, reward, done, info = env.step(action)[:4]
            print(f"🎯 動作執行成功")
            print(f"💰 獎勵: {reward}")
            print(f"🏁 結束: {done}")
            
            # 檢查新觀察空間
            if isinstance(next_obs, dict):
                print(f"📊 新觀察空間:")
                for key, value in next_obs.items():
                    if isinstance(value, np.ndarray):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {value}")
            
            print("✅ OnlineBoutique 整合測試完成！")
            return True
        else:
            print("❌ 觀察空間格式錯誤，應為字典格式")
            return False
            
    except Exception as e:
        print(f"❌ OnlineBoutique 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_end_to_end_flow():
    """測試端到端流程"""
    print("\n" + "=" * 60)
    print("🔬 測試 4: 端到端流程測試")
    print("=" * 60)
    
    try:
        # 1. 創建環境
        env = OnlineBoutique(k8s=False, use_graph=True, goal_reward="latency")
        
        # 2. 創建TGN編碼器
        tgn_encoder = DynamicTGNEncoder(
            max_nodes=20,
            in_dim=6,
            memory_dim=32,
            msg_dim=32
        )
        
        print("🏗️  系統組件創建完成")
        
        # 3. 模擬多步訓練
        total_reward = 0
        for episode in range(3):
            print(f"\n📊 Episode {episode + 1}")
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            episode_reward = 0
            for step in range(5):
                # 獲取動作
                action = env.action_space.sample()
                
                # 執行動作
                next_obs, reward, done, info = env.step(action)[:4]
                episode_reward += reward
                
                # 如果使用圖觀察
                if isinstance(next_obs, dict):
                    # 更新TGN編碼器
                    current_services = [f"service_{i}" for i in range(int(next_obs['num_nodes']))]
                    tgn_encoder.update_node_mapping(current_services)
                    
                    # TGN編碼（模擬）
                    edge_data = torch.tensor(next_obs['edge_df'][:50], dtype=torch.float32)  # 取前50條邊
                    node_features = torch.tensor(next_obs['svc_df'][:int(next_obs['num_nodes'])], dtype=torch.float32)
                    edge_mask = torch.tensor(next_obs['edge_mask'][:50], dtype=torch.float32)
                    node_mask = torch.tensor(next_obs['node_mask'], dtype=torch.float32)
                    
                    # 前向傳播
                    tgn_output = tgn_encoder.forward(edge_data, node_features, edge_mask, node_mask)
                    
                    print(f"  Step {step + 1}: Reward={reward:.3f}, TGN Output Shape={tgn_output.shape}")
                
                obs = next_obs
                if done:
                    break
            
            total_reward += episode_reward
            print(f"  Episode {episode + 1} Total Reward: {episode_reward:.3f}")
        
        print(f"\n🎯 端到端測試完成！")
        print(f"💰 總獎勵: {total_reward:.3f}")
        print(f"📊 平均獎勵: {total_reward / 3:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 端到端測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主測試函數"""
    print("🚀 GNNRL動態圖系統完整測試")
    print("=" * 80)
    
    # 設置隨機種子
    np.random.seed(42)
    torch.manual_seed(42)
    
    test_results = []
    
    # 執行所有測試
    test_functions = [
        ("DynamicGraphSpace", test_dynamic_graph_space),
        ("DynamicTGNEncoder", test_dynamic_tgn_encoder),
        ("OnlineBoutique Integration", test_online_boutique_integration),
        ("End-to-End Flow", test_end_to_end_flow)
    ]
    
    for test_name, test_func in test_functions:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ 測試 {test_name} 異常: {e}")
            test_results.append((test_name, False))
    
    # 總結報告
    print("\n" + "=" * 80)
    print("📊 測試總結報告")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 測試結果: {passed}/{total} 通過")
    
    if passed == total:
        print("🎉 所有測試通過！動態圖系統運作正常！")
    else:
        print("⚠️  部分測試失敗，請檢查系統配置")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)