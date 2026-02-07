"""
src/envs/slicing_env.py

Gymnasium 環境封裝 (Gym Wrapper)。
將 BaseStation 的物理行為轉換為強化學習標準介面 (Env)。

主要功能：
1. 定義 Action Space (資源權重) 與 Observation Space (Buffer狀態 + CQI)。
2. 計算 Reward Function (吞吐量 - 延遲懲罰)。
3. 執行標準 Gym 流程 (Reset, Step)。
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.core.base_station import BaseStation
import src.utils.constants as C

class NetworkSlicingEnv(gym.Env):
    """
    5G 網路切片資源分配環境 (Custom Environment for 5G Slicing).
    
    Observation Space (6維):
        [eMBB_Load, eMBB_HoL, URLLC_Load, URLLC_HoL, CQI_eMBB, CQI_URLLC]
        
    Action Space (2維 Continuous):
        [Weight_eMBB, Weight_URLLC] (數值範圍 0~1)
    """
    
    metadata = {'render_modes': ['human']}

    def __init__(self, config):
        super(NetworkSlicingEnv, self).__init__()
        
        self.config = config
        
        # 1. 建立物理模擬引擎 (Engine)
        self.bs = BaseStation(config)
        
        # 2. 定義動作空間 (Action Space)
        # 輸出兩個權重值 [w_embb, w_urllc]，範圍 0.0 ~ 1.0
        # 注意：雖然我們會正規化，但讓 Agent 輸出 0~1 比較容易收斂
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # 3. 定義觀察空間 (Observation Space)
        # [eMBB_Load, eMBB_HoL, URLLC_Load, URLLC_HoL, CQI_eMBB, CQI_URLLC]
        # 定義上下界：Load 和 HoL 理論上是無限大 (np.inf)，CQI 是 0~15
        low_bound = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        high_bound = np.array(
            [np.inf, np.inf, np.inf, np.inf, C.MAX_CQI, C.MAX_CQI], 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=low_bound, high=high_bound, shape=(6,), dtype=np.float32
        )
        
        # 4. 讀取獎勵參數
        self.w_throughput = config['reward']['w_throughput']
        self.w_latency = config['reward']['w_latency']
        self.drop_penalty = config['reward']['drop_penalty']

        # 5. 約束參數 (可選)
        constraints = config.get('constraints', {})
        self.min_urllc_weight = float(constraints.get('min_urllc_weight', 0.0))
        
        # 狀態追蹤
        self.current_step = 0
        self.max_steps = 2000

    def reset(self, seed=None, options=None):
        """
        重置環境，開始新的 Episode。
        """
        super().reset(seed=seed)
        
        # 重置物理引擎
        self.bs.reset()
        self.current_step = 0
        
        # 獲取初始狀態
        obs = self.bs.get_observation()
        info = {}
        
        return obs, info

    def step(self, action):
        """
        執行一步 (1 TTI)。
        
        Args:
            action (np.array): [w_embb, w_urllc]
            
        Returns:
            obs, reward, terminated, truncated, info
        """
        # 1. 動作前處理：歸一化 + 最小 URLLC 配額約束 (避免資源全給 eMBB)
        action_raw = np.array(action, dtype=np.float32)
        action_clipped = np.clip(action_raw, 0.0, 1.0)
        total_w = float(np.sum(action_clipped))
        if total_w <= 0.0:
            action_norm = np.array([0.5, 0.5], dtype=np.float32)
        else:
            action_norm = action_clipped / total_w

        if self.min_urllc_weight > 0.0:
            w_urllc = max(action_norm[1], self.min_urllc_weight)
            w_embb = 1.0 - w_urllc
            action_applied = np.array([w_embb, w_urllc], dtype=np.float32)
        else:
            action_applied = action_norm

        # 2. 呼叫基地台引擎執行物理計算
        # 引擎會回傳這個 TTI 發生了什麼事 (Throughput, Latency, Drops...)
        info = self.bs.step(action_applied)
        
        # 2. 獲取新狀態 (Next State)
        obs = self.bs.get_observation()
        
        # 3. 計算獎勵 (Reward Function) ⭐️ 論文核心
        # 目標：最大化 eMBB 流量，同時懲罰 URLLC 的延遲和丟包
        
        # (A) 正向獎勵：總吞吐量 (Mbps)
        total_throughput = info['throughput_embb_mbps'] + info['throughput_urllc_mbps']
        reward_throughput = total_throughput * self.w_throughput
        
        # (B) 負向懲罰：URLLC 延遲
        # 延遲越接近 1ms (0.001s)，懲罰越大
        # 將秒換算成毫秒 (x1000) 以配合權重數量級
        latency_ms = info['latency_urllc'] * 1000.0
        reward_latency = latency_ms * self.w_latency
        
        # (C) 嚴重懲罰：URLLC 丟包 (Violation)
        reward_drop = info['dropped_urllc'] * self.drop_penalty
        
        # 總獎勵
        reward = reward_throughput - reward_latency - reward_drop
        
        # 將細項放入 info 以便 TensorBoard 觀察
        info['reward_throughput'] = reward_throughput
        info['reward_latency'] = reward_latency
        info['reward_drop'] = reward_drop
        info['action_raw'] = action_raw
        info['action_applied'] = action_applied
        info['min_urllc_weight'] = self.min_urllc_weight
        
        # 4. 檢查終止條件
        self.current_step += 1
        terminated = False
        truncated = False
        
        # 如果超過設定的步數，結束 Episode
        # 注意：在持續性任務中，通常由外部 Runner 控制，這裡設一個安全上限
        if self.current_step >= self.max_steps:
            truncated = True
            
        return obs, reward, terminated, truncated, info

    def render(self):
        """簡單的文字輸出"""
        pass # 實際訓練通常不 render 以加速


# ==========================================
# 介面驗證 (Interface Check)
# ==========================================
if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env
    
    print("--- Verifying Gym Environment ---")
    
    # 1. 建立 Mock Config
    mock_config = {
        'random_seed': 42,
        'total_timesteps': 1000,
        'traffic': {
            'embb_arrival_rate_mbps': 100.0,
            'urllc_arrival_rate_mbps': 5.0
        },
        'agent': {
            'buffer_size': 10000
        },
        'reward': {
            'w_throughput': 0.1,
            'w_latency': 10.0,
            'drop_penalty': 50.0
        }
    }
    
    # 2. 實例化環境
    env = NetworkSlicingEnv(mock_config)
    
    # 3. 使用 Gym 官方工具檢查
    # 如果有任何不符合標準的地方 (如 dtype 不對, shape 不對)，這裡會報錯
    print("Running check_env()...")
    check_env(env)
    print("✅ check_env() passed! The environment is compatible with Stable-Baselines3.")
    
    # 4. 手動跑幾個 Step 測試
    print("\nRunning manual loop test...")
    obs, _ = env.reset()
    print(f"Initial Observation: {obs}")
    
    for i in range(5):
        # 隨機動作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.4f} (Thr: {info['reward_throughput']:.2f}, Lat: {info['reward_latency']:.2f}, Drop: {info['reward_drop']})")
        print(f"  Obs: {obs}")
        
        if terminated or truncated:
            break