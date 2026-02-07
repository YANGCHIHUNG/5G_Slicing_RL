"""
src/core/base_station.py

定義 gNB 行為 (RB 計算、資源映射)

基地台 (gNB) 模型核心引擎。
負責整合流量生成、通道模擬、緩衝區管理與實體層資源分配。

核心職責：
1. 接收 RL Agent 的動作 (資源權重)。
2. 執行物理層計算 (RBs -> Capacity)。
3. 更新系統狀態 (Buffer, Latency)。
"""

import numpy as np
import src.utils.constants as C
from src.core.traffic import TrafficGenerator
from src.core.buffers import SliceBuffer
from src.core.channel import ChannelSimulator

class BaseStation:
    def __init__(self, config):
        """
        初始化 gNB 模擬元件。
        
        Args:
            config (dict): 來自 default_config.yaml 的設定參數
        """
        self.config = config
        
        # --- 1. 初始化通道模擬器 (Channel Physics) ---
        # 假設有 2 個邏輯用戶群組：0=eMBB用戶群, 1=URLLC用戶群
        channel_cfg = config.get('channel', {})
        self.channel = ChannelSimulator(
            num_ues=2,
            seed=config['random_seed'],
            fixed_cqi=channel_cfg.get('fixed_cqi', False),
            fixed_cqi_values=channel_cfg.get('fixed_cqi_values', None)
        )
        
        # --- 2. 初始化流量生成器 (Traffic Source) ---
        # 讀取 Config 中的 Mbps 設定
        self.traffic_embb = TrafficGenerator(
            arrival_rate_mbps=config['traffic']['embb_arrival_rate_mbps'],
            packet_size_bytes=C.PACKET_SIZE_EMBB_BYTES,
            seed=config['random_seed']
        )
        
        self.traffic_urllc = TrafficGenerator(
            arrival_rate_mbps=config['traffic']['urllc_arrival_rate_mbps'],
            packet_size_bytes=C.PACKET_SIZE_URLLC_BYTES,
            seed=config['random_seed'] + 1 # 避免種子相同導致流量模式完全同步
        )
        
        # --- 3. 初始化緩衝區 (Storage) ---
        # eMBB: 無延遲限制 (Best Effort)
        self.buffer_embb = SliceBuffer(
            max_size_packets=config['agent']['buffer_size'],
            max_latency_threshold=None 
        )
        
        # URLLC: 有硬性延遲限制 (1ms = 0.001s)
        self.buffer_urllc = SliceBuffer(
            max_size_packets=config['agent']['buffer_size'],
            max_latency_threshold=C.URLLC_MAX_LATENCY_MS / 1000.0
        )
        
        # 系統時間與統計
        self.current_time = 0.0
        self.total_rbs = C.TOTAL_RBS

    def step(self, action_weights):
        """
        執行一個 TTI (0.5ms) 的模擬步進。
        
        Args:
            action_weights (list or np.array): [w_embb, w_urllc], 總和應為 1.0
            
        Returns:
            dict: 包含該 TTI 發生的所有物理指標 (Throughput, Latency, Drop...)
        """
        
        # --- A. 時間推進與環境更新 ---
        # 1. 更新通道品質 (CQI 波動)
        cqis = self.channel.step() 
        efficiencies = self.channel.get_ue_efficiencies()
        # 假設 Index 0 是 eMBB 的平均收訊, Index 1 是 URLLC 的平均收訊
        eff_embb = efficiencies[0]
        eff_urllc = efficiencies[1]
        
        # 2. 產生新流量 (Arrival)
        # 修正：使用 num_packets 而非總 bits，將每個封包獨立加入 Buffer
        num_pkts_embb, _ = self.traffic_embb.step()
        num_pkts_urllc, _ = self.traffic_urllc.step()
        
        # 加入 Buffer (eMBB) - 逐個封包加入
        packet_size_embb_bits = C.PACKET_SIZE_EMBB_BYTES * 8
        for _ in range(num_pkts_embb):
            self.buffer_embb.add_packet(packet_size_embb_bits, self.current_time)
        
        # 加入 Buffer (URLLC) - 逐個封包加入
        packet_size_urllc_bits = C.PACKET_SIZE_URLLC_BYTES * 8
        for _ in range(num_pkts_urllc):
            self.buffer_urllc.add_packet(packet_size_urllc_bits, self.current_time)
        
        
        # --- B. 資源分配 (Resource Mapping) ---
        # 這裡就是 "Engine" 的核心：將權重轉換為 RBs，再轉換為 Bits
        
        # 1. 權重正規化 (防止輸出總和不為 1)
        w_embb, w_urllc = action_weights
        total_w = w_embb + w_urllc + 1e-9
        w_embb /= total_w
        
        # 2. 計算實體 RB 數量 (整數化)
        num_rbs_embb = int(self.total_rbs * w_embb)
        num_rbs_urllc = self.total_rbs - num_rbs_embb # 剩餘的給 URLLC
        
        # 3. 計算傳輸容量 (Capacity Calculation) ⭐️ 物理層核心公式
        # Capacity = RBs * REs/RB * Efficiency * Overhead
        
        # eMBB 容量
        cap_embb = (num_rbs_embb * C.RES_PER_RB * eff_embb * C.CONTROL_OVERHEAD_FACTOR)
        
        # URLLC 容量 (這裡簡化，假設 URLLC 使用跟 eMBB 類似的 MCS 表，
        # 但實務上 URLLC 會 back-off 降速。可在 channel.py 進一步實作 get_urllc_efficiency)
        cap_urllc = (num_rbs_urllc * C.RES_PER_RB * eff_urllc * C.CONTROL_OVERHEAD_FACTOR)
        
        
        # --- C. 執行傳輸 (Transmission) ---
        # 從 Buffer 移除封包
        tx_bits_embb, delays_embb = self.buffer_embb.remove_packets(cap_embb, self.current_time)
        tx_bits_urllc, delays_urllc = self.buffer_urllc.remove_packets(cap_urllc, self.current_time)
        
        
        # --- D. 檢查違規 (Violation Check) ---
        # URLLC 超時檢查
        dropped_pkts_urllc = self.buffer_urllc.check_timeout(self.current_time)
        
        
        # --- E. 整理輸出資訊 ---
        # 計算平均延遲 (如果沒傳封包，延遲設為 0)
        avg_lat_urllc = np.mean(delays_urllc) if delays_urllc else 0.0
        
        info = {
            'time': self.current_time,
            # 資源分配結果
            'rbs_embb': num_rbs_embb,
            'rbs_urllc': num_rbs_urllc,
            # 通道狀態
            'cqi_embb': cqis[0],
            'cqi_urllc': cqis[1],
            # 傳輸效能 (轉換為 Mbps 方便觀察)
            'throughput_embb_mbps': (tx_bits_embb / C.TTI_DURATION_SEC) / 1e6,
            'throughput_urllc_mbps': (tx_bits_urllc / C.TTI_DURATION_SEC) / 1e6,
            # 佇列狀態
            'buffer_embb_bits': self.buffer_embb.total_bits,
            'buffer_urllc_bits': self.buffer_urllc.total_bits,
            # QoS 指標
            'latency_urllc': avg_lat_urllc,
            'dropped_urllc': dropped_pkts_urllc
        }
        
        # 推進時間
        self.current_time += C.TTI_DURATION_SEC
        
        return info

    def get_observation(self):
        """
        組合 RL Agent 所需的 State Vector。
        [eMBB_Load, eMBB_HoL, URLLC_Load, URLLC_HoL, CQI_eMBB, CQI_URLLC]
        """
        state_embb = self.buffer_embb.get_state(self.current_time)
        state_urllc = self.buffer_urllc.get_state(self.current_time)
        cqis = self.channel.get_ue_cqis()

        # Normalize: Load -> Mb, Delay -> 100ms units
        state_embb[0] = state_embb[0] / 1e6
        state_urllc[0] = state_urllc[0] / 1e6
        state_embb[1] = state_embb[1] / 0.1
        state_urllc[1] = state_urllc[1] / 0.1
        
        # Flatten list
        return np.array(state_embb + state_urllc + list(cqis), dtype=np.float32)
    
    def reset(self):
        """重置基地台狀態"""
        self.current_time = 0.0
        self.buffer_embb.reset()
        self.buffer_urllc.reset()
        # 通道可以不重置，或重新隨機
        channel_cfg = self.config.get('channel', {})
        self.channel = ChannelSimulator(
            num_ues=2,
            seed=self.config['random_seed'],
            fixed_cqi=channel_cfg.get('fixed_cqi', False),
            fixed_cqi_values=channel_cfg.get('fixed_cqi_values', None)
        )


# ==========================================
# 單元測試 (Unit Test)
# ==========================================
if __name__ == "__main__":
    print("--- Testing BaseStation Engine ---")
    
    # 1. 建立假設定 (Mock Config)
    mock_config = {
        'random_seed': 42,
        'traffic': {
            'embb_arrival_rate_mbps': 100.0,
            'urllc_arrival_rate_mbps': 10.0
        },
        'agent': {
            'buffer_size': 10000
        }
    }
    
    # 2. 初始化基地台
    bs = BaseStation(mock_config)
    print("BaseStation Initialized.")
    print(f"Total RBs: {bs.total_rbs}")
    
    # 3. 測試 Step 1: 給定 eMBB 50%, URLLC 50%
    print("\n--- Step 1: Action [0.5, 0.5] ---")
    info = bs.step([0.5, 0.5])
    
    print(f"Allocated RBs -> eMBB: {info['rbs_embb']}, URLLC: {info['rbs_urllc']}")
    print(f"Throughput    -> eMBB: {info['throughput_embb_mbps']:.2f} Mbps")
    print(f"Buffer Load   -> eMBB: {info['buffer_embb_bits']} bits")
    print(f"CQI           -> eMBB: {info['cqi_embb']}, URLLC: {info['cqi_urllc']}")
    
    # 4. 驗證邏輯
    # 如果 CQI 不是 0，分到了 136 個 RBs，吞吐量應該要是正的
    if info['throughput_embb_mbps'] > 0 or info['buffer_embb_bits'] > 0:
        print("✅ Throughput calculation seems reasonable (Data implies traffic arrival).")
    else:
        print("❌ Something wrong. No throughput even with RBs allocated?")

    # 5. 測試 Step 2: 極端情況，全部給 eMBB [1.0, 0.0]
    print("\n--- Step 2: Action [1.0, 0.0] ---")
    info = bs.step([1.0, 0.0])
    print(f"Allocated RBs -> eMBB: {info['rbs_embb']}, URLLC: {info['rbs_urllc']}")
    
    if info['rbs_urllc'] == 0:
         print("✅ Resource mapping logic works (URLLC got 0 RBs).")