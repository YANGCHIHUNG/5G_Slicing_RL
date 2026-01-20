"""
src/core/traffic.py

定義 eMBB/URLLC 流量生成器

流量生成器 (Traffic Generator)。
使用泊松分佈 (Poisson Distribution) 來模擬封包的隨機到達過程。
負責將應用層的速率 (Mbps) 轉換為每個 TTI 的封包到達數量。
"""

import numpy as np
import src.utils.constants as C

class TrafficGenerator:
    def __init__(self, arrival_rate_mbps: float, packet_size_bytes: int, seed: int = None):
        """
        初始化流量生成器。
        
        Args:
            arrival_rate_mbps (float): 平均到達速率 (Mbps)
            packet_size_bytes (int): 每個封包的大小 (Bytes)
            seed (int, optional): 隨機種子
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.arrival_rate_mbps = arrival_rate_mbps
        self.packet_size_bytes = packet_size_bytes
        self.packet_size_bits = packet_size_bytes * 8
        
        # --- 計算 Lambda (每個 TTI 平均到達的封包數) ---
        # 公式推導：
        # 1. 每秒總 bits = Mbps * 10^6
        # 2. 每個 TTI 總 bits = (Mbps * 10^6) * TTI_DURATION
        # 3. 每個 TTI 平均封包數 (Lambda) = 每個 TTI 總 bits / 封包大小(bits)
        
        bits_per_sec = arrival_rate_mbps * 1e6
        bits_per_tti = bits_per_sec * C.TTI_DURATION_SEC
        
        # 這是 Poisson 分佈的參數 Lambda (期望值)
        self.lam = bits_per_tti / self.packet_size_bits
        
        # 用於統計驗證
        self.total_packets_generated = 0
        self.steps_taken = 0

    def step(self):
        """
        推進一個時間步 (TTI)，生成新到達的流量。
        
        Returns:
            tuple: (num_packets, total_bits)
                - num_packets (int): 這次產生了幾個封包
                - total_bits (float): 總共多少 bits (用於加入 Buffer)
        """
        # 使用 Numpy 的泊松分佈生成隨機整數
        num_packets = np.random.poisson(self.lam)
        
        total_bits = num_packets * self.packet_size_bits
        
        # 更新統計數據
        self.total_packets_generated += num_packets
        self.steps_taken += 1
        
        return num_packets, total_bits
    
    def get_statistics(self):
        """回傳目前的統計資訊"""
        if self.steps_taken == 0:
            return 0.0
        return self.total_packets_generated / self.steps_taken

# ==========================================
# 單元測試 (Unit Test)
# ==========================================
if __name__ == "__main__":
    print("--- Testing Traffic Generator ---")
    
    # 測試設定
    TEST_RATE_MBPS = 100.0  # 測試 100 Mbps
    TEST_PKT_SIZE = 1500    # 1500 Bytes (標準 MTU)
    SIM_STEPS = 1000        # 跑 1000 個 TTI
    
    # 初始化
    traffic_gen = TrafficGenerator(TEST_RATE_MBPS, TEST_PKT_SIZE, seed=42)
    
    # 計算理論期望值
    # 100 Mbps = 100,000,000 bits/sec
    # 1 TTI = 0.0005 sec
    # Bits per TTI = 50,000 bits
    # Packet Size = 1500 * 8 = 12,000 bits
    # Expected Lambda = 50,000 / 12,000 = 4.166... packets/TTI
    expected_lambda = traffic_gen.lam
    print(f"Config: {TEST_RATE_MBPS} Mbps, {TEST_PKT_SIZE} Bytes")
    print(f"Theoretical Lambda (Avg Packets/TTI): {expected_lambda:.4f}")
    
    # 開始模擬
    print(f"\nRunning {SIM_STEPS} steps simulation...")
    history = []
    for _ in range(SIM_STEPS):
        n_pkts, _ = traffic_gen.step()
        history.append(n_pkts)
        
    # 統計結果
    actual_mean = np.mean(history)
    total_mbps = (sum(history) * TEST_PKT_SIZE * 8) / (SIM_STEPS * C.TTI_DURATION_SEC) / 1e6
    
    print(f"Actual Mean (Avg Packets/TTI): {actual_mean:.4f}")
    print(f"Calculated Rate from simulation: {total_mbps:.2f} Mbps")
    
    # 驗證誤差 (容許 5% 誤差內)
    error_rate = abs(actual_mean - expected_lambda) / expected_lambda * 100
    print(f"Error Rate: {error_rate:.2f}%")
    
    if error_rate < 5.0:
        print("\n✅ Test Passed: Traffic generation is accurate.")
    else:
        print("\n❌ Test Failed: Deviation is too high.")