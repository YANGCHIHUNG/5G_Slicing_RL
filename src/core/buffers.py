"""
src/core/buffers.py

定義封包佇列與 Dropping 機制

封包緩衝區管理 (Buffer Management)。
實作 FIFO (First-In-First-Out) 佇列，用於儲存等待傳輸的封包。
負責計算關鍵狀態指標：佇列長度 (Queue Length) 與 隊頭延遲 (HoL Delay)。
"""

import collections
import numpy as np
import src.utils.constants as C

class SliceBuffer:
    def __init__(self, max_size_packets: int = C.MAX_BUFFER_SIZE, max_latency_threshold: float = None):
        """
        初始化切片緩衝區。
        
        Args:
            max_size_packets (int): 緩衝區最大容量 (封包數)，防止記憶體溢出
            max_latency_threshold (float): 最大容忍延遲 (秒)，用於 URLLC 超時丟包判定。
                                           若為 None，代表 eMBB (無限容忍)。
        """
        # 使用 deque 實作 FIFO，popleft 效率為 O(1)
        self.queue = collections.deque()
        
        self.max_size = max_size_packets
        self.max_latency_threshold = max_latency_threshold
        
        # 狀態追蹤變數
        self.total_bits = 0  # 目前累積的總位元數 (Buffer Load)
        self.dropped_packets = 0 # 累積丟包數
        self.total_delay_sum = 0.0 # 用於計算平均延遲
        self.total_served_packets = 0

    def add_packet(self, size_bits: int, arrival_time: float):
        """
        加入一個新封包到佇列尾端。
        
        Args:
            size_bits (int): 封包大小 (bits)
            arrival_time (float): 到達時間 (模擬器當前時間)
        
        Returns:
            bool: 是否成功加入 (若 Buffer 滿了則回傳 False)
        """
        if len(self.queue) > 200000:
            self.dropped_packets += 1
            return False

        if len(self.queue) >= self.max_size:
            # Buffer Overflow (溢位丟包)
            self.dropped_packets += 1
            return False
        
        # 封包結構：簡單的 Dictionary
        packet = {
            'size': size_bits,
            'arrival_time': arrival_time
        }
        
        self.queue.append(packet)
        self.total_bits += size_bits
        return True

    def remove_packets(self, capacity_bits: float, current_time: float):
        """
        根據基地台分配的容量，從佇列頭部移除 (傳送) 封包。
        
        Args:
            capacity_bits (float): 這個 TTI 分配到的傳輸容量 (bits)
            current_time (float): 當前時間 (用於計算延遲)
            
        Returns:
            tuple: (transmitted_bits, latencies)
                - transmitted_bits (int): 實際傳出的位元數
                - latencies (list): 每個被傳出封包的延遲時間 (秒)
        """
        transmitted_bits = 0
        latencies = []
        
        # 持續傳送直到容量用完或佇列清空
        # 模擬策略：Whole Packet Transmission (完整封包傳輸)
        # 簡化起見，假設剩餘容量不足以傳一個完整封包時，就停止 (不切分)
        while len(self.queue) > 0:
            packet = self.queue[0] # 查看隊頭 (Peek)
            
            if packet['size'] <= capacity_bits:
                # 容量足夠，傳送此封包
                pkt = self.queue.popleft()
                
                # 更新狀態
                capacity_bits -= pkt['size']
                transmitted_bits += pkt['size']
                self.total_bits -= pkt['size']
                
                # 計算延遲
                delay = current_time - pkt['arrival_time']
                latencies.append(delay)
                
                # 統計
                self.total_served_packets += 1
                self.total_delay_sum += delay
            else:
                # 容量不足以傳送下一個封包，停止
                break
                
        return transmitted_bits, latencies

    def check_timeout(self, current_time: float):
        """
        (URLLC 專用) 檢查並丟棄超時的封包。
        只檢查隊頭 (Head)，因為它是最老的。如果隊頭沒超時，後面的更不會超時。
        
        Args:
            current_time (float): 當前時間
            
        Returns:
            int: 這次丟棄了幾個超時封包
        """
        if self.max_latency_threshold is None:
            return 0 # eMBB 不檢查超時
            
        timeout_count = 0
        
        while len(self.queue) > 0:
            packet = self.queue[0]
            delay = current_time - packet['arrival_time']
            
            if delay > self.max_latency_threshold:
                # 超時了！丟棄！
                pkt = self.queue.popleft()
                self.total_bits -= pkt['size']
                self.dropped_packets += 1
                timeout_count += 1
            else:
                # 隊頭沒超時，代表後面的也都還很新鮮，直接結束檢查
                break
                
        return timeout_count

    def get_state(self, current_time: float):
        """
        獲取 RL Agent 所需的觀測狀態。
        
        Returns:
            list: [Queue_Load (bits), HoL_Delay (sec)]
        """
        load = self.total_bits
        
        if len(self.queue) > 0:
            # 計算 Head-of-Line Delay
            hol_packet = self.queue[0]
            hol_delay = current_time - hol_packet['arrival_time']
        else:
            hol_delay = 0.0
            
        return [load, hol_delay]
    
    def reset(self):
        """重置 Buffer (用於 env.reset)"""
        self.queue.clear()
        self.total_bits = 0
        self.dropped_packets = 0
        self.total_delay_sum = 0
        self.total_served_packets = 0


# ==========================================
# 單元測試 (Unit Test)
# ==========================================
if __name__ == "__main__":
    print("--- Testing SliceBuffer ---")
    
    # 設定：最大容忍 1ms (0.001s)
    urllc_buffer = SliceBuffer(max_size_packets=10, max_latency_threshold=0.001)
    
    # 時間點 T=0: 加入兩個封包 (大小 100 bits)
    urllc_buffer.add_packet(100, arrival_time=0.0)
    urllc_buffer.add_packet(100, arrival_time=0.0)
    
    print(f"Initial State (T=0): {urllc_buffer.get_state(0.0)}")
    
    # 時間點 T=0.0008 (0.8ms): 還沒超時
    # 嘗試傳送 150 bits (只能傳走 1 個，剩下 1 個)
    tx_bits, delays = urllc_buffer.remove_packets(150, current_time=0.0008)
    print(f"\nTime 0.8ms: Transmitted {tx_bits} bits, Delay: {delays}")
    print(f"State after tx: {urllc_buffer.get_state(0.0008)}")
    
    # 時間點 T=0.0012 (1.2ms): 剩下的那個封包應該已經過期了 (0.0012 - 0.0 > 0.001)
    print("\nChecking timeout at 1.2ms...")
    dropped = urllc_buffer.check_timeout(current_time=0.0012)
    print(f"Dropped packets: {dropped}")
    
    # 檢查是否清空
    print(f"Final State: {urllc_buffer.get_state(0.0012)}")
    
    if dropped == 1 and urllc_buffer.total_bits == 0:
        print("\n✅ Test Passed: FIFO and Timeout logic works.")
    else:
        print("\n❌ Test Failed.")