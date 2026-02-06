"""
src/core/channel.py

定義 CQI, MCS, SINR 計算

5G 無線通道模擬模組 (Physical Layer Channel Model)。
負責處理 CQI (Channel Quality Indicator) 的狀態演變，
以及將 CQI 映射至 MCS (Modulation and Coding Scheme) 的頻譜效率。
"""

import numpy as np
import src.utils.constants as C

def get_efficiency(cqi: int) -> float:
    """
    根據 CQI 查表獲取頻譜效率 (Spectral Efficiency)。
    
    Args:
        cqi (int): 通道品質指示 (0-15)
        
    Returns:
        float: 頻譜效率 (bits/symbol/RE)
    """
    # 確保 CQI 在合法範圍內 (0 ~ 15)
    # 物理上 CQI 0 代表 Out of Service
    safe_cqi = int(np.clip(cqi, C.MIN_CQI, C.MAX_CQI))
    
    # 從 constants.py 的表格中讀取效率
    return C.CQI_TO_EFFICIENCY.get(safe_cqi, 0.0)


class ChannelSimulator:
    """
    模擬無線通道的時變特性 (Time-Varying Channel)。
    使用限制隨機漫步 (Bounded Random Walk) 來模擬 CQI 的動態變化。
    """
    
    def __init__(self, num_ues: int, seed: int = None, fixed_cqi: bool = False, fixed_cqi_values=None):
        """
        初始化通道模擬器。
        
        Args:
            num_ues (int): 要模擬的用戶數量
            seed (int): 隨機種子
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.num_ues = num_ues
        self.fixed_cqi = fixed_cqi

        if self.fixed_cqi:
            if fixed_cqi_values is None:
                fixed_cqi_values = [10] * num_ues
            fixed_arr = np.array(fixed_cqi_values, dtype=int)
            if fixed_arr.size == 1:
                fixed_arr = np.full(num_ues, int(fixed_arr.item()))
            self.current_cqis = np.clip(fixed_arr, 1, C.MAX_CQI)
        else:
            # 初始化所有用戶的 CQI
            # 預設讓用戶分佈在收訊較好的區間 (7~15)，避免一開始就斷線
            self.current_cqis = np.random.randint(7, C.MAX_CQI + 1, size=num_ues)
        
        # 定義 CQI 變化的轉移機率 (Transition Probabilities)
        # 模擬慢衰落 (Slow Fading)：大部分時間 CQI 保持不變
        self.prob_stay = 0.90  # 90% 機率不變
        self.prob_change = 0.10 # 10% 機率發生變化 (變好或變壞)

    def step(self) -> np.ndarray:
        """
        推進一個時間步 (TTI)，更新所有用戶的 CQI。
        
        Returns:
            np.ndarray: 更新後的 CQI 陣列
        """
        if self.fixed_cqi:
            return self.current_cqis

        # 為每個 UE 生成一個隨機動作：-1 (變差), 0 (不變), +1 (變好)
        # 邏輯：先決定「是否改變」，再決定「變好還是變壞」
        
        # 1. 決定哪些用戶的 CQI 會改變 (mask)
        change_mask = np.random.random(self.num_ues) < self.prob_change
        
        # 2. 對於需要改變的用戶，隨機決定 +1 或 -1
        # 0.5 機率 +1, 0.5 機率 -1
        directions = np.random.choice([-1, 1], size=self.num_ues)
        
        # 3. 更新 CQI
        # 只有在 change_mask 為 True 的地方才加上 direction
        self.current_cqis += (change_mask * directions).astype(int)
        
        # 4. 邊界限制 (Clipping)
        # 確保 CQI 不會超過 15 或低於 1 (假設最低維持在 1 避免完全斷線邏輯太複雜)
        self.current_cqis = np.clip(self.current_cqis, 1, C.MAX_CQI)
        
        return self.current_cqis

    def get_ue_cqis(self) -> np.ndarray:
        """獲取當前所有用戶的 CQI"""
        return self.current_cqis

    def get_ue_efficiencies(self) -> np.ndarray:
        """
        獲取當前所有用戶的頻譜效率。
        
        Returns:
            np.ndarray: float array of efficiencies
        """
        # 使用 numpy vectorization 快速查表
        # 這裡利用 Python list comprehension 搭配 helper function
        return np.array([get_efficiency(cqi) for cqi in self.current_cqis])


# ==========================================
# 單元測試 (Unit Test)
# ==========================================
if __name__ == "__main__":
    print("--- Testing Channel Module ---")
    
    # 測試 1: 查表功能
    print(f"CQI 15 Efficiency: {get_efficiency(15)} (Expected: ~5.55)")
    print(f"CQI 7  Efficiency: {get_efficiency(7)}  (Expected: ~1.47)")
    
    # 測試 2: 通道波動
    sim = ChannelSimulator(num_ues=5, seed=42)
    print("\nInitial CQIs:", sim.get_ue_cqis())
    
    print("Simulating 10 TTIs...")
    for t in range(10):
        cqis = sim.step()
        effs = sim.get_ue_efficiencies()
        print(f"TTI {t+1}: CQIs {cqis} | Effs {[round(e, 2) for e in effs]}")