"""
src/utils/constants.py

此檔案定義 5G 網路切片模擬器的物理層常數與標準規範。
數值參考來源：3GPP TS 38.104, TS 38.211, TS 38.214

注意：此檔案內容為全域常數，原則上不隨實驗變更。
若需調整實驗變數 (如流量到達率)，請修改 configs/default_config.yaml。
"""

# ==========================================
# 1. 物理層與系統架構 (Physics & System)
# ==========================================

# 系統頻寬 (Bandwidth)
SYSTEM_BANDWIDTH_MHZ = 100

# 子載波間隔 (Subcarrier Spacing, SCS) - Numerology 1
SCS_KHZ = 30 

# 每個 RB 的子載波數量 (Subcarriers per RB) - 固定物理定義
SUBCARRIERS_PER_RB = 12

# 每個 Slot 的 OFDM 符號數量 (Symbols per Slot) - 固定物理定義
SYMBOLS_PER_SLOT = 14

# 傳輸時間間隔 (Transmission Time Interval, TTI)
# 對於 SCS 30kHz, 1 Slot = 0.5 ms
TTI_DURATION_MS = 0.5
TTI_DURATION_SEC = 0.0005

# 總資源區塊數量 (Total Resource Blocks)
# 參考 3GPP TS 38.101-1 Table 5.3.2-1
# 在 100 MHz 頻寬與 30 kHz SCS 下，扣除保護頻帶後的標準值
TOTAL_RBS = 273

# 每個 RB 的頻寬 (kHz) = 12 * 30 = 360 kHz
RB_BANDWIDTH_KHZ = SUBCARRIERS_PER_RB * SCS_KHZ

# 每個 RB 在一個 TTI 內的總資源元素 (Resource Elements, REs)
# 注意：這包含導頻 (DMRS) 等 Overhead，計算 Throughput 時通常需打折 (Overhead Factor)
RES_PER_RB = SUBCARRIERS_PER_RB * SYMBOLS_PER_SLOT  # 12 * 14 = 168 REs


# ==========================================
# 2. MCS 與 通道品質 (MCS & Channel)
# ==========================================

# 通道品質指示 (CQI) 範圍
MIN_CQI = 1
MAX_CQI = 15

# 控制通道與導頻開銷係數 (Overhead Factor)
# 假設約 14% 的資源用於 PDCCH, DMRS, CSI-RS 等，剩下 86% 傳數據
CONTROL_OVERHEAD_FACTOR = 0.86

# MCS 頻譜效率對照表 (Spectral Efficiency Map)
# 參考 3GPP TS 38.214 Table 5.1.3.1-1 (256QAM Table)
# Key: CQI Index, Value: Efficiency (bits/symbol)
# 這裡簡化為直接對應 CQI，省略了 MCS Index 的中間轉換
CQI_TO_EFFICIENCY = {
    0: 0.0,    # Out of range / Disconnected
    1: 0.1523, # QPSK
    2: 0.2344,
    3: 0.3770,
    4: 0.6016,
    5: 0.8770,
    6: 1.1758, # 16QAM
    7: 1.4766,
    8: 1.9141,
    9: 2.4063,
    10: 2.7305, # 64QAM
    11: 3.3223,
    12: 3.9023,
    13: 4.5234, # 256QAM
    14: 5.1152,
    15: 5.5547  # 理論最大值可達 7.4，但考慮真實編碼率通常設定在此範圍
}

# ==========================================
# 3. 流量與封包特徵 (Traffic & Packet)
# ==========================================

# 封包大小定義 (Bytes)
# eMBB 模擬影片串流封包 (約 MTU 大小)
PACKET_SIZE_EMBB_BYTES = 1500 

# URLLC 模擬控制訊號或感測器數據 (小封包)
PACKET_SIZE_URLLC_BYTES = 32

# QoS 需求 (QoS Requirements)
# URLLC 最大容忍延遲 (毫秒)
URLLC_MAX_LATENCY_MS = 1.0

# eMBB 最小速率需求 (Mbps) - 用於計算 Reward 懲罰 (選用)
EMBB_MIN_RATE_MBPS = 10.0


# ==========================================
# 4. 強化學習環境設定 (RL Environment)
# ==========================================

# Buffer 最大容量 (避免記憶體溢出)，單位：封包數
MAX_BUFFER_SIZE = 10000

# Reward 權重係數 (這是經驗值，可在此微調或移至 Config)
# Reward = alpha * Throughput - beta * Latency_Penalty
REWARD_SCALE_THROUGHPUT = 0.001  # 將 Mbps 縮放到 0~1 之間
REWARD_SCALE_LATENCY = 10.0      # 對 URLLC 超時給予重罰