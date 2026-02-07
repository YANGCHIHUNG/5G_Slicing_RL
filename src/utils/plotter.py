"""
src/utils/plotter.py

繪圖工具庫。
負責將實驗數據視覺化，生成符合學術論文標準的圖表。
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def plot_evaluation_results(df: pd.DataFrame, save_dir: str):
    """
    繪製並儲存評估結果圖表。
    
    Args:
        df (pd.DataFrame): 包含模擬過程數據的 DataFrame
        save_dir (str):圖片儲存路徑
    """
    # 設定畫圖風格
    sns.set_theme(style="whitegrid")
    
    # 確保儲存目錄存在
    os.makedirs(save_dir, exist_ok=True)

    # ==========================================
    # 1. 吞吐量變化圖 (Throughput over Time)
    # ==========================================
    plt.figure(figsize=(10, 5))
    # 為了圖表清晰，我們取前 200 個 TTI 或使用滑動平均
    window = 50
    plt.plot(df['time'], df['throughput_embb_mbps'].rolling(window).mean(), label='eMBB (Moving Avg)', color='blue')
    plt.plot(df['time'], df['throughput_urllc_mbps'].rolling(window).mean(), label='URLLC (Moving Avg)', color='red')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Throughput (Mbps)')
    plt.title('Slice Throughput Performance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'throughput_time.png'), dpi=300)
    plt.close()

    # ==========================================
    # 2. RB 資源分配圖 (Resource Allocation)
    # ==========================================
    plt.figure(figsize=(10, 5))
    # 堆疊面積圖
    plt.stackplot(df['time'], df['rbs_embb'], df['rbs_urllc'], labels=['eMBB RBs', 'URLLC RBs'], colors=['#a1c9f4', '#ff9f9b'])
    
    plt.xlabel('Time (s)')
    plt.ylabel('Number of RBs')
    plt.title('Dynamic Resource Block Allocation')
    plt.legend(loc='upper right')
    plt.margins(0, 0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rb_allocation.png'), dpi=300)
    plt.close()

    # ==========================================
    # 3. URLLC 延遲 CDF 圖 (Latency CDF) ⭐️ 論文關鍵
    # ==========================================
    plt.figure(figsize=(8, 6))
    
    # 過濾掉沒有傳輸 (Throughput > 0) 的數據，保留延遲為 0 的有效數據
    valid_latencies = df[df['throughput_urllc_mbps'] > 0]['latency_urllc'] * 1000 # 轉為 ms
    
    if len(valid_latencies) > 0:
        sns.ecdfplot(data=valid_latencies, label='RL Agent')
        
        # 畫一條 1ms 的紅線 (Deadline)
        plt.axvline(x=1.0, color='r', linestyle='--', label='Latency Budget (1ms)')
        
        plt.xlabel('Latency (ms)')
        plt.ylabel('CDF (Probability)')
        plt.title('URLLC Latency Distribution (CDF)')
        plt.legend()
        plt.grid(True, which="both", ls="-")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'latency_cdf.png'), dpi=300)
    else:
        print("Warning: No valid URLLC latency data to plot CDF.")
    
    plt.close()

    print(f"📊 Plots saved to: {save_dir}")