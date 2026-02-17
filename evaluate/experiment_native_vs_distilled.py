"""
evaluate_native_vs_distilled.py

模型對決評估腳本 (Model Showdown Evaluation).
直接比較兩個已訓練好的小模型：
1. Native Student: 透過 main.py 訓練的純 SAC 小模型。
2. Distilled Student: 透過 train_distillation.py 訓練的蒸餾小模型。

目的：驗證在「參數量相同」的情況下，經過蒸餾的模型是否在 QoS (延遲/違規率) 上優於原生模型。
"""

import os
import time
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from src.envs.slicing_env import NetworkSlicingEnv

# ==========================================
# 1. 參數與路徑設定 (請修改這裡)
# ==========================================
CONFIG_PATH = "configs/distillation_config.yaml"  # 確保環境設定一致

# [Native Student] 路徑 (您用 main.py 訓練出來的小模型)
# 請修改為實際路徑，例如 "logs/sac_native_small_v1/best_model.zip"
NATIVE_MODEL_PATH = "logs/small_model_sac_slicing_baseline_v1_20260210-214644/best_model.zip" 

# [Distilled Student] 路徑 (您用 train_distillation.py 訓練出來的模型)
# 請修改為實際路徑
DISTILLED_MODEL_PATH = "models/distilled_sac_student_v1_20260207-172248/best_model.zip"

# 評估步數 (2000 steps = 1秒模擬時間)
EVAL_STEPS = 5000 

# ==========================================
# 2. 工具函式
# ==========================================
def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_evaluation(model, env, steps, name):
    """執行單一模型的評估迴圈"""
    print(f"▶️ Running simulation for {name} ({steps} steps)...")
    
    obs, _ = env.reset()
    history = []
    
    # 預熱
    _ = model.predict(obs, deterministic=True)
    
    start_time = time.time()
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        history.append(info)
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    duration = time.time() - start_time
    print(f"   Done in {duration:.2f}s.")
    
    return pd.DataFrame(history)

def calculate_metrics(df):
    """計算關鍵指標"""
    # 吞吐量
    avg_embb = df['throughput_embb_mbps'].mean()
    avg_urllc = df['throughput_urllc_mbps'].mean()
    
    # 延遲 (只計算有數據的 TTI)
    latencies = df[df['latency_urllc'] > 0]['latency_urllc'] * 1000 # ms
    if len(latencies) > 0:
        avg_lat = latencies.mean()
        p99_lat = np.percentile(latencies, 99)
        max_lat = latencies.max()
    else:
        avg_lat = p99_lat = max_lat = 0.0
        
    # 違規率
    violation_rate = (df['dropped_urllc'] > 0).mean() * 100
    
    return {
        "eMBB (Mbps)": avg_embb,
        "URLLC (Mbps)": avg_urllc,
        "Avg Latency (ms)": avg_lat,
        "P99 Latency (ms)": p99_lat,
        "Max Latency (ms)": max_lat,
        "Violation Rate (%)": violation_rate
    }

# ==========================================
# 3. 主程式
# ==========================================
def main():
    print("="*60)
    print("🥊 Native vs. Distilled Student: The Showdown")
    print("="*60)
    
    # 檢查檔案
    if not os.path.exists(NATIVE_MODEL_PATH):
        print(f"❌ Native model not found: {NATIVE_MODEL_PATH}")
        print("   Tip: Update 'NATIVE_MODEL_PATH' in the script.")
        return
    if not os.path.exists(DISTILLED_MODEL_PATH):
        print(f"❌ Distilled model not found: {DISTILLED_MODEL_PATH}")
        print("   Tip: Update 'DISTILLED_MODEL_PATH' in the script.")
        return

    # 載入環境
    print(f"Loading Config: {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)
    
    # 確保流量負載足夠大，才能看出差異 (建議 eMBB > 350)
    print(f"Traffic Settings: eMBB={config['traffic']['embb_arrival_rate_mbps']} Mbps, URLLC={config['traffic']['urllc_arrival_rate_mbps']} Mbps")
    
    env = NetworkSlicingEnv(config)
    
    # 載入模型
    print(f"\nLoading Native Student: {NATIVE_MODEL_PATH}")
    native_model = SAC.load(NATIVE_MODEL_PATH, env=env)
    
    print(f"Loading Distilled Student: {DISTILLED_MODEL_PATH}")
    distilled_model = SAC.load(DISTILLED_MODEL_PATH, env=env)
    
    # 執行評估
    print("\n" + "-"*30)
    df_native = run_evaluation(native_model, env, EVAL_STEPS, "Native Student")
    
    print("-" * 30)
    # 重置環境種子以確保公平 (如果 env 支援 seed)
    # env.reset(seed=config['random_seed']) 
    df_distilled = run_evaluation(distilled_model, env, EVAL_STEPS, "Distilled Student")
    
    # 計算指標
    metrics_native = calculate_metrics(df_native)
    metrics_distilled = calculate_metrics(df_distilled)
    
    # ==========================================
    # 4. 輸出比較報表
    # ==========================================
    print("\n" + "="*60)
    print("📊 Final Comparison Results")
    print("="*60)
    
    # 建立比較 DataFrame
    comp_df = pd.DataFrame([metrics_native, metrics_distilled], index=["Native", "Distilled"])
    
    # 格式化輸出
    print(comp_df.round(4).to_string())
    
    print("\n" + "="*60)
    print("🏆 Verdict (Analysis):")
    
    # 自動判讀
    native_vio = metrics_native['Violation Rate (%)']
    distilled_vio = metrics_distilled['Violation Rate (%)']
    native_p99 = metrics_native['P99 Latency (ms)']
    distilled_p99 = metrics_distilled['P99 Latency (ms)']
    
    if distilled_vio < native_vio:
        print(f"✅ Distilled model has LOWER Violation Rate ({distilled_vio:.2f}% vs {native_vio:.2f}%).")
        print("   -> Knowledge Distillation improved reliability!")
    elif distilled_vio == native_vio:
        if distilled_p99 < native_p99:
            print(f"✅ Distilled model has LOWER P99 Latency ({distilled_p99:.3f}ms vs {native_p99:.3f}ms).")
            print("   -> Knowledge Distillation improved tail latency!")
        else:
             print(f"⚖️ Performance is similar. (Diff: P99 {distilled_p99 - native_p99:.3f}ms)")
    else:
        print(f"❌ Native model performed better. ({native_vio:.2f}% vs {distilled_vio:.2f}%)")
        print("   -> Check if the Teacher model was actually good, or if 'distillation_alpha' needs tuning.")

    # 存檔
    results_dir = "results/native_vs_distilled"
    os.makedirs(results_dir, exist_ok=True)
    comp_df.to_csv(os.path.join(results_dir, "comparison_report.csv"))
    print(f"\n💾 Report saved to: {results_dir}/comparison_report.csv")

if __name__ == "__main__":
    main()