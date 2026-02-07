"""
evaluate.py

模型評估與測試腳本。
1. 載入訓練好的最佳模型 (best_model.zip)。
2. 在環境中執行推論 (Inference)。
3. 收集數據並繪製圖表 (Results)。
"""

import os
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from src.envs.slicing_env import NetworkSlicingEnv
from src.utils.plotter import plot_evaluation_results

# ==========================================
# 設定區 (Configuration)
# ==========================================
# 請將此路徑改為您實際訓練出來的實驗資料夾名稱
# 例如: "logs/sac_slicing_baseline_v1_20260121-120000"
EXPERIMENT_DIR = "logs/sac_slicing_baseline_v1_20260206-154124" 

# 參數設定
CONFIG_PATH = "configs/best_config_optuna.yaml"
MODEL_FILENAME = "best_model.zip" # 優先讀取最佳模型
# MODEL_FILENAME = "final_model.zip" # 若沒有 best_model 則讀這個

# 測試長度 (Steps)
EVAL_STEPS = 2000 # 測試 1 秒鐘 (2000 * 0.5ms)


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def evaluate():
    print(f"--- Starting Evaluation ---")
    
    # 1. 檢查路徑
    # 如果使用者還沒改 EXPERIMENT_DIR，嘗試自動尋找最新的 log
    global EXPERIMENT_DIR
    if "YOUR_TIMESTAMP_HERE" in EXPERIMENT_DIR:
        if os.path.exists("models"):
            # 找 models 資料夾裡最新的資料夾
            dirs = [os.path.join("models", d) for d in os.listdir("models") if os.path.isdir(os.path.join("models", d))]
            if dirs:
                EXPERIMENT_DIR = max(dirs, key=os.path.getmtime)
                print(f"⚠️ Auto-detected latest experiment: {EXPERIMENT_DIR}")
            else:
                print("❌ No models found. Please train first!")
                return
        else:
            print("❌ 'models' directory not found.")
            return

    model_path = os.path.join(EXPERIMENT_DIR, MODEL_FILENAME)
    if not os.path.exists(model_path):
        # 嘗試去 models/ 資料夾找 (因為 main.py 存檔邏輯可能分開)
        # 這裡做一個容錯處理
        alt_path = model_path.replace("logs", "models")
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f"❌ Model not found at: {model_path}")
            return

    print(f"📂 Loading Model: {model_path}")

    # 2. 建立環境
    config = load_config(CONFIG_PATH)
    env = NetworkSlicingEnv(config)
    
    # 3. 載入模型
    model = SAC.load(model_path, env=env)
    
    # 4. 執行模擬迴圈
    print(f"▶️ Running simulation for {EVAL_STEPS} steps...")
    
    obs, _ = env.reset()
    history = []
    
    for step in range(EVAL_STEPS):
        # deterministic=True 代表關閉隨機探索 (Exploration)，只使用學到的最佳策略
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 收集數據
        history.append(info)
        
        if terminated or truncated:
            obs, _ = env.reset()

    # 5. 數據處理
    df = pd.DataFrame(history)
    
    # 計算關鍵指標
    avg_embb = df['throughput_embb_mbps'].mean()
    avg_urllc = df['throughput_urllc_mbps'].mean()
    violation_rate = (df['dropped_urllc'] > 0).mean() * 100
    
    print("\n--- 📊 Evaluation Summary ---")
    print(f"Avg eMBB Throughput : {avg_embb:.2f} Mbps")
    print(f"Avg URLLC Throughput: {avg_urllc:.2f} Mbps")
    print(f"URLLC Violation Rate: {violation_rate:.2f}%") # 目標應該要是 0%
    
    # 6. 繪圖
    results_dir = os.path.join("results", os.path.basename(EXPERIMENT_DIR))
    plot_evaluation_results(df, results_dir)
    
    # 儲存原始數據 CSV 以便後續分析
    csv_path = os.path.join(results_dir, "eval_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"💾 Data saved to: {csv_path}")

if __name__ == "__main__":
    evaluate()