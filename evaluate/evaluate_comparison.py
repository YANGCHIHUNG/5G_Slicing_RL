"""
evaluate_comparison.py

教師模型與學生模型全方位評估與比較腳本。
Comprehensive Evaluation Script for Teacher vs. Student Models.

功能：
1. 靜態分析：比較參數量 (Model Size) 與 壓縮率 (Compression Ratio)。
2. 動態效能：分別執行 Teacher 與 Student，比較吞吐量 (Throughput) 與 QoS 指標。
3. 蒸餾品質：在學生執行期間，計算與老師的動作差異 (Fidelity/MSE)。
4. 運算效率：測量推論延遲 (Inference Latency) 與 加速倍率 (Speedup)。
"""

import os
import time
import yaml
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import SAC
from src.envs.slicing_env import NetworkSlicingEnv
from src.utils.plotter import plot_evaluation_results

# ==========================================
# 參數設定區 (Configuration)
# ==========================================
# 設定檔路徑 (請確保與訓練時一致)
CONFIG_PATH = "configs/default_config.yaml"

# 模型路徑 (請修改為您實際的 .zip 檔案路徑)
# 範例： "logs/sac_teacher_v1/best_model.zip"
TEACHER_MODEL_PATH = "logs/(Light Load)sac_slicing_baseline_v1_20260206-154124/best_model.zip" 

# 範例： "logs/distilled_student_v1/best_model.zip"
STUDENT_MODEL_PATH = "models/distilled_sac_student_v1_20260207-172248/best_model.zip"

# 評估步數 (2000 steps = 1秒鐘模擬時間)
EVAL_STEPS = 2000

# ==========================================
# 工具函式
# ==========================================

def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def count_parameters(model):
    """計算模型策略網路 (Policy Network) 的可訓練參數量"""
    if not model.policy:
        return 0
    return sum(p.numel() for p in model.policy.parameters() if p.requires_grad)

def benchmark_pure_inference(model, steps=10000):
    """
    極限推論測試：排除所有數據轉換，只測量神經網路 forward 時間
    """
    import torch
    import time
    
    # 建立一個固定的 Dummy Input (Batch Size = 1, Obs Dim = 6)
    # 假設 observation space 維度是 6，請根據您的環境調整
    obs_dim = model.observation_space.shape[0]
    dummy_input = torch.randn(1, obs_dim, device=model.device)
    
    # 預熱 (Warmup)
    for _ in range(100):
        with torch.no_grad():
            model.policy.forward(dummy_input)
            
    # 開始計時
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            model.policy.forward(dummy_input)
    end = time.perf_counter()
    
    avg_time_ms = ((end - start) / steps) * 1000
    return avg_time_ms

def run_simulation(model, env, steps, name="Model", teacher_model=None):
    print(f"▶️ Running simulation for {name} ({steps} steps)...")
    
    obs, _ = env.reset()
    history = []
    inference_times = []
    action_diffs = []
    
    # [新增] 預先準備 PyTorch Tensor 格式的 observation，避免測量到 numpy->tensor 的轉換時間
    # 注意：這只是為了測量純推論速度的極限值
    dummy_obs_tensor = torch.as_tensor(obs, device=model.device).unsqueeze(0)

    # 預熱 (Warm-up)
    _ = model.policy.forward(dummy_obs_tensor, deterministic=True)

    for _ in range(steps):
        # ============================================
        # [修正] 測量 Raw PyTorch 推論時間 (繞過 SB3 overhead)
        # ============================================
        obs_tensor = torch.as_tensor(obs, device=model.device).unsqueeze(0)
        
        start_t = time.perf_counter()
        with torch.no_grad():
            # 直接呼叫 Policy 網路進行推論
            action_tensor = model.policy.forward(obs_tensor, deterministic=True)
        end_t = time.perf_counter()
        
        # 轉換回 numpy 給環境使用 (這段不計入推論時間，因為實務上是在 C++ 端處理)
        action = action_tensor.cpu().numpy()[0]
        
        inference_times.append((end_t - start_t) * 1000.0) # 轉 ms
        # ============================================

        # 2. (選填) 影子模式：測量老師動作以計算差異
        if teacher_model:
            # 老師也用同樣方式取得動作以求公平，計算 MSE
            action_teacher, _ = teacher_model.predict(obs, deterministic=True)
            diff = np.mean((action - action_teacher)**2)
            action_diffs.append(diff)

        # 3. 環境互動
        obs, reward, terminated, truncated, info = env.step(action)
        history.append(info)

        if terminated or truncated:
            obs, _ = env.reset()
            
    df = pd.DataFrame(history)
    avg_inference_time = np.mean(inference_times)
    fidelity_score = np.mean(action_diffs) if action_diffs else None
    
    return df, avg_inference_time, fidelity_score

def print_metrics(df, label):
    """計算並列印關鍵 QoS 指標"""
    avg_embb = df['throughput_embb_mbps'].mean()
    avg_urllc = df['throughput_urllc_mbps'].mean()
    
    # 計算延遲統計 (過濾掉無數據的 0 值)
    latencies = df[df['latency_urllc'] > 0]['latency_urllc'] * 1000 # 轉 ms
    if len(latencies) > 0:
        avg_lat = latencies.mean()
        p99_lat = np.percentile(latencies, 99)
        max_lat = latencies.max()
    else:
        avg_lat = p99_lat = max_lat = 0.0

    violation_rate = (df['dropped_urllc'] > 0).mean() * 100
    
    print(f"--- {label} Metrics ---")
    print(f"   Throughput (Mbps) : eMBB {avg_embb:6.2f} | URLLC {avg_urllc:6.2f}")
    print(f"   URLLC Latency (ms): Avg  {avg_lat:6.3f} | P99   {p99_lat:6.3f} | Max {max_lat:6.3f}")
    print(f"   Violation Rate    : {violation_rate:6.2f}%")
    return avg_lat, violation_rate

# ==========================================
# 主程式
# ==========================================
def evaluate_comparison():
    print("="*60)
    print("🔬 Teacher-Student Model Comprehensive Evaluation")
    print("="*60)

    # 1. 檢查檔案
    if not os.path.exists(TEACHER_MODEL_PATH):
        print(f"❌ Teacher model not found: {TEACHER_MODEL_PATH}")
        return
    if not os.path.exists(STUDENT_MODEL_PATH):
        print(f"❌ Student model not found: {STUDENT_MODEL_PATH}")
        return

    # 2. 載入環境與模型
    print("\n[1/4] Loading Environment and Models...")
    config = load_config(CONFIG_PATH)
    env = NetworkSlicingEnv(config) # 使用同一份 Config 確保公平

    print(f"   Loading Teacher: {TEACHER_MODEL_PATH}")
    teacher_model = SAC.load(TEACHER_MODEL_PATH, env=env)
    
    print(f"   Loading Student: {STUDENT_MODEL_PATH}")
    student_model = SAC.load(STUDENT_MODEL_PATH, env=env)

    # 3. 靜態分析：模型大小
    print("\n[2/4] Static Analysis: Model Complexity")
    n_params_t = count_parameters(teacher_model)
    n_params_s = count_parameters(student_model)
    compression_ratio = n_params_t / n_params_s if n_params_s > 0 else 0
    
    print(f"   Teacher Params    : {n_params_t:,}")
    print(f"   Student Params    : {n_params_s:,}")
    print(f"   🚀 Compression Ratio: {compression_ratio:.2f}x smaller")

    print("\n[2.5/4] Benchmarking Pure Inference Speed (CPU/GPU Raw Performance)...")
    raw_time_t = benchmark_pure_inference(teacher_model)
    raw_time_s = benchmark_pure_inference(student_model)
    
    raw_speedup = raw_time_t / raw_time_s
    
    print(f"   Teacher Pure Compute: {raw_time_t:.5f} ms")
    print(f"   Student Pure Compute: {raw_time_s:.5f} ms")
    print(f"   ⚡ True Speedup      : {raw_speedup:.2f}x")

    # 4. Phase 1: 執行教師模型 (建立基準)
    print("\n[3/4] Phase 1: Evaluating Teacher Baseline...")
    df_t, time_t, _ = run_simulation(teacher_model, env, EVAL_STEPS, name="Teacher")
    _, vio_t = print_metrics(df_t, "Teacher")

    # 5. Phase 2: 執行學生模型 (並計算差異)
    print("\n[4/4] Phase 2: Evaluating Student Distillation...")
    # 這裡傳入 teacher_model 是為了計算 "Fidelity" (動作差異)，但環境是由學生控制
    df_s, time_s, mse_val = run_simulation(student_model, env, EVAL_STEPS, name="Student", teacher_model=teacher_model)
    _, vio_s = print_metrics(df_s, "Student")

    # ==========================================
    # 6. 最終評估報告
    # ==========================================
    speedup = time_t / time_s if time_s > 0 else 0
    
    print("\n" + "="*60)
    print("📊 FINAL COMPARISON REPORT")
    print("="*60)
    
    print(f"1. Efficiency (Speed & Size)")
    print(f"   - Model Size      : Reduced by {compression_ratio:.1f}x")
    print(f"   - Inference Time  : Teacher {time_t:.3f} ms vs Student {time_s:.3f} ms")
    print(f"   - Speedup         : {speedup:.2f}x faster ⚡")
    print(f"   - Real-time Check : {'✅ PASS (<0.5ms)' if time_s < 0.5 else '❌ FAIL (>0.5ms)'}")

    print(f"\n2. Fidelity (Imitation Quality)")
    print(f"   - Action MSE      : {mse_val:.6f} (Lower is better)")
    
    print(f"\n3. Performance Qualification (QoS)")
    # 簡單的合格判定邏輯
    is_qualified = True
    reasons = []
    
    if vio_s > 1.0: 
        is_qualified = False
        reasons.append(f"Violation Rate too high ({vio_s:.2f}%)")
    
    if time_s > 0.5:
        is_qualified = False
        reasons.append(f"Inference too slow ({time_s:.3f}ms)")
        
    print(f"   - Teacher Violation: {vio_t:.2f}%")
    print(f"   - Student Violation: {vio_s:.2f}%")
    print(f"   - Status           : {'✅ QUALIFIED' if is_qualified else '❌ FAILED'}")
    
    if not is_qualified:
        print(f"   - Issues           : {', '.join(reasons)}")

    print("="*60)
    
    # 儲存結果
    results_dir = "results/comparison"
    os.makedirs(results_dir, exist_ok=True)
    df_t.to_csv(os.path.join(results_dir, "eval_teacher.csv"), index=False)
    df_s.to_csv(os.path.join(results_dir, "eval_student.csv"), index=False)
    print(f"\n💾 Detailed logs saved to: {results_dir}")

if __name__ == "__main__":
    evaluate_comparison()