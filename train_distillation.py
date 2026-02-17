"""
train_distillation.py

知識蒸餾訓練腳本 (Knowledge Distillation Training Script).
修復版本：分離訓練與評估環境，避免 EvalCallback 中斷訓練流程。
使用方式：
    # 單執行緒測試
    python distillation_train.py

    # 使用 nohup 在背景執行
    nohup python train_distillation.py > distillation_train_202602071720.log 2>&1 &

    # 查看背景執行狀態
    ps aux | grep python | grep distillation_train

    # 停止背景執行
    kill <PID>
"""

import os
import yaml
import time
import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from src.envs.slicing_env import NetworkSlicingEnv
from src.algorithms.distilled_sac import DistilledSAC

def load_config(config_path="configs/distillation_config.yaml"):
    """讀取 YAML 設定檔"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def count_parameters(model):
    """計算模型的總參數量"""
    if not model.policy:
        return 0
    return sum(p.numel() for p in model.policy.parameters() if p.requires_grad)

def main():
    # ==========================================
    # 1. 初始設定 (Setup)
    # ==========================================
    print("--- 1. Loading Configuration for Distillation ---")
    config = load_config()
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{config['experiment_name']}_{timestamp}"
    
    log_dir = os.path.join(config['logging']['log_dir'], exp_name)
    save_dir = os.path.join(config['logging']['save_dir'], exp_name)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Experiment: {exp_name}")
    print(f"Logs: {log_dir}")
    print(f"Models: {save_dir}")

    # ==========================================
    # 2. 建立環境 (Environments) - CRITICAL FIX
    # ==========================================
    print("\n--- 2. Creating Environments (Train & Eval) ---")
    
    # [關鍵修復]：分別建立訓練環境與評估環境
    # 1. 訓練環境 (Training Env)：給 Agent 收集經驗用
    train_env = NetworkSlicingEnv(config)
    train_env = Monitor(train_env, filename=os.path.join(log_dir, "train_monitor"))
    
    # 2. 評估環境 (Evaluation Env)：給 Callback 測試用
    # 必須是獨立的實例，否則 EvalCallback 的 reset() 會打斷訓練
    eval_env = NetworkSlicingEnv(config)
    eval_env = Monitor(eval_env, filename=os.path.join(log_dir, "eval_monitor"))
    
    print("✅ Environments created successfully (Isolated Train & Eval instances).")

    # ==========================================
    # 3. 載入老師模型 (Teacher Model)
    # ==========================================
    print("\n--- 3. Loading Teacher Model ---")
    teacher_path = config['agent']['teacher_path']
    if not os.path.exists(teacher_path):
        # 嘗試在 models/ 根目錄找找看
        if os.path.exists(os.path.join("models", teacher_path)):
             teacher_path = os.path.join("models", teacher_path)
        else:
            raise FileNotFoundError(
                f"❌ Teacher model not found at: {teacher_path}\n"
                "Please train a teacher model first using 'main.py'."
            )
    
    # 載入老師模型
    # 注意：這裡傳入 train_env 主要是為了對齊 Action/Observation Space，
    # 老師模型在蒸餾過程中只用於 predict，不會與環境互動，所以共用 train_env 無妨。
    teacher_model = SAC.load(teacher_path, env=train_env)
    print(f"✅ Teacher Loaded: {teacher_path}")

    # ==========================================
    # 4. 初始化學生模型 (Student Model)
    # ==========================================
    print("\n--- 4. Initializing Student Model (DistilledSAC) ---")
    
    agent_params = config['agent']
    
    student_model = DistilledSAC(
        "MlpPolicy",
        train_env,  # [使用訓練環境]
        teacher_model=teacher_model,
        distillation_alpha=agent_params['distillation_alpha'],
        verbose=1,
        tensorboard_log=log_dir,
        seed=config['random_seed'],
        policy_kwargs=agent_params.get('policy_kwargs', {}),
        learning_rate=agent_params['learning_rate'],
        buffer_size=agent_params['buffer_size'],
        batch_size=agent_params['batch_size'],
        gamma=agent_params['gamma'],
        tau=agent_params['tau'],
        ent_coef=agent_params['ent_coef']
    )

    # ==========================================
    # 5. 模型複雜度比較
    # ==========================================
    print("\n" + "="*50)
    print("📊 Model Complexity Analysis")
    print("="*50)
    
    teacher_params = sum(p.numel() for p in teacher_model.policy.parameters())
    student_params = count_parameters(student_model)
    compression_ratio = teacher_params / student_params if student_params > 0 else 0
    
    print(f"👨‍🏫 Teacher Params: {teacher_params:,}")
    print(f"🧑‍🎓 Student Params: {student_params:,}")
    print(f"📉 Compression Ratio: {compression_ratio:.2f}x")
    print("="*50 + "\n")

    # ==========================================
    # 6. 開始訓練 (Training)
    # ==========================================
    print(f"--- 5. Starting Distillation Training for {config['total_timesteps']} steps ---")
    
    # Eval Callback 使用獨立的 eval_env
    eval_callback = EvalCallback(
        eval_env,  # [使用獨立的評估環境]
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=config['eval_freq'],
        n_eval_episodes=config['n_eval_episodes'],
        deterministic=True
    )

    start_time = time.time()
    
    student_model.learn(
        total_timesteps=config['total_timesteps'],
        callback=eval_callback,
        progress_bar=True
    )
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n--- Training Finished in {duration:.2f} seconds ---")

    # ==========================================
    # 7. 存檔與清理
    # ==========================================
    final_path = os.path.join(save_dir, "final_student_model")
    student_model.save(final_path)
    print(f"💾 Final Student Model saved to: {final_path}.zip")
    
    with open(os.path.join(save_dir, "distillation_config.yaml"), 'w') as f:
        yaml.dump(config, f)

    # 關閉環境，釋放資源
    train_env.close()
    eval_env.close()
    print("Done.")

if __name__ == "__main__":
    main()