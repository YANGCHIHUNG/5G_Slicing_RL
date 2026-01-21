"""
main.py

5G 網路切片強化學習訓練主程式 (Training Entry Point)。
負責讀取設定、建立環境、初始化 SAC Agent，並執行訓練迴圈。
"""

import os
import yaml
import time
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from src.envs.slicing_env import NetworkSlicingEnv

def load_config(config_path="configs/default_config.yaml"):
    """讀取 YAML 設定檔"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # ==========================================
    # 1. 初始設定 (Setup)
    # ==========================================
    print("--- 1. Loading Configuration ---")
    config = load_config()
    
    # 建立實驗 ID (加上時間戳記，避免覆蓋舊實驗)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{config['experiment_name']}_{timestamp}"
    
    # 設定路徑
    log_dir = os.path.join(config['logging']['log_dir'], exp_name)
    save_dir = os.path.join(config['logging']['save_dir'], exp_name)
    
    # 確保資料夾存在
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Experiment: {exp_name}")
    print(f"Logs: {log_dir}")
    print(f"Models: {save_dir}")

    # ==========================================
    # 2. 建立環境 (Environment)
    # ==========================================
    print("\n--- 2. Creating Environment ---")
    
    # 建立訓練環境 (Training Env)
    # 使用 Monitor 包裝環境，這樣 TensorBoard 才能紀錄 Episode Reward 和 Length
    train_env = NetworkSlicingEnv(config)
    train_env = Monitor(train_env, filename=os.path.join(log_dir, "train_monitor"))
    
    # 建立評估環境 (Evaluation Env)
    # 獨立於訓練環境，用於 EvalCallback 定期測試模型表現 (不含噪聲)
    eval_env = NetworkSlicingEnv(config)
    eval_env = Monitor(eval_env, filename=os.path.join(log_dir, "eval_monitor"))

    # ==========================================
    # 3. 建立回調函數 (Callbacks)
    # ==========================================
    # 這是非常重要的部分：每隔一段時間暫停訓練，用 eval_env 測試目前模型
    # 如果發現是目前為止最好的模型，就自動存檔 (best_model.zip)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=config['eval_freq'],
        n_eval_episodes=config['n_eval_episodes'],
        deterministic=True, # 測試時不使用隨機探索，評估真實實力
        render=False
    )

    # ==========================================
    # 4. 初始化 Agent (SAC)
    # ==========================================
    print("\n--- 3. Initializing SAC Agent ---")
    
    agent_params = config['agent']
    
    model = SAC(
        "MlpPolicy",          # 使用多層感知機 (MLP) 作為神經網路架構
        train_env,
        verbose=1,
        tensorboard_log=log_dir,
        # 從 config 載入超參數
        learning_rate=agent_params['learning_rate'],
        buffer_size=agent_params['buffer_size'],
        batch_size=agent_params['batch_size'],
        gamma=agent_params['gamma'],
        tau=agent_params['tau'],
        ent_coef=agent_params['ent_coef'],
        seed=config['random_seed']
    )
    
    print(model.policy) # 印出網路架構確認

    # ==========================================
    # 5. 開始訓練 (Training)
    # ==========================================
    print(f"\n--- 4. Starting Training for {config['total_timesteps']} steps ---")
    
    start_time = time.time()
    
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=eval_callback,
        progress_bar=True # 顯示進度條
    )
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n--- Training Finished in {duration:.2f} seconds ---")

    # ==========================================
    # 6. 存檔 (Saving)
    # ==========================================
    # 儲存最終模型 (不一定是最好的，但包含了最後的訓練狀態)
    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    print(f"Final model saved to: {final_path}.zip")
    
    # 另外儲存一份 config 備份，方便未來查閱當時是用什麼參數跑的
    with open(os.path.join(save_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)
        
    print("Done.")

if __name__ == "__main__":
    main()