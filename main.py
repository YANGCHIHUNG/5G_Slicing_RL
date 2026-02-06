"""
main.py

5G 網路切片強化學習訓練主程式 (Training Entry Point)。
負責讀取設定、建立環境、初始化 SAC Agent，並執行訓練迴圈。

使用方式：
    # 單執行緒測試
    python -m main

    # 使用 nohup 在背景執行
    nohup python -m main > main_202602051621.log 2>&1 &

    # 查看背景執行狀態
    ps aux | grep python | grep main

    # 停止背景執行
    kill <PID>
"""

import os
import yaml
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from src.envs.slicing_env import NetworkSlicingEnv

def load_config(config_path="configs/best_config_optuna.yaml"):
    """讀取 YAML 設定檔"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


class DetailedLoggingCallback(BaseCallback):
    """
    自定義 Callback：每隔一定步數顯示詳細的訓練狀態
    包括網路切片分配比例、Buffer 狀況、吞吐量、延遲等
    """
    def __init__(self, log_freq=100, verbose=0):
        """
        Args:
            log_freq (int): 每隔多少 timesteps 列印一次
            verbose (int): 詳細程度 (0=簡潔, 1=詳細)
        """
        super(DetailedLoggingCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        """
        每個 step 後被呼叫
        """
        # 累積當前 episode 的獎勵
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # 檢查 episode 是否結束
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # 每隔 log_freq 步驟列印詳細資訊
        if self.num_timesteps % self.log_freq == 0:
            self._print_detailed_info()
            
        return True
    
    def _print_detailed_info(self):
        """列印詳細的訓練狀態"""
        # 從 locals 中獲取最新的資訊
        infos = self.locals.get('infos', [{}])
        if len(infos) > 0:
            info = infos[0]
        else:
            return
        
        # 獲取當前 observation (狀態)
        obs = self.locals.get('new_obs', None)
        if obs is not None and len(obs) > 0:
            obs = obs[0]  # [eMBB_Load, eMBB_HoL, URLLC_Load, URLLC_HoL, CQI_eMBB, CQI_URLLC]
        
        # 獲取當前 action (動作)
        actions = self.locals.get('actions', None)
        if actions is not None and len(actions) > 0:
            action = actions[0]  # [w_embb, w_urllc]
        else:
            action = [0, 0]

        # 若環境有套用約束，優先顯示實際套用的權重
        action_applied = info.get('action_applied', action)
        
        print("\n" + "="*80)
        print(f"📊 Training Status at Step {self.num_timesteps}")
        print("="*80)
        
        # --- 1. Episode 統計 ---
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(self.episode_lengths[-10:]) if len(self.episode_lengths) >= 10 else np.mean(self.episode_lengths)
            print(f"📈 Episodes completed: {len(self.episode_rewards)}")
            print(f"   Avg Reward (last 10): {avg_reward:.2f}")
            print(f"   Avg Length (last 10): {avg_length:.1f}")
        
        # --- 2. 網路切片資源分配 ---
        print(f"\n🔧 Resource Allocation:")
        total_w = action_applied[0] + action_applied[1] + 1e-9
        w_embb_norm = action_applied[0] / total_w
        w_urllc_norm = action_applied[1] / total_w
        print(f"   eMBB Weight:  {action_applied[0]:.4f} (normalized: {w_embb_norm:.1%})")
        print(f"   URLLC Weight: {action_applied[1]:.4f} (normalized: {w_urllc_norm:.1%})")
        
        if 'rbs_embb' in info and 'rbs_urllc' in info:
            total_rbs = info['rbs_embb'] + info['rbs_urllc']
            print(f"   eMBB RBs:  {info['rbs_embb']:3d} / {total_rbs} ({info['rbs_embb']/total_rbs:.1%})")
            print(f"   URLLC RBs: {info['rbs_urllc']:3d} / {total_rbs} ({info['rbs_urllc']/total_rbs:.1%})")
        
        # --- 3. Buffer 狀況 ---
        print(f"\n📦 Buffer Status:")
        if obs is not None:
            embb_load_bits = obs[0]
            embb_hol_delay = obs[1]
            urllc_load_bits = obs[2]
            urllc_hol_delay = obs[3]
            
            print(f"   eMBB Buffer:  {embb_load_bits:,.0f} bits ({embb_load_bits/1e6:.2f} Mb)")
            print(f"   eMBB HoL Delay: {embb_hol_delay*1000:.3f} ms")
            print(f"   URLLC Buffer: {urllc_load_bits:,.0f} bits ({urllc_load_bits/1e6:.2f} Mb)")
            print(f"   URLLC HoL Delay: {urllc_hol_delay*1000:.3f} ms")
        
        # --- 4. 通道品質 (CQI) ---
        print(f"\n📡 Channel Quality (CQI):")
        if obs is not None:
            cqi_embb = obs[4]
            cqi_urllc = obs[5]
            print(f"   eMBB CQI:  {cqi_embb:.1f} / 15")
            print(f"   URLLC CQI: {cqi_urllc:.1f} / 15")
        
        # --- 5. 效能指標 ---
        print(f"\n⚡ Performance Metrics:")
        if 'throughput_embb_mbps' in info:
            print(f"   eMBB Throughput:  {info['throughput_embb_mbps']:.2f} Mbps")
        if 'throughput_urllc_mbps' in info:
            print(f"   URLLC Throughput: {info['throughput_urllc_mbps']:.2f} Mbps")
        if 'latency_urllc' in info:
            print(f"   URLLC Latency: {info['latency_urllc']*1000:.3f} ms")
        if 'dropped_urllc' in info:
            print(f"   URLLC Dropped: {info['dropped_urllc']} packets")
        
        # --- 6. 獎勵細項 ---
        reward = self.locals.get('rewards', [0])[0]
        print(f"\n💰 Reward Breakdown:")
        print(f"   Total Reward: {reward:.4f}")
        if 'reward_throughput' in info:
            print(f"   + Throughput Reward: {info['reward_throughput']:.4f}")
        if 'reward_latency' in info:
            print(f"   - Latency Penalty:   {info['reward_latency']:.4f}")
        if 'reward_drop' in info:
            print(f"   - Drop Penalty:      {info['reward_drop']:.4f}")
        
        print("="*80 + "\n")

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
    # 使用多環境並行 (SubprocVecEnv)
    def make_env(rank: int):
        def _init():
            env = NetworkSlicingEnv(config)
            env.reset(seed=config['random_seed'] + rank)
            return env
        return _init

    num_envs = int(config['agent'].get('num_envs', 1))
    env_fns = [make_env(i) for i in range(num_envs)]
    train_env = SubprocVecEnv(env_fns)
    train_env = VecMonitor(train_env, filename=os.path.join(log_dir, "train_monitor"))
    
    # 建立評估環境 (Evaluation Env)
    # 獨立於訓練環境，用於 EvalCallback 定期測試模型表現 (不含噪聲)
    eval_env = NetworkSlicingEnv(config)
    eval_env = Monitor(eval_env, filename=os.path.join(log_dir, "eval_monitor"))

    # ==========================================
    # 3. 建立回調函數 (Callbacks)
    # ==========================================
    # 評估回調：定期測試模型並儲存最佳模型
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=config['eval_freq'],
        n_eval_episodes=config['n_eval_episodes'],
        deterministic=True, # 測試時不使用隨機探索，評估真實實力
        render=False
    )
    
    # 詳細日誌回調：顯示訓練過程中的詳細資訊
    # log_freq 可以根據需要調整 (每 1000 步顯示一次)
    detailed_logging_callback = DetailedLoggingCallback(
        log_freq=1000,  # 每 100 timesteps 顯示一次詳細資訊
        verbose=1
    )
    
    # 組合多個 callbacks
    callbacks = CallbackList([eval_callback, detailed_logging_callback])

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
        callback=callbacks,  # 使用組合的 callbacks
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