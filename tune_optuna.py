"""
tune_optuna.py

使用 Optuna 進行超參數自動化調優 (Hyperparameter Optimization)。

功能：
1. 定義多維搜索空間 (包含 SAC 超參數、獎勵權重、流量負載)
2. 使用 TPE Sampler 智能採樣 + Median Pruner 自動剪枝
3. 支援多執行緒平行化 (n_jobs)
4. 自動儲存最佳配置 + 視覺化結果

使用方式：
    # 單執行緒測試
    python tune_optuna.py --n-trials 20
    
    # 平行化執行 (4核心)
    python tune_optuna.py --n-trials 100 --n-jobs 4
    
    # 查看即時 Dashboard
    optuna-dashboard sqlite:///optuna_study.db

    # 使用 nohup 在背景執行
    nohup python tune_optuna.py --n-trials 50 --n-jobs 16 > tune_optuna_202602101730.log 2>&1 &

    # 查看背景執行狀態
    ps aux | grep tune_optuna.py

    # 停止背景執行
    pkill -f tune_optuna.py
"""

import copy
import shutil
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import os
import argparse
import numpy as np
import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from src.envs.slicing_env import NetworkSlicingEnv

def prepare_output_paths(args: argparse.Namespace):
    """
    確保輸出資料夾與檔案為乾淨狀態：
    - 資料夾若已存在則刪除後重建
    - 檔案若已存在則刪除
    """

    # 需要建立/重建的資料夾
    output_dirs = [
        args.optuna_log_dir,
        args.optuna_model_dir,
    ]

    for dir_path in output_dirs:
        if os.path.exists(dir_path):
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
            else:
                os.remove(dir_path)
        os.makedirs(dir_path, exist_ok=True)

    # 可能會產生的輸出檔案（先清除舊檔）
    output_files = [
        "optuna_history.html",
        "optuna_importance.html",
        "optuna_parallel_coordinate.html",
        "optuna_slice.html",
        "configs/best_config_optuna.yaml",
    ]

    for file_path in output_files:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            os.remove(file_path)

    # Optuna SQLite 資料庫若存在則刪除
    if isinstance(args.storage, str) and args.storage.startswith("sqlite:///"):
        db_path = args.storage.replace("sqlite:///", "", 1)
        if db_path and os.path.exists(db_path) and os.path.isfile(db_path):
            os.remove(db_path)

def objective(trial: optuna.Trial, base_config: dict, args: argparse.Namespace):
    """
    Optuna 目標函數：只調整模型超參數，環境與獎勵權重固定讀取自 base_config
    """
    
    # ==========================================
    # 1. 定義搜索空間 (Search Space) - 僅包含 Agent 大腦結構
    # ==========================================
    
    # --- SAC 核心超參數 ---
    # 稍微縮小 LR 範圍，避免過小導致不收斂
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True) 
    # Gamma 鎖定在健康區間
    gamma = trial.suggest_float("gamma", 0.90, 0.995) 
    tau = trial.suggest_float("tau", 0.001, 0.05, log=True)
    
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    # 增加 Buffer Size 選項，應對複雜環境
    buffer_size = trial.suggest_categorical("buffer_size", [50000, 100000, 200000]) 
    
    # Entropy Coefficient 處理
    ent_coef_mode = trial.suggest_categorical("ent_coef_mode", ["auto", "fixed"])
    if ent_coef_mode == "fixed":
        ent_coef = trial.suggest_float("ent_coef_val", 0.001, 0.5, log=True)
    else:
        ent_coef = "auto"
    
    # --- 網路架構 (Neural Network Architecture) ---
    n_layers = trial.suggest_int("n_layers", 2, 3)
    
    if n_layers == 2:
        neurons_layer1 = trial.suggest_categorical("neurons_layer1", [128, 256, 512])
        neurons_layer2 = trial.suggest_categorical("neurons_layer2", [128, 256, 512])
        net_arch = [neurons_layer1, neurons_layer2]
    elif n_layers == 3:
        neurons_layer1 = trial.suggest_categorical("neurons_layer1", [128, 256, 512])
        neurons_layer2 = trial.suggest_categorical("neurons_layer2", [128, 256, 512])
        neurons_layer3 = trial.suggest_categorical("neurons_layer3", [128, 256, 512])
        net_arch = [neurons_layer1, neurons_layer2, neurons_layer3]
    
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, qf=net_arch)
    )
    
    # ==========================================
    # 2. 建立 Trial 專屬配置
    # ==========================================
    
    config = copy.deepcopy(base_config)
    config['experiment_name'] = f"optuna_trial_{trial.number}"
    config['random_seed'] = args.seed + trial.number
    
    # 增加評估頻率與次數，減少隨機誤差
    config['eval_freq'] = max(5000, config['total_timesteps'] // 10)
    config['n_eval_episodes'] = 15 
    
    # --- 更新 Agent 超參數 ---
    config['agent']['learning_rate'] = learning_rate
    config['agent']['buffer_size'] = buffer_size
    config['agent']['batch_size'] = batch_size
    config['agent']['gamma'] = gamma
    config['agent']['tau'] = tau
    config['agent']['ent_coef'] = ent_coef # 已經處理過 auto/float 邏輯
    
    # 路徑設定
    config['logging']['log_dir'] = os.path.join(args.optuna_log_dir, f"trial_{trial.number}")
    config['logging']['save_dir'] = os.path.join(args.optuna_model_dir, f"trial_{trial.number}")
    config['logging']['verbose'] = 0
    
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    
    # ==========================================
    # 3. 建立環境
    # ==========================================
    
    try:
        train_env = NetworkSlicingEnv(config)
        train_env = Monitor(train_env, filename=os.path.join(config['logging']['log_dir'], "train_monitor"))
        
        eval_env = NetworkSlicingEnv(config)
        eval_env = Monitor(eval_env, filename=os.path.join(config['logging']['log_dir'], "eval_monitor"))
        
    except Exception as e:
        print(f"Trial {trial.number} 環境建立失敗: {e}")
        raise optuna.TrialPruned()
    
    # ==========================================
    # 4. 建立 SAC 模型
    # ==========================================
    
    try:
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            ent_coef=ent_coef, # 直接使用處理後的變數
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=config['random_seed'],
        )
    except Exception as e:
        print(f"Trial {trial.number} 模型建立失敗: {e}")
        train_env.close()
        eval_env.close()
        raise optuna.TrialPruned()
    
    # ==========================================
    # 5. 分段訓練 + 剪枝 (邏輯保持不變)
    # ==========================================
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config['logging']['save_dir'],
        log_path=config['logging']['log_dir'],
        eval_freq=config['eval_freq'],
        n_eval_episodes=config['n_eval_episodes'],
        deterministic=True,
        render=False,
        verbose=0
    )
    
    try:
        n_checkpoints = 5
        timesteps_per_checkpoint = config['total_timesteps'] // n_checkpoints
        
        for checkpoint in range(n_checkpoints):
            model.learn(
                total_timesteps=timesteps_per_checkpoint,
                callback=eval_callback,
                reset_num_timesteps=False,
                progress_bar=False
            )
            
            # 讀取評估結果
            eval_log_path = os.path.join(config['logging']['log_dir'], "evaluations.npz")
            if os.path.exists(eval_log_path):
                evaluations = np.load(eval_log_path)
                results = evaluations['results']
                if len(results) > 0:
                    # 取最後幾次的平均作為當前分數
                    current_mean_reward = float(results[-1].mean())
                    trial.report(current_mean_reward, checkpoint)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
        
    except optuna.TrialPruned:
        print(f"Trial {trial.number} Pruned.")
        train_env.close()
        eval_env.close()
        raise
    except Exception as e:
        print(f"Trial {trial.number} Failed: {e}")
        train_env.close()
        eval_env.close()
        raise optuna.TrialPruned()
    
    # ==========================================
    # 6. 提取最終結果
    # ==========================================
    
    eval_log_path = os.path.join(config['logging']['log_dir'], "evaluations.npz")
    if os.path.exists(eval_log_path):
        evaluations = np.load(eval_log_path)
        results = evaluations['results']
        # 取最後 5 次評估的平均，比較能代表最終收斂效果
        if len(results) >= 5:
            mean_reward = float(results[-5:].mean())
        else:
            mean_reward = float(results.mean())
    else:
        mean_reward = -1e10
    
    train_env.close()
    eval_env.close()
    
    # 記錄關鍵配置 (僅供紀錄，方便日後查找用的是哪個流量場景)
    trial.set_user_attr("final_timesteps", config['total_timesteps'])
    # 注意：這裡改為記錄 config 裡的值，因為局部變數已經刪除了
    if 'traffic' in config:
        trial.set_user_attr("embb_rate", config['traffic'].get('embb_arrival_rate_mbps', 'N/A'))
        trial.set_user_attr("urllc_rate", config['traffic'].get('urllc_arrival_rate_mbps', 'N/A'))
    
    return mean_reward

def main():
    # ==========================================
    # 命令列參數解析
    # ==========================================
    
    parser = argparse.ArgumentParser(description="Optuna 超參數自動化調優")
    
    parser.add_argument(
        "--n-trials", 
        type=int, 
        default=50, 
        help="總試驗次數 (建議 50-200)"
    )
    parser.add_argument(
        "--n-jobs", 
        type=int, 
        default=1, 
        help="平行工作數 (建議設為 CPU 核心數，如 4 或 8)"
    )
    parser.add_argument(
        "--study-name", 
        type=str, 
        default="5g_slicing_sac_v1", 
        help="Study 名稱 (可重複使用以繼續上次的搜索)"
    )
    parser.add_argument(
        "--storage", 
        type=str, 
        default="sqlite:///optuna_study.db", 
        help="Optuna 資料庫位置 (SQLite 或 MySQL)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_config.yaml", 
        help="基礎配置檔路徑"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="基礎隨機種子"
    )
    parser.add_argument(
        "--optuna-log-dir", 
        type=str, 
        default="./optuna_logs", 
        help="Optuna 訓練日誌根目錄"
    )
    parser.add_argument(
        "--optuna-model-dir", 
        type=str, 
        default="./optuna_models", 
        help="Optuna 模型儲存根目錄"
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=None, 
        help="最大搜索時間 (秒)，None 表示不限制"
    )
    
    args = parser.parse_args()

    # ==========================================
    # 執行前清理與建立輸出資料夾/檔案
    # ==========================================

    prepare_output_paths(args)
    
    # ==========================================
    # 載入基礎配置
    # ==========================================
    
    print("=== Optuna 超參數調優開始 ===\n")
    print(f"基礎配置: {args.config}")
    print(f"總試驗次數: {args.n_trials}")
    print(f"平行工作數: {args.n_jobs}")
    print(f"Study 名稱: {args.study_name}")
    print(f"資料庫: {args.storage}\n")
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置檔不存在: {args.config}")
    
    with open(args.config, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    # 建立輸出目錄
    os.makedirs(args.optuna_log_dir, exist_ok=True)
    os.makedirs(args.optuna_model_dir, exist_ok=True)
    
    # ==========================================
    # 建立 Optuna Study
    # ==========================================
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,  # 允許繼續之前的搜索
        direction="maximize",  # 最大化 mean_reward
        sampler=TPESampler(
            seed=args.seed,
            n_startup_trials=10,  # 前 10 個 trial 用隨機採樣 (warm-up)
            multivariate=True  # 考慮參數間的相關性
        ),
        pruner=MedianPruner(
            n_startup_trials=5,  # 前 5 個 trial 不剪枝
            n_warmup_steps=5,    # 評估 5 次後才開始剪枝
            interval_steps=1     # 每次評估後都檢查是否要剪枝
        )
    )
    
    # ==========================================
    # 開始優化
    # ==========================================
    
    print("開始搜索最佳超參數...\n")
    
    study.optimize(
        lambda trial: objective(trial, base_config, args),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        show_progress_bar=True,
        catch=(Exception,)  # 捕捉異常但繼續執行其他 trials
    )
    
    # ==========================================
    # 輸出結果
    # ==========================================
    
    print("\n" + "="*60)
    print("=== 優化完成 ===")
    print("="*60 + "\n")
    
    print(f"完成的 Trials 數量: {len(study.trials)}")
    print(f"被剪枝的 Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"失敗的 Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    print(f"成功的 Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}\n")
    
    if len(study.best_trials) > 0:
        print(f"🏆 最佳 Trial: #{study.best_trial.number}")
        print(f"🏆 最佳 Mean Reward: {study.best_value:.2f}\n")
        
        print("最佳超參數:")
        print("-" * 60)
        for key, value in study.best_params.items():
            print(f"  {key:30s}: {value}")
        print("-" * 60 + "\n")
        
        # ==========================================
        # 儲存最佳配置
        # ==========================================
        
        best_config_path = "configs/best_config_optuna.yaml"
        
        # 建立完整的配置檔 (包含最佳超參數)
        final_config = base_config.copy()
        
        # 更新 Agent 超參數
        final_config['agent']['learning_rate'] = study.best_params['learning_rate']
        final_config['agent']['buffer_size'] = study.best_params['buffer_size']
        final_config['agent']['batch_size'] = study.best_params['batch_size']
        final_config['agent']['gamma'] = study.best_params['gamma']
        final_config['agent']['tau'] = study.best_params['tau']
        if study.best_params['ent_coef_mode'] == 'auto':
            final_config['agent']['ent_coef'] = 'auto'
        else:
            final_config['agent']['ent_coef'] = study.best_params['ent_coef_val']
        
        # 更新神經網路架構
        n_layers = study.best_params['n_layers']
        net_arch = []
        for i in range(1, n_layers + 1):
            layer_key = f'neurons_layer{i}'
            if layer_key in study.best_params:
                net_arch.append(study.best_params[layer_key])
        
        # 儲存網路架構到配置檔
        if 'policy_kwargs' not in final_config['agent']:
            final_config['agent']['policy_kwargs'] = {}
        final_config['agent']['policy_kwargs']['net_arch'] = {
            'pi': net_arch,  # Actor 網路
            'qf': net_arch   # Critic 網路
        }
        
        # 儲存到 YAML
        with open(best_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(final_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ 最佳配置已儲存至: {best_config_path}")
        print(f"   可使用以下指令進行完整訓練:")
        print(f"   python main.py --config {best_config_path}\n")
        
    else:
        print("⚠️  沒有成功完成的 Trials，請檢查配置或降低 timesteps")
    
    # ==========================================
    # 視覺化結果 (需要 plotly)
    # ==========================================
    
    try:
        import plotly
        
        print("生成視覺化圖表...")
        
        # 1. 優化歷史
        fig_history = optuna.visualization.plot_optimization_history(study)
        fig_history.write_html("optuna_history.html")
        
        # 2. 參數重要性
        if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) >= 10:
            fig_importance = optuna.visualization.plot_param_importances(study)
            fig_importance.write_html("optuna_importance.html")
        
        # 3. 超參數關係 (Parallel Coordinate)
        fig_parallel = optuna.visualization.plot_parallel_coordinate(study)
        fig_parallel.write_html("optuna_parallel_coordinate.html")
        
        # 4. Slice Plot (每個參數的影響)
        fig_slice = optuna.visualization.plot_slice(study)
        fig_slice.write_html("optuna_slice.html")
        
        print("✅ 視覺化結果已儲存:")
        print("   - optuna_history.html (優化歷史)")
        print("   - optuna_importance.html (參數重要性)")
        print("   - optuna_parallel_coordinate.html (參數關係)")
        print("   - optuna_slice.html (參數影響)\n")
        
    except ImportError:
        print("⚠️  未安裝 plotly，跳過視覺化")
        print("   安裝方式: pip install plotly\n")
    
    # ==========================================
    # 輸出 Dashboard 指令
    # ==========================================
    
    print("="*60)
    print("💡 提示：使用以下指令查看即時 Dashboard:")
    print(f"   optuna-dashboard {args.storage}")
    print("   (需先安裝: pip install optuna-dashboard)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
