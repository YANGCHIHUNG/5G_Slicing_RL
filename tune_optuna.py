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
"""

import os
import yaml
import argparse
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from src.envs.slicing_env import NetworkSlicingEnv


def objective(trial: optuna.Trial, base_config: dict, args: argparse.Namespace):
    """
    Optuna 目標函數：訓練一個 SAC Agent 並返回評估獎勵
    
    Args:
        trial: Optuna Trial 物件
        base_config: 基礎配置檔 (來自 default_config.yaml)
        args: 命令列參數
        
    Returns:
        float: 評估階段的平均獎勵 (mean_reward)
    """
    
    # ==========================================
    # 1. 定義搜索空間 (Search Space)
    # ==========================================
    
    # --- SAC 核心超參數 ---
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.9999)
    tau = trial.suggest_float("tau", 0.001, 0.02, log=True)
    ent_coef = trial.suggest_categorical("ent_coef", ["auto", "auto_0.1", "auto_0.01"])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    buffer_size = trial.suggest_categorical("buffer_size", [10000, 30000, 50000, 100000])
    
    # --- 網路架構 ---
    # 搜索神經網路層數與神經元數
    n_layers = trial.suggest_int("n_layers", 2, 4)  # 2-4 層隱藏層
    
    # 為每一層選擇神經元數（可以不同）
    if n_layers == 2:
        neurons_layer1 = trial.suggest_categorical("neurons_layer1", [64, 128, 256, 512])
        neurons_layer2 = trial.suggest_categorical("neurons_layer2", [64, 128, 256, 512])
        net_arch = [neurons_layer1, neurons_layer2]
    elif n_layers == 3:
        neurons_layer1 = trial.suggest_categorical("neurons_layer1", [64, 128, 256, 512])
        neurons_layer2 = trial.suggest_categorical("neurons_layer2", [64, 128, 256, 512])
        neurons_layer3 = trial.suggest_categorical("neurons_layer3", [64, 128, 256, 512])
        net_arch = [neurons_layer1, neurons_layer2, neurons_layer3]
    else:  # n_layers == 4
        neurons_layer1 = trial.suggest_categorical("neurons_layer1", [64, 128, 256, 512])
        neurons_layer2 = trial.suggest_categorical("neurons_layer2", [64, 128, 256, 512])
        neurons_layer3 = trial.suggest_categorical("neurons_layer3", [64, 128, 256, 512])
        neurons_layer4 = trial.suggest_categorical("neurons_layer4", [64, 128, 256, 512])
        net_arch = [neurons_layer1, neurons_layer2, neurons_layer3, neurons_layer4]
    
    # 建立 policy_kwargs
    policy_kwargs = dict(
        net_arch=dict(
            pi=net_arch,  # Actor 網路架構
            qf=net_arch   # Critic 網路架構（Q-function）
        )
    )
    
    # --- 獎勵函數權重 ---
    w_throughput = trial.suggest_float("w_throughput", 0.01, 2.0, log=True)
    w_latency = trial.suggest_float("w_latency", 10.0, 300.0, log=True)
    drop_penalty = trial.suggest_float("drop_penalty", 10.0, 500.0, log=True)
    
    # --- 流量負載場景 ---
    embb_rate = trial.suggest_float("embb_arrival_rate_mbps", 50.0, 250.0)
    urllc_rate = trial.suggest_float("urllc_arrival_rate_mbps", 2.0, 25.0)
    
    # --- 環境設定 ---
    env_max_steps = trial.suggest_categorical("env_max_steps", [1000, 2000, 3000, 5000])
    min_rbs_urllc = trial.suggest_int("min_rbs_urllc", 0, 40)
    normalize_obs = trial.suggest_categorical("normalize_obs", [True, False])
    
    # ==========================================
    # 2. 建立 Trial 專屬配置
    # ==========================================
    
    config = base_config.copy()
    config['experiment_name'] = f"optuna_trial_{trial.number}"
    config['random_seed'] = args.seed + trial.number  # 每個 trial 不同 seed
    config['total_timesteps'] = args.timesteps  # 使用縮短的訓練步數加速搜索
    config['eval_freq'] = max(2000, args.timesteps // 10)  # 至少評估 10 次
    config['n_eval_episodes'] = 5  # 減少評估 episodes 加速
    
    # 更新搜索到的超參數
    config['agent']['learning_rate'] = learning_rate
    config['agent']['buffer_size'] = buffer_size
    config['agent']['batch_size'] = batch_size
    config['agent']['gamma'] = gamma
    config['agent']['tau'] = tau
    
    # 處理 ent_coef (Optuna 不支援直接 suggest "auto"，需轉換)
    if ent_coef == "auto":
        config['agent']['ent_coef'] = "auto"
    elif ent_coef == "auto_0.1":
        config['agent']['ent_coef'] = "auto_0.1"
    elif ent_coef == "auto_0.01":
        config['agent']['ent_coef'] = "auto_0.01"
    else:
        config['agent']['ent_coef'] = float(ent_coef)
    
    config['reward']['w_throughput'] = w_throughput
    config['reward']['w_latency'] = w_latency
    config['reward']['drop_penalty'] = drop_penalty
    
    config['traffic']['embb_arrival_rate_mbps'] = embb_rate
    config['traffic']['urllc_arrival_rate_mbps'] = urllc_rate
    
    # 環境設定 (如果 base_config 沒有 'env' key，建立它)
    if 'env' not in config:
        config['env'] = {}
    config['env']['env_max_steps'] = env_max_steps
    config['env']['min_rbs_urllc'] = min_rbs_urllc
    config['env']['normalize_obs'] = normalize_obs
    
    # 路徑設定
    config['logging']['log_dir'] = os.path.join(args.optuna_log_dir, f"trial_{trial.number}")
    config['logging']['save_dir'] = os.path.join(args.optuna_model_dir, f"trial_{trial.number}")
    config['logging']['verbose'] = 0  # 減少輸出
    
    # 建立目錄
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    
    # ==========================================
    # 3. 建立環境
    # ==========================================
    
    try:
        train_env = NetworkSlicingEnv(config)
        train_env = Monitor(
            train_env, 
            filename=os.path.join(config['logging']['log_dir'], "train_monitor"),
            allow_early_resets=True
        )
        
        eval_env = NetworkSlicingEnv(config)
        eval_env = Monitor(
            eval_env,
            filename=os.path.join(config['logging']['log_dir'], "eval_monitor"),
            allow_early_resets=True
        )
        
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
            ent_coef=config['agent']['ent_coef'],
            policy_kwargs=policy_kwargs,  # 使用自定義網路架構
            verbose=0,
            seed=config['random_seed'],
            tensorboard_log=None  # 不使用 TensorBoard 節省空間
        )
    except Exception as e:
        print(f"Trial {trial.number} 模型建立失敗: {e}")
        train_env.close()
        eval_env.close()
        raise optuna.TrialPruned()
    
    # ==========================================
    # 5. 建立 Callbacks
    # ==========================================
    
    # 評估 Callback
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
    
    # ==========================================
    # 6. 分段訓練 + 手動剪枝
    # ==========================================
    
    try:
        # 將訓練分成多個階段，每個階段後檢查是否要剪枝
        n_checkpoints = 5
        timesteps_per_checkpoint = config['total_timesteps'] // n_checkpoints
        
        for checkpoint in range(n_checkpoints):
            # 訓練一段時間
            model.learn(
                total_timesteps=timesteps_per_checkpoint,
                callback=eval_callback,
                reset_num_timesteps=False,
                progress_bar=False
            )
            
            # 讀取當前評估結果
            eval_log_path = os.path.join(config['logging']['log_dir'], "evaluations.npz")
            if os.path.exists(eval_log_path):
                evaluations = np.load(eval_log_path)
                results = evaluations['results']
                if len(results) > 0:
                    current_mean_reward = float(results[-1].mean())
                    
                    # 回報給 Optuna
                    trial.report(current_mean_reward, checkpoint)
                    
                    # 檢查是否要剪枝
                    if trial.should_prune():
                        raise optuna.TrialPruned()
        
    except optuna.TrialPruned:
        print(f"Trial {trial.number} 被剪枝 (Pruned) at checkpoint {checkpoint}/{n_checkpoints}")
        train_env.close()
        eval_env.close()
        raise
    except Exception as e:
        print(f"Trial {trial.number} 訓練失敗: {e}")
        train_env.close()
        eval_env.close()
        raise optuna.TrialPruned()
    
    # ==========================================
    # 7. 提取評估結果
    # ==========================================
    
    # 從 evaluations.npz 讀取最終評估獎勵
    eval_log_path = os.path.join(config['logging']['log_dir'], "evaluations.npz")
    
    if os.path.exists(eval_log_path):
        try:
            evaluations = np.load(eval_log_path)
            # 取最後幾次評估的平均值 (更穩定)
            results = evaluations['results']
            if len(results) >= 3:
                mean_reward = float(results[-3:].mean())  # 最後 3 次評估的平均
            else:
                mean_reward = float(results.mean())
        except Exception as e:
            print(f"Trial {trial.number} 讀取評估結果失敗: {e}")
            mean_reward = -1e10  # 給予極差的分數
    else:
        print(f"Trial {trial.number} 沒有找到評估日誌")
        mean_reward = -1e10
    
    # ==========================================
    # 8. 清理資源
    # ==========================================
    
    train_env.close()
    eval_env.close()
    
    # 記錄額外資訊到 Trial (供後續分析)
    trial.set_user_attr("final_timesteps", config['total_timesteps'])
    trial.set_user_attr("embb_rate", embb_rate)
    trial.set_user_attr("urllc_rate", urllc_rate)
    
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
        "--timesteps", 
        type=int, 
        default=50000, 
        help="每個 trial 的訓練步數 (縮短以加速搜索，建議 30k-100k)"
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
    # 載入基礎配置
    # ==========================================
    
    print("=== Optuna 超參數調優開始 ===\n")
    print(f"基礎配置: {args.config}")
    print(f"總試驗次數: {args.n_trials}")
    print(f"平行工作數: {args.n_jobs}")
    print(f"每個 Trial 訓練步數: {args.timesteps}")
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
        final_config['agent']['ent_coef'] = study.best_params['ent_coef']
        
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
        
        # 更新 Reward 權重
        final_config['reward']['w_throughput'] = study.best_params['w_throughput']
        final_config['reward']['w_latency'] = study.best_params['w_latency']
        final_config['reward']['drop_penalty'] = study.best_params['drop_penalty']
        
        # 更新流量場景
        final_config['traffic']['embb_arrival_rate_mbps'] = study.best_params['embb_arrival_rate_mbps']
        final_config['traffic']['urllc_arrival_rate_mbps'] = study.best_params['urllc_arrival_rate_mbps']
        
        # 更新環境設定
        if 'env' not in final_config:
            final_config['env'] = {}
        final_config['env']['env_max_steps'] = study.best_params['env_max_steps']
        final_config['env']['min_rbs_urllc'] = study.best_params['min_rbs_urllc']
        final_config['env']['normalize_obs'] = study.best_params['normalize_obs']
        
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
