# 5G Network Slicing RL - 流程 Pseudocode

## 1. 主訓練流程 (main.py)

```
ALGORITHM TrainSACAgent
INPUT: config_file (default_config.yaml)
OUTPUT: trained_model, training_logs

BEGIN
    // ==================== 初始化 ====================
    config ← LoadConfigFromYAML(config_file)
    
    // 建立環境 (訓練+評估)
    train_env ← CREATE NetworkSlicingEnv(config, mode="train")
    eval_env ← CREATE NetworkSlicingEnv(config, mode="eval")
    
    // 初始化 SAC Agent
    agent ← CREATE SACAgent(
        policy="MlpPolicy",
        learning_rate=config.agent.learning_rate,
        gamma=config.agent.gamma,
        buffer_size=config.agent.buffer_size,
        batch_size=config.agent.batch_size,
        policy_kwargs=config.agent.policy_kwargs
    )
    
    // 建立回調函數
    eval_callback ← CREATE EvalCallback(
        eval_env,
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        save_path="./models/"
    )
    
    // ==================== 訓練迴圈 ====================
    best_mean_reward ← -∞
    total_timesteps ← config.total_timesteps
    
    FOR timestep = 1 TO total_timesteps DO
        // 在訓練環境中運行一步
        obs ← train_env.reset() IF timestep % episode_length == 0 ELSE obs
        action ← agent.predict(obs, deterministic=FALSE)
        obs_next, reward, done, info ← train_env.step(action)
        
        // 儲存轉移到 Replay Buffer
        agent.replay_buffer.add(obs, action, reward, obs_next, done)
        
        // SAC 訓練更新 (若 Replay Buffer 充足)
        IF agent.replay_buffer.size() > config.batch_size THEN
            loss ← agent.train_step()
            log_loss(loss)
        END IF
        
        // 定期評估
        IF timestep % config.eval_freq == 0 THEN
            mean_reward ← eval_callback.evaluate()
            log_metric("Mean Reward", mean_reward)
            
            // 保存最佳模型
            IF mean_reward > best_mean_reward THEN
                best_mean_reward ← mean_reward
                agent.save("./models/best_model.zip")
            END IF
        END IF
        
        obs ← obs_next
    END FOR
    
    // ==================== 後處理 ====================
    agent.save("./models/final_model.zip")
    training_logs ← export_tensorboard_logs()
    
    RETURN (agent, training_logs)
END

```

---

## 2. 環境模擬迴圈 (slicing_env.py)

```
ALGORITHM EnvironmentStep
INPUT: action [w_embb, w_urllc]
OUTPUT: observation, reward, done, info

BEGIN
    // ==================== 狀態獲取 ====================
    cqis ← self.channel.step()                    // 更新通道 CQI
    bits_arrived ← self.traffic.step()            // 生成流量
    
    // ==================== 加入新流量 ====================
    FOR slice IN [eMBB, URLLC] DO
        num_packets ← bits_arrived[slice] / packet_size[slice]
        FOR i = 1 TO num_packets DO
            self.buffer[slice].add_packet(
                size_bits=packet_size[slice],
                arrival_time=current_time
            )
        END FOR
    END FOR
    
    // ==================== 基地台資源分配 ====================
    info ← BaseStationStep(action, cqis)
    
    // 結果包含：
    // - throughput_embb, throughput_urllc
    // - latency_embb, latency_urllc
    // - num_dropped_urllc
    // - buffer_state
    
    // ==================== 獎勵計算 ====================
    total_throughput ← info.throughput_embb + info.throughput_urllc
    latency_urllc ← info.latency_urllc
    dropped_urllc ← info.num_dropped_urllc
    
    reward ← (config.reward.w_throughput × total_throughput
              - config.reward.w_latency × latency_urllc
              - config.reward.drop_penalty × dropped_urllc)
    
    // ==================== 狀態構建 ====================
    observation ← [
        info.buffer_load_embb,
        info.head_of_line_embb,
        info.buffer_load_urllc,
        info.head_of_line_urllc,
        cqis[eMBB],
        cqis[URLLC]
    ]
    
    // ==================== 終止條件判定 ====================
    done ← (current_timestep >= env_max_steps)
    
    RETURN (observation, reward, done, info)
END

```

---

## 3. 基地台資源分配核心 (base_station.py)

```
ALGORITHM BaseStationStep
INPUT: action_weights [w_embb, w_urllc], cqis
OUTPUT: info {throughput, latency, dropped, buffer_state, ...}

BEGIN
    // ==================== 參數準備 ====================
    w_embb, w_urllc ← normalize(action_weights)     // 和 = 1.0
    
    // ==================== 資源分配（權重 → RBs） ====================
    total_rbs ← 273                                 // 3GPP 標準
    num_rbs_embb ← FLOOR(total_rbs × w_embb)
    num_rbs_urllc ← total_rbs - num_rbs_embb
    
    // ==================== 物理層計算 ====================
    FUNCTION CalculateCapacity(num_rbs, cqi)
        efficiency ← get_spectral_efficiency(cqi)   // bits/symbol (查表)
        subcarriers_per_rb ← 12
        symbols_per_tti ← 14
        overhead_factor ← 0.86                      // 控制信道開銷
        
        capacity_bits ← (num_rbs 
                        × subcarriers_per_rb 
                        × symbols_per_tti 
                        × efficiency 
                        × overhead_factor)
        
        RETURN capacity_bits / tti_duration_us      // bits/s (Mbps)
    END FUNCTION
    
    cap_embb ← CalculateCapacity(num_rbs_embb, cqis[eMBB])
    cap_urllc ← CalculateCapacity(num_rbs_urllc, cqis[URLLC])
    
    // ==================== 封包傳輸 ====================
    FOR slice IN [eMBB, URLLC] DO
        tx_bits, delays ← self.buffer[slice].remove_packets(
            capacity=cap[slice],
            current_time=current_time
        )
        
        // 記錄統計
        info.throughput[slice] ← tx_bits / tti_duration
        info.latency[slice] ← MEAN(delays)
    END FOR
    
    // ==================== 超時檢查（URLLC 專用） ====================
    max_latency_urllc ← 1.0 ms                      // QoS 要求
    dropped ← 0
    
    WHILE self.buffer_urllc.head_of_line_age() > max_latency_urllc DO
        packet ← self.buffer_urllc.dequeue()        // 移除超時封包
        dropped ← dropped + 1
    END WHILE
    
    info.num_dropped_urllc ← dropped
    
    // ==================== 緩衝區狀態提取 ====================
    FOR slice IN [eMBB, URLLC] DO
        info.buffer_load[slice] ← self.buffer[slice].total_bits / max_buffer_bits
        info.head_of_line[slice] ← self.buffer[slice].head_of_line_age()
    END FOR
    
    // ==================== 時間推進 ====================
    current_time ← current_time + tti_duration
    
    RETURN info
END

```

---

## 4. 流量生成 (traffic.py)

```
ALGORITHM TrafficGeneratorStep
INPUT: arrival_rate_mbps, packet_size_bits, tti_duration_us
OUTPUT: num_packets, total_bits

BEGIN
    // Poisson 流量模型
    // Lambda = (arrival_rate_Mbps × 10^6 × tti_duration_s) / packet_size_bits
    
    arrival_rate_bps ← arrival_rate_mbps × 10^6
    expected_bits_per_tti ← arrival_rate_bps × tti_duration_us / 10^6
    expected_packets ← expected_bits_per_tti / packet_size_bits
    
    // 從 Poisson 分佈採樣
    num_packets ← POISSON(expected_packets)
    total_bits ← num_packets × packet_size_bits
    
    RETURN (num_packets, total_bits)
END

```

---

## 5. 緩衝區管理 (buffers.py)

```
ALGORITHM SliceBufferRemovePackets
INPUT: capacity_bits, current_time
OUTPUT: transmitted_bits, latencies

BEGIN
    transmitted_bits ← 0
    latencies ← []
    
    // FIFO 佇列處理
    WHILE buffer.size() > 0 AND transmitted_bits < capacity_bits DO
        packet ← buffer.front()
        
        // 檢查容量
        IF transmitted_bits + packet.size_bits > capacity_bits THEN
            BREAK  // 容量不足，無法傳送此封包
        END IF
        
        // 傳送封包
        buffer.dequeue()
        transmitted_bits ← transmitted_bits + packet.size_bits
        latency ← current_time - packet.arrival_time
        latencies.append(latency)
    END WHILE
    
    RETURN (transmitted_bits, latencies)
END

ALGORITHM SliceBufferCheckTimeout
INPUT: current_time, max_latency_ms
OUTPUT: num_dropped

BEGIN
    num_dropped ← 0
    
    // 刪除超時封包
    WHILE buffer.size() > 0 DO
        packet ← buffer.front()
        age ← current_time - packet.arrival_time
        
        IF age > max_latency_ms THEN
            buffer.dequeue()
            num_dropped ← num_dropped + 1
        ELSE
            BREAK  // 由於 FIFO，之後的都不會超時
        END IF
    END WHILE
    
    RETURN num_dropped
END

```

---

## 6. 通道模擬 (channel.py)

```
ALGORITHM ChannelSimulatorStep
INPUT: None
OUTPUT: cqis [cqi_embb, cqi_urllc]

BEGIN
    // Bounded Random Walk 模型
    // CQI 值範圍：[1, 15]
    // 90% 機率不變，10% 機率 ±1
    
    FOR slice IN [eMBB, URLLC] DO
        u ← UNIFORM(0, 1)
        
        IF u < 0.9 THEN
            // 保持不變
            cqis[slice] ← cqis[slice]
        ELSE
            // 隨機漂移
            delta ← RANDOM_SIGN() × 1  // -1 或 +1
            cqis[slice] ← CLIP(cqis[slice] + delta, min=1, max=15)
        END IF
    END FOR
    
    RETURN cqis
END

```

---

## 7. 評估流程 (evaluate.py)

```
ALGORITHM EvaluateTrainedModel
INPUT: model_path, eval_steps
OUTPUT: kpis {throughput, latency, violation_rate, ...}

BEGIN
    // ==================== 模型載入 ====================
    agent ← load_model(model_path)
    env ← CREATE NetworkSlicingEnv(config, mode="eval")
    
    // ==================== 初始化數據收集 ====================
    metrics ← {
        throughput_embb: [],
        throughput_urllc: [],
        latency_urllc: [],
        dropped_urllc: 0,
        total_packets_urllc: 0
    }
    
    // ==================== 評估迴圈 ====================
    obs ← env.reset()
    
    FOR step = 1 TO eval_steps DO
        // 純確定性推論（無探索噪聲）
        action ← agent.predict(obs, deterministic=TRUE)
        obs, reward, done, info ← env.step(action)
        
        // 收集指標
        metrics.throughput_embb.append(info.throughput_embb)
        metrics.throughput_urllc.append(info.throughput_urllc)
        metrics.latency_urllc.append(info.latency_urllc)
        metrics.dropped_urllc ← metrics.dropped_urllc + info.num_dropped_urllc
        metrics.total_packets_urllc ← metrics.total_packets_urllc + info.num_packets_urllc
        
        IF done THEN
            obs ← env.reset()
        END IF
    END FOR
    
    // ==================== KPI 計算 ====================
    kpis.mean_throughput_embb ← MEAN(metrics.throughput_embb)
    kpis.mean_throughput_urllc ← MEAN(metrics.throughput_urllc)
    kpis.mean_latency_urllc ← MEAN(metrics.latency_urllc)
    
    // 違規率：超過 1ms 的 URLLC 延遲比例
    violations ← COUNT(metrics.latency_urllc > 1.0)
    kpis.violation_rate ← violations / LENGTH(metrics.latency_urllc)
    
    // 丟包率
    IF metrics.total_packets_urllc > 0 THEN
        kpis.drop_rate ← metrics.dropped_urllc / metrics.total_packets_urllc
    ELSE
        kpis.drop_rate ← 0
    END IF
    
    // ==================== 視覺化生成 ====================
    plot_throughput_time_series(metrics.throughput_embb, metrics.throughput_urllc)
    plot_rb_allocation_stacked(recorded_rb_allocations)
    plot_latency_cdf(metrics.latency_urllc)
    
    RETURN kpis
END

```

---

## 8. 超參數自動調優 (tune_optuna.py)

```
ALGORITHM Optuna_HyperparameterTuning
INPUT: n_trials, n_jobs
OUTPUT: best_config_yaml, visualization_htmls

BEGIN
    // ==================== 初始化清理 ====================
    IF optuna_logs_dir EXISTS THEN
        DELETE optuna_logs_dir RECURSIVELY
    END IF
    IF optuna_models_dir EXISTS THEN
        DELETE optuna_models_dir RECURSIVELY
    END IF
    IF optuna_study.db EXISTS THEN
        DELETE optuna_study.db
    END IF
    
    CREATE optuna_logs_dir
    CREATE optuna_models_dir
    
    // ==================== 建立 Study ====================
    study ← CREATE OptunaStudy(
        name="5g_slicing_sac_v1",
        storage="sqlite:///optuna_study.db",
        direction="maximize",           // 最大化 mean_reward
        sampler=TPESampler(
            n_startup_trials=10,        // 前 10 個用隨機採樣
            multivariate=TRUE           // 考慮參數相關性
        ),
        pruner=MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        )
    )
    
    // ==================== 目標函數定義 ====================
    FUNCTION Objective(trial)
        // ----- 搜索空間定義 -----
        learning_rate ← trial.suggest_float(
            "learning_rate", 5e-5, 1e-3, log=TRUE
        )
        gamma ← trial.suggest_float("gamma", 0.90, 0.995)
        tau ← trial.suggest_float("tau", 0.001, 0.05, log=TRUE)
        
        batch_size ← trial.suggest_categorical(
            "batch_size", [128, 256, 512]
        )
        buffer_size ← trial.suggest_categorical(
            "buffer_size", [50000, 100000, 200000]
        )
        
        ent_coef_mode ← trial.suggest_categorical(
            "ent_coef_mode", ["auto", "fixed"]
        )
        IF ent_coef_mode == "fixed" THEN
            ent_coef ← trial.suggest_float(
                "ent_coef_val", 0.001, 0.5, log=TRUE
            )
        ELSE
            ent_coef ← "auto"
        END IF
        
        n_layers ← trial.suggest_int("n_layers", 2, 3)
        net_arch ← []
        FOR i = 1 TO n_layers DO
            neurons ← trial.suggest_categorical(
                f"neurons_layer{i}", [128, 256, 512]
            )
            net_arch.append(neurons)
        END FOR
        
        // ----- Trial 專屬配置 -----
        config ← COPY(base_config)
        config.agent.learning_rate ← learning_rate
        config.agent.buffer_size ← buffer_size
        config.agent.batch_size ← batch_size
        config.agent.gamma ← gamma
        config.agent.tau ← tau
        config.agent.ent_coef ← ent_coef
        config.agent.policy_kwargs.net_arch ← {
            "pi": net_arch,
            "qf": net_arch
        }
        
        config.logging.log_dir ← f"optuna_logs/trial_{trial.number}"
        config.logging.save_dir ← f"optuna_models/trial_{trial.number}"
        
        // ----- 訓練環境建立 -----
        train_env ← CREATE NetworkSlicingEnv(config)
        eval_env ← CREATE NetworkSlicingEnv(config)
        
        // ----- 模型訓練 -----
        agent ← CREATE SACAgent(config)
        
        n_checkpoints ← 5
        timesteps_per_checkpoint ← config.total_timesteps / n_checkpoints
        
        FOR checkpoint = 0 TO n_checkpoints-1 DO
            // 訓練一個檢查點
            agent.learn(
                total_timesteps=timesteps_per_checkpoint,
                callback=EvalCallback(eval_env)
            )
            
            // 讀取評估結果
            IF evaluations.npz EXISTS THEN
                results ← LOAD(evaluations.npz)
                current_mean_reward ← MEAN(results[-1:])
                
                // 回報給 Optuna（用於剪枝）
                trial.report(current_mean_reward, checkpoint)
                
                // 檢查是否應該剪枝
                IF trial.should_prune() THEN
                    RAISE TrialPruned
                END IF
            END IF
        END FOR
        
        // ----- 最終評估 -----
        evaluations ← LOAD(evaluations.npz)
        final_reward ← MEAN(evaluations.results[-5:])  // 最後 5 次平均
        
        RETURN final_reward
    END FUNCTION
    
    // ==================== 優化執行 ====================
    study.optimize(
        objective=Objective,
        n_trials=n_trials,
        n_jobs=n_jobs,           // 並行執行
        catch=(Exception,)       // 容錯繼續
    )
    
    // ==================== 結果輸出 ====================
    best_trial ← study.best_trial
    best_params ← study.best_params
    best_value ← study.best_value
    
    PRINT "Best Trial: #" + best_trial.number
    PRINT "Best Value: " + best_value
    PRINT "Best Params: " + best_params
    
    // ==================== 配置保存 ====================
    final_config ← COPY(base_config)
    
    // 更新最佳 Agent 超參數
    final_config.agent.learning_rate ← best_params.learning_rate
    final_config.agent.buffer_size ← best_params.buffer_size
    final_config.agent.batch_size ← best_params.batch_size
    final_config.agent.gamma ← best_params.gamma
    final_config.agent.tau ← best_params.tau
    
    IF best_params.ent_coef_mode == "auto" THEN
        final_config.agent.ent_coef ← "auto"
    ELSE
        final_config.agent.ent_coef ← best_params.ent_coef_val
    END IF
    
    net_arch ← []
    FOR i = 1 TO best_params.n_layers DO
        net_arch.append(best_params[f"neurons_layer{i}"])
    END FOR
    final_config.agent.policy_kwargs.net_arch ← {
        "pi": net_arch,
        "qf": net_arch
    }
    
    SAVE_YAML(final_config, "configs/best_config_optuna.yaml")
    
    // ==================== 視覺化生成 ====================
    fig_history ← plot_optimization_history(study)
    fig_history.save("optuna_history.html")
    
    fig_importance ← plot_param_importances(study)
    fig_importance.save("optuna_importance.html")
    
    fig_parallel ← plot_parallel_coordinate(study)
    fig_parallel.save("optuna_parallel_coordinate.html")
    
    fig_slice ← plot_slice(study)
    fig_slice.save("optuna_slice.html")
    
    RETURN (best_params, final_config, visualization_htmls)
END

```

---

## 9. 資料流圖 (Data Flow)

```
┌─────────────────────────────────────────────────────────────┐
│                   訓練開始 (main.py)                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
            ┌────────────────────┐
            │  建立環境 & Agent   │
            └────────────┬───────┘
                         │
                         ↓
          ╔════════════════════════════════╗
          ║   訓練迴圈 (200k 步)            ║
          ╠════════════════════════════════╣
          ║ 1. Agent 產生動作 (2D 權重)    ║
          ║ 2. 環境執行 step()             ║
          ║    ├─ 調用 BaseStationStep     ║
          ║    ├─ 計算獎勵                ║
          ║    └─ 回傳觀察、獎勵           ║
          ║ 3. Replay Buffer 儲存轉移      ║
          ║ 4. SAC 訓練更新 (梯度下降)    ║
          ║ 5. 每 eval_freq 步執行評估     ║
          ║    └─ 若平均獎勵↑ → 保存模型  ║
          ╚════════════┬═══════════════════╝
                       │
                       ↓
            ┌──────────────────────┐
            │  訓練完成            │
            │  保存 final_model    │
            └──────────┬───────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ↓                           ↓
    ┌─────────────┐        ┌──────────────────┐
    │ evaluate()  │        │ tune_optuna()    │
    └─────────────┘        └──────────────────┘
         │                           │
         ↓                           ↓
    計算 KPIs                  搜索最佳超參數
    生成論文圖表              保存 best_config

```

---

## 10. 單個 TTI 內的完整執行序列

```
時刻 t (TTI開始)
  │
  ├─ Agent.predict(obs_t) → action_t
  │         │
  │         └─ Neural Network 前向傳播
  │             └─ 輸出：[w_embb=0.65, w_urllc=0.35]
  │
  ├─ Environment.step(action_t)
  │   │
  │   ├─ 1. Channel.step()
  │   │   └─ CQI_eMBB: 12 → 13
  │   │
  │   ├─ 2. Traffic.step()
  │   │   ├─ eMBB: Poisson(lambda=300) = 5 packets
  │   │   └─ URLLC: Poisson(lambda=15) = 2 packets
  │   │
  │   ├─ 3. Buffer.add_packets()
  │   │   ├─ eMBB: + 5 packets (7500B)
  │   │   └─ URLLC: + 2 packets (64B)
  │   │
  │   ├─ 4. BaseStation.step(action, cqis)
  │   │   │
  │   │   ├─ A. normalize(action) 
  │   │   │     → w_embb=0.65, w_urllc=0.35
  │   │   │
  │   │   ├─ B. RB 分配
  │   │   │     num_rbs_embb = 273 × 0.65 = 177
  │   │   │     num_rbs_urllc = 273 × 0.35 = 96
  │   │   │
  │   │   ├─ C. 容量計算 (物理層公式)
  │   │   │     cap_embb = 177 × 168 × 4.76 × 0.86 = 123k bits
  │   │   │     cap_urllc = 96 × 168 × 3.15 × 0.86 = 44k bits
  │   │   │
  │   │   ├─ D. 封包傳輸 (FIFO)
  │   │   │     eMBB 傳出：9 packets (13.5k bits)
  │   │   │     URLLC 傳出：2 packets (64 bits)
  │   │   │
  │   │   ├─ E. 延遲計算
  │   │   │     latency_embb = mean([0.2, 0.3, 0.4, ...]) = 0.35ms
  │   │   │     latency_urllc = mean([0.1, 0.2]) = 0.15ms ✅
  │   │   │
  │   │   ├─ F. URLLC 超時檢查
  │   │   │     oldest_packet.age = 0.3ms < 1.0ms ✅
  │   │   │     dropped = 0
  │   │   │
  │   │   └─ G. 統計彙整
  │   │       throughput_embb = 123000 / 500us = 246 Mbps
  │   │       throughput_urllc = 44000 / 500us = 88 Mbps
  │   │
  │   ├─ 5. Reward 計算
  │   │     total_thr = 246 + 88 = 334 Mbps
  │   │     reward = 0.1 × 334 - 10.0 × 0.15 - 50 × 0
  │   │             = 33.4 - 1.5 - 0 = 31.9 ✅
  │   │
  │   └─ 6. Observation 構建
  │       obs_next = [
  │           buffer_load_embb = 0.25,
  │           hol_embb = 0.35,
  │           buffer_load_urllc = 0.05,
  │           hol_urllc = 0.15,
  │           cqi_embb = 13,
  │           cqi_urllc = 10
  │       ]
  │
  ├─ Replay Buffer 儲存轉移
  │   buffer.add(obs_t, action_t, reward_t, obs_next, done)
  │
  ├─ SAC 訓練 (若 buffer 充足)
  │   │
  │   ├─ 從 Replay Buffer 採樣 mini-batch
  │   ├─ Actor 損失：最大化 Q - entropy
  │   ├─ Critic 損失：最小化 (Q_target - Q_pred)²
  │   ├─ Alpha 損失：熵調節
  │   └─ 梯度更新：θ ← θ - α∇L
  │
  └─ t ← t + 0.5ms (下一個 TTI)

```

---

## 11. 超參數調優的並行流程

```
主進程 (Master)
│
├─ Thread 1 (Trial 0)          ├─ Thread 2 (Trial 1)          ├─ Thread 3 (Trial 2)
│   │                           │                               │
│   ├─ lr=1e-4, gamma=0.92      ├─ lr=5e-4, gamma=0.95       ├─ lr=3e-4, gamma=0.98
│   ├─ train 200k steps         ├─ train 200k steps           ├─ train 200k steps
│   │                           │                               │
│   ├─ final reward: -1200      ├─ final reward: -8000 ❌      ├─ final reward: -500 ✅
│   │ (pruned at step 40k)      │ (pruned at step 60k)         │ (completed)
│   └─ return -1200             └─ return -8000                └─ return -500
│
└─ Study.optimize() 彙總結果
   │
   ├─ Trial 0: value=-1200
   ├─ Trial 1: value=-8000 (pruned)
   ├─ Trial 2: value=-500 (best ✅)
   │
   └─ Best Params:
      lr=3e-4, gamma=0.98, batch_size=256, buffer_size=100k
      
```

---

這份 Pseudocode 涵蓋了系統從訓練、評估到超參數調優的完整流程，可作為實現或論文說明的參考。
