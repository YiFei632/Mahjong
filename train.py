from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner
import os
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json
from tqdm import tqdm
import threading
import re

if __name__ == '__main__':

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    base_dir = './training_runs'
    os.makedirs(base_dir, exist_ok=True)
    
    # 为本次训练创建专用目录
    current_training_dir = os.path.join(base_dir, f'run_{timestamp}')
    os.makedirs(current_training_dir, exist_ok=True)
    
    model_dir = os.path.join(current_training_dir, 'models')
    log_dir = os.path.join(current_training_dir, 'logs')
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"开始新的训练任务")
    print(f"训练目录: {current_training_dir}")
    print(f"模型保存: {model_dir}")
    print(f"日志保存: {log_dir}")
    print("-" * 50)

    config = {
        'replay_buffer_size': 50000,
        'replay_buffer_episode': 400,
        'model_pool_size': 20,
        'model_pool_name': 'model-pool',  
        'num_actors': 24,
        'episodes_per_actor': 1000,
        'gamma': 0.98,
        'lambda': 0.95,
        'min_sample': 200,
        'batch_size': 256,
        'epochs': 5,
        'clip': 0.2,
        'lr': 5e-5,
        'value_coeff': 1,
        'entropy_coeff': 0.01,
        'device': 'cuda',
        
        'ckpt_save_interval': 300,
        'ckpt_save_path': model_dir,  
        'save_every_n_iterations': 100,
        'save_state_dict_every': 500,
        'log_dir': log_dir,  
        'training_dir': current_training_dir,  
        'timestamp': timestamp,  

        'max_learner_iterations': 20000,
        
        # LSTM相关配置
        'lstm_hidden_size': 256,
        'lstm_layers': 2,
        'dropout': 0.1,
        'max_grad_norm': 1.0,
        'weight_decay': 1e-5,        
    }

    print(f"Training configuration:")
    print(f"- Training directory: {config['training_dir']}")
    print(f"- Model save directory: {config['ckpt_save_path']}")
    print(f"- Log directory: {config['log_dir']}")
    print(f"- Model pool name: {config['model_pool_name']} (保持不变)")
    print(f"- Timestamp: {config['timestamp']}")
    print(f"- Save interval: {config['ckpt_save_interval']} seconds")
    print(f"- Save every: {config['save_every_n_iterations']} iterations")
    print(f"- Device: {config['device']}")
    print(f"- Number of actors: {config['num_actors']}")
    print(f"- Max iterations: {config['max_learner_iterations']}")
    
    config_file = os.path.join(current_training_dir, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"- Config saved to: {config_file}")

    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])
    
    learner = Learner(config, replay_buffer)
    
    actors = []
    for i in range(config['num_actors']):
        config['name'] = 'Actor-%d' % i
        actor = Actor(config, replay_buffer)
        actors.append(actor)
    
    # 初始化进度条
    progress_bar = None
    monitor_thread = None
    
    try:
        learner.start()
        time.sleep(2) 

        # 创建进度条
        progress_bar = tqdm(
            total=config['max_learner_iterations'], 
            desc="Training Progress", 
            unit="iter",
            ncols=120,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Loss: {postfix}"
        )
        
        # 创建进度监控线程
        def monitor_progress():
            last_iteration = 0
            last_loss = 0.0
            
            while learner.is_alive():
                try:
                    # 方法1: 读取标准输出日志（如果有的话）
                    # 方法2: 读取训练进度文件
                    progress_file = os.path.join(current_training_dir, 'training_progress.txt')
                    
                    if os.path.exists(progress_file):
                        with open(progress_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            
                            # 查找最新的迭代信息
                            for line in reversed(lines[-20:]):  # 检查最后20行
                                # 查找形如 "Iteration X, Average Loss: Y.YYYY" 的行
                                iteration_match = re.search(r'Iteration (\d+).*Average Loss:\s*([\d.]+)', line)
                                if iteration_match:
                                    try:
                                        iteration = int(iteration_match.group(1))
                                        loss = float(iteration_match.group(2))
                                        
                                        if iteration > last_iteration:
                                            # 更新进度条
                                            progress_increment = iteration - last_iteration
                                            progress_bar.update(progress_increment)
                                            progress_bar.set_postfix_str(f"{loss:.4f}")
                                            last_iteration = iteration
                                            last_loss = loss
                                        break
                                    except (ValueError, IndexError):
                                        continue
                    
                    # 如果没有进度文件，尝试从learner的输出中获取信息
                    # 这里可以添加其他监控方式
                    
                except Exception as e:
                    # 静默处理异常，避免影响主训练流程
                    pass
                
                time.sleep(3)  # 每3秒检查一次
            
            # 训练结束时完成进度条
            if progress_bar and not progress_bar.disable:
                progress_bar.n = progress_bar.total
                progress_bar.set_postfix_str(f"Final: {last_loss:.4f}")
                progress_bar.refresh()

        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()

        for actor in actors: 
            actor.start()
        
        print("\nMain process is waiting for all Actors to complete their episodes...")
        print("training progress will be shown above...")
        
        for actor in actors: 
            actor.join()
        
        print("\nAll Actors have completed.")

        print("Terminating Learner process...")
        if learner.is_alive():
            learner.terminate() 
        
    except KeyboardInterrupt:
        print("Training interrupted by user. Cleaning up...")
        if progress_bar:
            progress_bar.set_description("Interrupted")
    
    finally:
        print("Final cleanup: Terminating all processes...")
        
        if learner.is_alive():
            learner.terminate()
        
        for actor in actors:
            if actor.is_alive():
                actor.terminate()
        
        # 等待进程完全退出
        learner.join(timeout=5)
        for actor in actors:
            actor.join(timeout=5)


        if progress_bar:
            progress_bar.close()

        print("Training stopped and all processes cleaned up.")
    
    print("\n" + "="*60)
    print("Training finished!")
    print(f"All results saved in: {current_training_dir}")
    print(f"To view the training logs, run:")
    print(f"tensorboard --logdir={log_dir}")
    print(f"Best model location: {os.path.join(model_dir, 'best_model.pt')}")
    print("="*60)