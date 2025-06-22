from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner
import os
# 增加 time 模块导入，以便在启动 Actors 前稍作等待
import time
# --- 新增导入 ---
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


if __name__ == '__main__':
    # Create model directory if it doesn't exist
    model_dir = './models'  # You can change this path
    os.makedirs(model_dir, exist_ok=True)
    
    config = {
        'replay_buffer_size': 50000,
        'replay_buffer_episode': 400,
        'model_pool_size': 20,
        'model_pool_name': 'model-pool',
        'num_actors': 24,
        'episodes_per_actor': 1000, # 每个 Actor 运行100个episode后会退出
        'gamma': 0.98,
        'lambda': 0.95,
        'min_sample': 200,
        'batch_size': 256,
        'epochs': 5,
        'clip': 0.2,
        'lr': 1e-4,
        'value_coeff': 1,
        'entropy_coeff': 0.01,
        'device': 'cuda',
        
        'ckpt_save_interval': 300,
        'ckpt_save_path': model_dir,
        'save_every_n_iterations': 100,
        'save_state_dict_every': 500,

        'max_learner_iterations': 20000, # 新增: 设置一个合理的迭代上限
    }

    # --- 新增: 创建 TensorBoard 日志目录 ---
    # 使用当前时间创建一个唯一的目录名，例如 "logs/run-20250622-003055"
    log_dir = f"./logs/run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    config['log_dir'] = log_dir  # 将日志目录路径添加到配置中
    print(f"TensorBoard log directory: {log_dir}")
    # --- 新增结束 ---
    
    print(f"Training configuration:")
    print(f"- Model save directory: {config['ckpt_save_path']}")
    print(f"- Save interval: {config['ckpt_save_interval']} seconds")
    print(f"- Save every: {config['save_every_n_iterations']} iterations")
    print(f"- Device: {config['device']}")
    print(f"- Number of actors: {config['num_actors']}")
    
    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])
    
    # Learner 需要先于 Actors 创建，以确保 model_pool 先存在
    learner = Learner(config, replay_buffer)
    
    actors = []
    for i in range(config['num_actors']):
        config['name'] = 'Actor-%d' % i
        actor = Actor(config, replay_buffer)
        actors.append(actor)
    
    # --- 这是修改的核心部分 ---
    try:
        learner.start()
        # 短暂等待 Learner 初始化 ModelPoolServer，避免 Actors 启动时连接失败
        time.sleep(2) 

        for actor in actors: 
            actor.start()
        
        # 1. 等待所有 Actor 进程完成它们有限的 episodes
        print("Main process is waiting for all Actors to complete their episodes...")
        for actor in actors: 
            actor.join()
        print("All Actors have completed.")

        # 2. Actors 全部结束后，我们知道数据采集已完成，可以终止 Learner
        print("Terminating Learner process...")
        if learner.is_alive():
            learner.terminate() # 这会安全地触发 Learner 的清理逻辑
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Cleaning up...")
    
    finally:
        # 3. 确保所有进程都被终止和清理
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

        print("Training stopped and all processes cleaned up.")
    
    print("\n" + "="*50)
    print("Training finished!")
    print(f"To view the training logs, run the following command in your terminal:")
    print(f"tensorboard --logdir={log_dir}")
    print("="*50)