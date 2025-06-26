from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner
import os
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json

if __name__ == '__main__':

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    base_dir = 'autodl-tmp/Mahjong/training_runs'
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
        'replay_buffer_size': 30000, 
        'replay_buffer_episode': 200, 
        'model_pool_size': 20,
        'model_pool_name': 'model-pool',  
        'num_actors': 24,
        'episodes_per_actor': 200000,   # 增加训练轮次
        'gamma': 0.99,                 # 稍微增加折扣因子
        'lambda': 0.95,
        'min_sample': 200,             # 增加最小采样数
        'batch_size': 256,             # 增加批次大小
        'epochs': 2,
        'clip': 0.1,
        'lr': 2e-5,                    # 调整学习率
        'lr_decay': 0.99,              # 添加学习率衰减
        'value_coeff': 1,            # 调整价值函数权重
        'entropy_coeff': 0.01,
        'device': 'cuda',
        
        'ckpt_save_interval': 300,
        'ckpt_save_path': model_dir,  
        'save_every_n_iterations': 1200,  # 增加保存间隔
        'save_state_dict_every': 3000,
        'log_dir': log_dir,  
        'training_dir': current_training_dir,  
        'timestamp': timestamp,  

        'max_learner_iterations': 10000000,  # 增加最大迭代次数
        
        'batch_norm': True,
        'advantage_norm': True,
        
        # LSTM相关配置
        'lstm_hidden_size': 256,
        'lstm_layers': 2,
        'dropout': 0.1,
        'max_grad_norm': 0.3,
        'weight_decay': 5e-4,        
    }

    print(f"Training configuration:")
    print(f"- Training directory: {config['training_dir']}")
    print(f"- Model save directory: {config['ckpt_save_path']}")
    print(f"- Log directory: {config['log_dir']}")
    print(f"- Model pool name: {config['model_pool_name']} (保持不变)")
    print(f"- Timestamp: {config['timestamp']}")
    print(f"- Total episodes: {config['num_actors'] * config['episodes_per_actor']}")
    print(f"- Device: {config['device']}")
    print(f"- Learning rate: {config['lr']}")
    print(f"- Batch size: {config['batch_size']}")
    print(f"- Buffer size: {config['replay_buffer_size']}")
    
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
    
    try:
        learner.start()
        time.sleep(2) 

        print("Learner started, training progress will be shown by Learner process...")

        for actor in actors: 
            actor.start()
        
        print("\nMain process is waiting for all Actors to complete their episodes...")
        print(f"Total episodes to complete: {config['num_actors'] * config['episodes_per_actor']}")
        print("Training progress will be shown by Learner...")
        
        for actor in actors: 
            actor.join()
        
        print("\nAll Actors have completed.")

        print("Terminating Learner process...")
        if learner.is_alive():
            learner.terminate() 
        
    except KeyboardInterrupt:
        print("Training interrupted by user. Cleaning up...")
    
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

        print("Training stopped and all processes cleaned up.")
    
    print("\n" + "="*60)
    print("Training finished!")
    print(f"All results saved in: {current_training_dir}")
    print(f"To view the training logs, run:")
    print(f"tensorboard --logdir={log_dir}")
    print(f"Best model location: {os.path.join(model_dir, 'best_model.pt')}")
    print("="*60)