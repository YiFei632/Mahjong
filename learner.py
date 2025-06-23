from multiprocessing import Process
import time
import os
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm  # 新增

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from model import CNNLSTMModel

class Learner(Process):
    
    def __init__(self, config, replay_buffer):
        super(Learner, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.model_pool = None # 将 model_pool 提升为成员变量
    
    def run(self):
        # --- 使用配置中的日志目录 ---
        writer = SummaryWriter(log_dir=self.config['log_dir'])
        
        # 确保模型保存目录存在
        os.makedirs(self.config['ckpt_save_path'], exist_ok=True)
        
        # create model pool
        model_pool = ModelPoolServer(self.config['model_pool_size'], self.config['model_pool_name'])
        
        # initialize model params
        device = torch.device(self.config['device'])
        model = CNNLSTMModel(
            lstm_hidden_size=self.config.get('lstm_hidden_size', 256),
            lstm_layers=self.config.get('lstm_layers', 2),
            dropout=self.config.get('dropout', 0.1)
        )
        
        # send to model pool
        model_pool.push(model.state_dict()) # push cpu-only tensor to model_pool
        model = model.to(device)
        
        # training
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.config.get('lr', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        max_grad_norm = self.config.get('max_grad_norm', 1.0)
        
        progress_file = os.path.join(self.config['training_dir'], 'training_progress.txt')
        
        def log_progress(message):
            """记录训练进度到文件"""
            with open(progress_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] {message}\n")
            print(message)
        
        # wait for initial samples
        while self.replay_buffer.size() < self.config['min_sample']:
            time.sleep(0.1)
        
        cur_time = time.time()
        iterations = 0
        best_loss = float('inf')
        max_iterations = self.config.get('max_learner_iterations', float('inf'))
        
        log_progress(f"Starting training in {self.config['training_dir']}")
        log_progress(f"Learner will run for a maximum of {max_iterations} iterations")
        

        progress_bar = tqdm(
            total=max_iterations,
            desc="Training",
            unit="iter",
            ncols=100,
            position=0,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Loss: {postfix}"
        )
        
        try:
            while iterations < max_iterations:
                if self.replay_buffer.size() < self.config['min_sample']:
                    time.sleep(1) # 等待 Actors 产生更多数据
                    continue
                
                # sample batch
                batch = self.replay_buffer.sample(self.config['batch_size'])
                obs = torch.tensor(batch['state']['observation']).to(device)
                mask = torch.tensor(batch['state']['action_mask']).to(device)
                states = {
                    'observation': obs,
                    'action_mask': mask
                }
                actions = torch.tensor(batch['action']).unsqueeze(-1).to(device)
                advs = torch.tensor(batch['adv']).to(device)
                targets = torch.tensor(batch['target']).to(device)
                

                
                # calculate PPO loss
                model.train(True) # Batch Norm training mode
                old_logits, _, _ = model(states)
                old_probs = F.softmax(old_logits, dim = 1).gather(1, actions)
                old_log_probs = torch.log(old_probs + 1e-8).detach()
                
                epoch_losses = []
                for epoch in range(self.config['epochs']):
                    logits, values, _ = model(states)  
                    action_dist = torch.distributions.Categorical(logits = logits)
                    probs = F.softmax(logits, dim = 1).gather(1, actions)
                    log_probs = torch.log(probs + 1e-8)
                    ratio = torch.exp(log_probs - old_log_probs)
                    surr1 = ratio * advs
                    surr2 = torch.clamp(ratio, 1 - self.config['clip'], 1 + self.config['clip']) * advs
                    policy_loss = -torch.mean(torch.min(surr1, surr2))
                    value_loss = torch.mean(F.mse_loss(values.squeeze(-1), targets))
                    entropy_loss = -torch.mean(action_dist.entropy())
                    loss = policy_loss + self.config['value_coeff'] * value_loss + self.config['entropy_coeff'] * entropy_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                
                avg_loss = np.mean(epoch_losses)
                

                progress_bar.update(1)
                progress_bar.set_postfix_str(f"{avg_loss:.4f}")
                
                # 每100次迭代输出详细信息
                if iterations % 100 == 0:
                    tqdm.write(f'Iteration {iterations}, Avg Loss: {avg_loss:.4f}, Buffer: {self.replay_buffer.stats["sample_in"]}/{self.replay_buffer.stats["sample_out"]}')

                # 将损失写入 TensorBoard
                writer.add_scalar('Loss/total', avg_loss, iterations)
                writer.add_scalar('Loss/policy', policy_loss.item(), iterations)
                writer.add_scalar('Loss/value', value_loss.item(), iterations)
                writer.add_scalar('Loss/entropy', entropy_loss.item(), iterations)
                writer.add_scalar('Misc/learning_rate', optimizer.param_groups[0]['lr'], iterations)
                writer.add_scalar('Misc/replay_buffer_size', self.replay_buffer.size(), iterations)

                # push new model
                model = model.to('cpu')
                model_pool.push(model.state_dict())
                model = model.to(device)
                
                # Save model checkpoints
                t = time.time()
                should_save_time = t - cur_time > self.config['ckpt_save_interval']
                should_save_best = avg_loss < best_loss
                should_save_periodic = iterations % self.config.get('save_every_n_iterations', 100) == 0
                
                if should_save_time or should_save_best or should_save_periodic:
                    model_cpu = model.to('cpu')
                    
                    if should_save_time or should_save_periodic:
                        checkpoint_path = os.path.join(
                            self.config['ckpt_save_path'], 
                            f'model_iter_{iterations:06d}_loss_{avg_loss:.4f}.pt'
                        )
                        torch.save({
                            'model_state_dict': model_cpu.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'iteration': iterations,
                            'loss': avg_loss,
                            'config': self.config,
                            'timestamp': self.config['timestamp'],
                            'training_time': t - cur_time if should_save_time else None
                        }, checkpoint_path)
                        tqdm.write(f'Saved checkpoint: {os.path.basename(checkpoint_path)}')
                    
                    if should_save_best:
                        best_loss = avg_loss
                        best_path = os.path.join(self.config['ckpt_save_path'], 'best_model.pt')
                        torch.save({
                            'model_state_dict': model_cpu.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'iteration': iterations,
                            'loss': avg_loss,
                            'config': self.config,
                            'timestamp': self.config['timestamp'],
                            'is_best': True
                        }, best_path)
                        tqdm.write(f'New best model saved: loss={avg_loss:.4f}')
                    
                    latest_path = os.path.join(self.config['ckpt_save_path'], 'latest_model.pt')
                    torch.save({
                        'model_state_dict': model_cpu.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'iteration': iterations,
                        'loss': avg_loss,
                        'config': self.config,
                        'timestamp': self.config['timestamp']
                    }, latest_path)
                    
                    model = model_cpu.to(device)
                    
                    if should_save_time:
                        cur_time = t
                
                iterations += 1
                
                # Optional: Save model state dict only
                if iterations % self.config.get('save_state_dict_every', 500) == 0:
                    model_cpu = model.to('cpu')
                    state_dict_path = os.path.join(self.config['ckpt_save_path'], f'state_dict_iter_{iterations:06d}.pt')
                    torch.save(model_cpu.state_dict(), state_dict_path)
                    model = model_cpu.to(device)
                    tqdm.write(f'Saved state dict: {os.path.basename(state_dict_path)}')
        
        finally:

            progress_bar.close()
        
        # 训练完成
        tqdm.write(f"Training completed after {iterations} iterations")
        tqdm.write(f"Best loss achieved: {best_loss:.4f}")
        
        # 保存最终模型
        final_model_path = os.path.join(self.config['ckpt_save_path'], 'final_model.pt')
        model_cpu = model.to('cpu')
        torch.save({
            'model_state_dict': model_cpu.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': iterations,
            'loss': avg_loss,
            'best_loss': best_loss,
            'config': self.config,
            'timestamp': self.config['timestamp'],
            'training_completed': True
        }, final_model_path)
        tqdm.write(f"Final model saved: {os.path.basename(final_model_path)}")
        
        log_progress("Closing TensorBoard writer")
        writer.close()

    def terminate(self):
        print("Learner process terminating...")
        if self.model_pool:
            self.model_pool.shutdown()
        super().terminate()
    
    def save_final_model(self, model, optimizer, iterations, loss):
        """Save final model when training is complete"""
        model_cpu = model.to('cpu')
        final_path = os.path.join(self.config['ckpt_save_path'], 'final_model.pt')
        torch.save({
            'model_state_dict': model_cpu.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': iterations,
            'loss': loss,
            'config': self.config
        }, final_path)
        print(f'Saved final model: {final_path}')
        
        # Also save just the state dict for compatibility
        state_dict_path = os.path.join(self.config['ckpt_save_path'], 'testrl.pt')
        torch.save(model_cpu.state_dict(), state_dict_path)
        print(f'Saved final state dict: {state_dict_path}')