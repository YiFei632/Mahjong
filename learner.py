from multiprocessing import Process
import time
import os
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from model import CNNLSTMModel

class Learner(Process):
    
    def __init__(self, config, replay_buffer):
        super(Learner, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.model_pool = None
    
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
        model_pool.push(model.state_dict())
        model = model.to(device)
        
        # training
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.config.get('lr', 5e-5),
            weight_decay=self.config.get('weight_decay', 5e-4),
            eps=1e-8
        )
        
        # 学习率调度器
        lr_decay = self.config.get('lr_decay', 0.99)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
        
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
        e_per_actor = self.config.get('episodes_per_actor', float('inf'))
        num_actor = self.config.get('num_actors', float('inf'))
        iter = e_per_actor + num_actor * 80
        
        # 额外的统计指标
        running_losses = {'policy': [], 'value': [], 'entropy': [], 'total': []}
        gradient_norms = []
        sample_efficiency_log = []
        
        log_progress(f"Starting training in {self.config['training_dir']}")
        log_progress(f"Learner will run for a maximum of {max_iterations} iterations")
        

        progress_bar = tqdm(
            total=iter,
            desc="Learner Training",
            unit="iter",
            ncols=120,
            position=0,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Loss: {postfix}"
        )
        
        try:
            while iterations < max_iterations:
                if self.replay_buffer.size() < self.config['min_sample']:
                    time.sleep(1)
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
                model.train(True)
                old_logits, _, _ = model(states)
                old_probs = F.softmax(old_logits, dim=1).gather(1, actions)
                old_log_probs = torch.log(old_probs + 1e-8).detach()
                
                epoch_losses = {'policy': [], 'value': [], 'entropy': [], 'total': []}
                
                for epoch in range(self.config['epochs']):
                    logits, values, _ = model(states)
                    action_dist = torch.distributions.Categorical(logits=logits)
                    probs = F.softmax(logits, dim=1).gather(1, actions)
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
                    
                    # 记录梯度范数
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    gradient_norms.append(grad_norm.item())
                    
                    optimizer.step()
                    
                    epoch_losses['policy'].append(policy_loss.item())
                    epoch_losses['value'].append(value_loss.item())
                    epoch_losses['entropy'].append(entropy_loss.item())
                    epoch_losses['total'].append(loss.item())
                
                # 计算平均损失
                avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
                
                # 更新运行损失历史
                for k, v in avg_losses.items():
                    running_losses[k].append(v)
                    if len(running_losses[k]) > 1000:  # 保持最近1000个值
                        running_losses[k].pop(0)
                
                # 更新进度条
                progress_bar.update(1)
                progress_bar.set_postfix_str(f"{avg_losses['total']:.4f}")
                
                # 每100次迭代输出详细信息
                if iterations % 100 == 0:
                    buffer_stats = self.replay_buffer.stats
                    sample_efficiency = buffer_stats['sample_out'] / max(1, buffer_stats['sample_in'])
                    sample_efficiency_log.append(sample_efficiency)
                    
                    tqdm.write(f'Iteration {iterations}, Average Loss: {avg_losses["total"]:.4f}, '
                              f'Buffer: {self.replay_buffer.size()}, Sample Efficiency: {sample_efficiency:.3f}')
                
                # 记录到 TensorBoard
                writer.add_scalar('Loss/total', avg_losses['total'], iterations)
                writer.add_scalar('Loss/policy', avg_losses['policy'], iterations)
                writer.add_scalar('Loss/value', avg_losses['value'], iterations)
                writer.add_scalar('Loss/entropy', avg_losses['entropy'], iterations)
                writer.add_scalar('Training/learning_rate', optimizer.param_groups[0]['lr'], iterations)
                writer.add_scalar('Training/gradient_norm', gradient_norms[-1], iterations)
                writer.add_scalar('Training/replay_buffer_size', self.replay_buffer.size(), iterations)
                
                # 额外的统计指标
                if iterations % 50 == 0:
                    # 损失趋势 (最近100次迭代的平均值)
                    recent_window = 100
                    if len(running_losses['total']) >= recent_window:
                        recent_avg_loss = np.mean(running_losses['total'][-recent_window:])
                        writer.add_scalar('Loss/recent_avg_total', recent_avg_loss, iterations)
                        writer.add_scalar('Loss/recent_avg_policy', np.mean(running_losses['policy'][-recent_window:]), iterations)
                        writer.add_scalar('Loss/recent_avg_value', np.mean(running_losses['value'][-recent_window:]), iterations)
                    
                    # 梯度统计
                    if len(gradient_norms) >= 10:
                        writer.add_scalar('Training/avg_gradient_norm', np.mean(gradient_norms[-10:]), iterations)
                        writer.add_scalar('Training/gradient_norm_std', np.std(gradient_norms[-10:]), iterations)
                    
                    # 采样效率
                    if sample_efficiency_log:
                        writer.add_scalar('Training/sample_efficiency', sample_efficiency_log[-1], iterations)
                    
                    # 策略稳定性指标
                    with torch.no_grad():
                        policy_entropy = torch.mean(action_dist.entropy()).item()
                        writer.add_scalar('Policy/avg_entropy', policy_entropy, iterations)
                        
                        # KL散度估计 (近似)
                        kl_div = torch.mean(old_log_probs - log_probs).item()
                        writer.add_scalar('Policy/kl_divergence', kl_div, iterations)
                
                # push new model
                model = model.to('cpu')
                model_pool.push(model.state_dict())
                model = model.to(device)
                
                # Save model checkpoints
                t = time.time()
                should_save_time = t - cur_time > self.config['ckpt_save_interval']
                should_save_best = avg_losses['total'] < best_loss
                should_save_periodic = iterations % self.config.get('save_every_n_iterations', 100) == 0
                
                if should_save_time or should_save_best or should_save_periodic:
                    model_cpu = model.to('cpu')
                    
                    if should_save_time or should_save_periodic:
                        checkpoint_path = os.path.join(
                            self.config['ckpt_save_path'], 
                            f'model_iter_{iterations:06d}_loss_{avg_losses["total"]:.4f}.pt'
                        )
                        torch.save({
                            'model_state_dict': model_cpu.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'iteration': iterations,
                            'loss': avg_losses['total'],
                            'losses_breakdown': avg_losses,
                            'config': self.config,
                            'timestamp': self.config['timestamp'],
                            'training_time': t - cur_time if should_save_time else None,
                            'gradient_norm': gradient_norms[-1] if gradient_norms else 0
                        }, checkpoint_path)
                        tqdm.write(f'Saved checkpoint: {os.path.basename(checkpoint_path)}')
                    
                    if should_save_best:
                        best_loss = avg_losses['total']
                        best_path = os.path.join(self.config['ckpt_save_path'], 'best_model.pt')
                        torch.save({
                            'model_state_dict': model_cpu.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'iteration': iterations,
                            'loss': avg_losses['total'],
                            'losses_breakdown': avg_losses,
                            'config': self.config,
                            'timestamp': self.config['timestamp'],
                            'is_best': True
                        }, best_path)
                        tqdm.write(f'New best model saved: loss={avg_losses["total"]:.4f}')
                    
                    latest_path = os.path.join(self.config['ckpt_save_path'], 'latest_model.pt')
                    torch.save({
                        'model_state_dict': model_cpu.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'iteration': iterations,
                        'loss': avg_losses['total'],
                        'losses_breakdown': avg_losses,
                        'config': self.config,
                        'timestamp': self.config['timestamp']
                    }, latest_path)
                    
                    model = model_cpu.to(device)
                    
                    if should_save_time:
                        cur_time = t
                
                iterations += 1
                
                # 学习率衰减
                if iterations % 1000 == 0:
                    scheduler.step()
                    writer.add_scalar('Training/learning_rate_decay', optimizer.param_groups[0]['lr'], iterations)
                
                # Optional: Save model state dict only
                if iterations % self.config.get('save_state_dict_every', 500) == 0:
                    model_cpu = model.to('cpu')
                    state_dict_path = os.path.join(self.config['ckpt_save_path'], f'state_dict_iter_{iterations:06d}.pt')
                    torch.save(model_cpu.state_dict(), state_dict_path)
                    model = model_cpu.to(device)
                    tqdm.write(f'Saved state dict: {os.path.basename(state_dict_path)}')
        
        finally:
            # 关闭进度条
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
            'scheduler_state_dict': scheduler.state_dict(),
            'iteration': iterations,
            'loss': avg_losses['total'],
            'losses_breakdown': avg_losses,
            'best_loss': best_loss,
            'config': self.config,
            'timestamp': self.config['timestamp'],
            'training_completed': True,
            'final_stats': {
                'total_iterations': iterations,
                'avg_gradient_norm': np.mean(gradient_norms[-100:]) if len(gradient_norms) >= 100 else 0,
                'final_sample_efficiency': sample_efficiency_log[-1] if sample_efficiency_log else 0
            }
        }, final_model_path)
        tqdm.write(f"Final model saved: {os.path.basename(final_model_path)}")
        
        log_progress("Closing TensorBoard writer")
        writer.close()

    def terminate(self):
        print("Learner process terminating...")
        if self.model_pool:
            self.model_pool.cleanup()
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