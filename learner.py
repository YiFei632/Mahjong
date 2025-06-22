from multiprocessing import Process
import time
import os
import numpy as np
import torch
from torch.nn import functional as F
# --- 新增导入 ---
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from model import CNNModel

class Learner(Process):
    
    def __init__(self, config, replay_buffer):
        super(Learner, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.model_pool = None # 将 model_pool 提升为成员变量
    
    def run(self):
        # --- 新增: 初始化 TensorBoard Writer ---
        writer = SummaryWriter(log_dir=self.config['log_dir'])
        # create model pool
        model_pool = ModelPoolServer(self.config['model_pool_size'], self.config['model_pool_name'])
        
        # initialize model params
        device = torch.device(self.config['device'])
        model = CNNModel()
        
        # send to model pool
        model_pool.push(model.state_dict()) # push cpu-only tensor to model_pool
        model = model.to(device)
        
        # training
        optimizer = torch.optim.Adam(model.parameters(), lr = self.config['lr'])
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.config['ckpt_save_path'], exist_ok=True)
        
        # wait for initial samples
        while self.replay_buffer.size() < self.config['min_sample']:
            time.sleep(0.1)
        
        cur_time = time.time()
        iterations = 0
        best_loss = float('inf')

        max_iterations = self.config.get('max_learner_iterations', float('inf'))
        
        print(f"Starting training... Saving checkpoints to {self.config['ckpt_save_path']}")
        print(f"Learner will run for a maximum of {max_iterations} iterations.")
        
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
            
            print('Iteration %d, replay buffer in %d out %d' % (iterations, self.replay_buffer.stats['sample_in'], self.replay_buffer.stats['sample_out']))
            
            # calculate PPO loss
            model.train(True) # Batch Norm training mode
            old_logits, _ = model(states)
            old_probs = F.softmax(old_logits, dim = 1).gather(1, actions)
            old_log_probs = torch.log(old_probs + 1e-8).detach()
            
            epoch_losses = []
            for epoch in range(self.config['epochs']):
                logits, values = model(states)
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
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            print(f'Iteration {iterations}, Average Loss: {avg_loss:.4f}')

            # --- 新增: 将损失写入 TensorBoard ---
            writer.add_scalar('Loss/total', avg_loss, iterations)
            writer.add_scalar('Loss/policy', policy_loss.item(), iterations)
            writer.add_scalar('Loss/value', value_loss.item(), iterations)
            writer.add_scalar('Loss/entropy', entropy_loss.item(), iterations)
            writer.add_scalar('Misc/learning_rate', optimizer.param_groups[0]['lr'], iterations)
            writer.add_scalar('Misc/replay_buffer_size', self.replay_buffer.size(), iterations)
            # --- 新增结束 ---

            # push new model
            model = model.to('cpu')
            model_pool.push(model.state_dict()) # push cpu-only tensor to model_pool
            model = model.to(device)
            
            # Save model checkpoints
            t = time.time()
            should_save_time = t - cur_time > self.config['ckpt_save_interval']
            should_save_best = avg_loss < best_loss
            should_save_periodic = iterations % self.config.get('save_every_n_iterations', 100) == 0
            
            if should_save_time or should_save_best or should_save_periodic:
                # Move model to CPU for saving
                model_cpu = model.to('cpu')
                
                # Save current model
                if should_save_time or should_save_periodic:
                    checkpoint_path = os.path.join(self.config['ckpt_save_path'], f'model_{iterations}.pt')
                    torch.save({
                        'model_state_dict': model_cpu.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'iteration': iterations,
                        'loss': avg_loss,
                        'config': self.config
                    }, checkpoint_path)
                    print(f'Saved checkpoint: {checkpoint_path}')
                
                # Save best model
                if should_save_best:
                    best_loss = avg_loss
                    best_path = os.path.join(self.config['ckpt_save_path'], 'best_model.pt')
                    torch.save({
                        'model_state_dict': model_cpu.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'iteration': iterations,
                        'loss': avg_loss,
                        'config': self.config
                    }, best_path)
                    print(f'Saved best model: {best_path} (loss: {avg_loss:.4f})')
                
                # Save latest model (for easy loading)
                latest_path = os.path.join(self.config['ckpt_save_path'], 'latest_model.pt')
                torch.save({
                    'model_state_dict': model_cpu.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iteration': iterations,
                    'loss': avg_loss,
                    'config': self.config
                }, latest_path)
                
                # Move model back to training device
                model = model_cpu.to(device)
                
                if should_save_time:
                    cur_time = t
            
            iterations += 1
            
            # Optional: Save model state dict only (for compatibility with __main__.py)
            if iterations % self.config.get('save_state_dict_every', 500) == 0:
                model_cpu = model.to('cpu')
                state_dict_path = os.path.join(self.config['ckpt_save_path'], f'testrl_{iterations}.pt')
                torch.save(model_cpu.state_dict(), state_dict_path)
                model = model_cpu.to(device)
                print(f'Saved state dict: {state_dict_path}')
        
        print(f"Learner reached max iterations ({iterations})... Stopping.")
        # --- 新增: 关闭 writer ---
        print("Learner finished training. Closing TensorBoard writer.")
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