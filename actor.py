from multiprocessing import Process
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time
from datetime import datetime

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolClient
from env import MahjongGBEnv
from feature import FeatureAgent
from model import CNNLSTMModel

class Actor(Process):
    
    def __init__(self, config, replay_buffer):
        super(Actor, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.name = config.get('name', 'Actor-?')
        self.episode_count = 0
        
    def run(self):
        # 初始化 TensorBoard Writer
        writer = SummaryWriter(log_dir=self.config['log_dir'])
        torch.set_num_threads(1)
    
        # connect to model pool
        model_pool = ModelPoolClient(self.config['model_pool_name'])
        
        # create network model
        model = CNNLSTMModel(
            lstm_hidden_size=self.config.get('lstm_hidden_size', 256),
            lstm_layers=self.config.get('lstm_layers', 2),
            dropout=self.config.get('dropout', 0.1)
        )
        
        # load initial model
        version = model_pool.get_latest_model()
        state_dict = model_pool.load_model(version)
        
        # 添加空值检查
        if state_dict is None:
            print(f"{self.name} failed to load initial model, retrying...")
            import time
            time.sleep(2)
            version = model_pool.get_latest_model()
            state_dict = model_pool.load_model(version)
            if state_dict is None:
                print(f"{self.name} still failed to load model, exiting")
                return
        
        model.load_state_dict(state_dict)
        
        # collect data
        env = MahjongGBEnv(config = {'agent_clz': FeatureAgent})
        policies = {player : model for player in env.agent_names}
        
        reset_failures = 0
        max_reset_failures = 5
        
        # 统计指标
        win_count = 0
        total_reward = 0
        episode_lengths = []
        action_counts = {'Pass': 0, 'Play': 0, 'Chi': 0, 'Peng': 0, 'Gang': 0, 'Hu': 0}
        
        episode_progress_file = os.path.join(self.config['training_dir'], 'episode_progress.txt')
        
        for episode in range(self.config['episodes_per_actor']):
            try:
                # update model
                latest = model_pool.get_latest_model()
                if latest is not None and latest['id'] > version['id']:
                    new_state_dict = model_pool.load_model(latest)
                    if new_state_dict is not None:  # 只在成功加载时更新
                        model.load_state_dict(new_state_dict)
                        version = latest
                
                # run one episode and collect data
                obs = None
                episode_reset_attempts = 0
                max_reset_attempts = 5
                
                while episode_reset_attempts < max_reset_attempts:
                    try:
                        obs = env.reset()
                        if obs and len(obs) > 0:
                            reset_failures = 0  # 重置失败计数
                            break
                        else:
                            print(f"{self.name} env.reset() returned empty obs, attempt {episode_reset_attempts+1}")
                            episode_reset_attempts += 1
                    except Exception as e:
                        print(f"{self.name} env.reset() failed, attempt {episode_reset_attempts+1}: {e}")
                        episode_reset_attempts += 1
                        if episode_reset_attempts >= max_reset_attempts:
                            reset_failures += 1
                            if reset_failures >= max_reset_failures:
                                print(f"{self.name} too many reset failures, recreating environment")
                                # 重新创建环境
                                try:
                                    env = MahjongGBEnv(config = {'agent_clz': FeatureAgent})
                                    reset_failures = 0
                                    obs = env.reset()
                                    if obs and len(obs) > 0:
                                        break
                                except Exception as recreate_error:
                                    print(f"{self.name} failed to recreate environment: {recreate_error}")
                                    obs = None
                
                if not obs or len(obs) == 0:
                    print(f"{self.name} skipping episode {episode} - reset failed after {max_reset_attempts} attempts")
                    continue
                
                episode_data = {agent_name: {
                    'state' : {
                        'observation': [],
                        'action_mask': []
                    },
                    'action' : [],
                    'reward' : [],
                    'value' : [],
                    'action_probs': [],  # 添加动作概率记录
                    'entropy': []        # 添加策略熵记录
                } for agent_name in env.agent_names}
                
                done = False
                step_count = 0
                max_steps = 500  # 限制最大步数，防止无限循环
                episode_action_counts = {'Pass': 0, 'Play': 0, 'Chi': 0, 'Peng': 0, 'Gang': 0, 'Hu': 0}
                
                while not done and step_count < max_steps:
                    step_count += 1
                    
                    # 检查观测是否有效
                    if not obs or len(obs) == 0:
                        print(f"{self.name} empty observation at step {step_count}, ending episode")
                        break
                    # each player take action
                    actions = {}
                    values = {}
                    for agent_name in obs:  # 只为当前需要行动的玩家选择动作
                        agent_data = episode_data[agent_name]
                        state = obs[agent_name]
                        agent_data['state']['observation'].append(state['observation'])
                        agent_data['state']['action_mask'].append(state['action_mask'])
                        state['observation'] = torch.tensor(state['observation'], dtype = torch.float).unsqueeze(0)
                        state['action_mask'] = torch.tensor(state['action_mask'], dtype = torch.float).unsqueeze(0)
                        model.train(False)
                        with torch.no_grad():
                            logits, value, _ = model(state)
                            action_dist = torch.distributions.Categorical(logits = logits)
                            action = action_dist.sample().item()
                            value = value.item()
                            
                            # 记录动作概率和熵
                            action_prob = action_dist.probs[0, action].item()
                            entropy = action_dist.entropy().item()
                            
                        actions[agent_name] = action
                        values[agent_name] = value
                        agent_data['action'].append(actions[agent_name])
                        agent_data['value'].append(values[agent_name])
                        agent_data['action_probs'].append(action_prob)
                        agent_data['entropy'].append(entropy)
                        
                        # 统计动作类型（简化版本）
                        if action == 0:
                            episode_action_counts['Pass'] += 1
                        elif action == 1:
                            episode_action_counts['Hu'] += 1
                        elif action < 36:
                            episode_action_counts['Play'] += 1
                        elif action < 99:
                            episode_action_counts['Chi'] += 1
                        elif action < 133:
                            episode_action_counts['Peng'] += 1
                        else:
                            episode_action_counts['Gang'] += 1
                            
                    # interact with env
                    try:
                        next_obs, rewards, done = env.step(actions)
                        
                        # 检查环境返回是否有效
                        if next_obs is None:
                            print(f"{self.name} env.step() returned None at step {step_count}, ending episode")
                            done = True
                            # 创建默认奖励
                            rewards = {name: 0 for name in env.agent_names}
                        elif len(next_obs) == 0:
                            print(f"{self.name} env.step() returned empty obs at step {step_count}, ending episode")
                            done = True
                            # 创建默认奖励
                            rewards = {name: 0 for name in env.agent_names}
                        
                        # 确保rewards不为空
                        if not rewards:
                            rewards = {name: 0 for name in env.agent_names}
                        
                        for agent_name in rewards:
                            if agent_name in episode_data:  # 只为存在的agent添加奖励
                                episode_data[agent_name]['reward'].append(rewards[agent_name])
                        obs = next_obs
                        
                    except Exception as e:
                        print(f"{self.name} env.step() failed at step {step_count}: {e}")
                        import traceback
                        traceback.print_exc()
                        done = True
                        rewards = {name: 0 for name in env.agent_names}
                        for agent_name in env.agent_names:
                            if agent_name in episode_data:  # 只为存在的agent添加奖励
                                episode_data[agent_name]['reward'].append(0)                        
                        for agent_name in rewards:
                            if agent_name in episode_data:  # 只为存在的agent添加奖励
                                episode_data[agent_name]['reward'].append(rewards[agent_name])
                        obs = next_obs
                        
                    except Exception as e:
                        print(f"{self.name} env.step() failed at step {step_count}: {e}")
                        import traceback
                        traceback.print_exc()
                        done = True
                        rewards = {name: 0 for name in env.agent_names}
                        for agent_name in env.agent_names:
                            if agent_name in episode_data:  # 只为存在的agent添加奖励
                                episode_data[agent_name]['reward'].append(0)
                
                if step_count >= max_steps:
                    print(f"{self.name} episode {episode} reached max steps ({max_steps})")
                
                # 记录episode统计信息
                if step_count > 0 and 'rewards' in locals() and rewards:
                    self.episode_count += 1
                    episode_reward = rewards['player_1']
                    total_reward += episode_reward
                    episode_lengths.append(step_count)
                    
                    # 判断是否获胜
                    if episode_reward > 0:
                        win_count += 1
                    
                    # 更新动作统计
                    for action_type, count in episode_action_counts.items():
                        action_counts[action_type] += count
                    
                    # 记录到 TensorBoard (每100个episode记录一次)
                    if episode % 100 == 0:
                        # 基础指标
                        writer.add_scalar(f'Episode/Reward/{self.name}', episode_reward, episode)
                        writer.add_scalar(f'Episode/Length/{self.name}', step_count, episode)
                        writer.add_scalar(f'Episode/WinRate/{self.name}', win_count / max(1, self.episode_count), episode)
                        
                        # 平均指标
                        avg_reward = total_reward / max(1, self.episode_count)
                        avg_length = np.mean(episode_lengths[-100:]) if episode_lengths else 0
                        writer.add_scalar(f'Episode/AvgReward/{self.name}', avg_reward, episode)
                        writer.add_scalar(f'Episode/AvgLength/{self.name}', avg_length, episode)
                        
                        # 动作分布
                        total_actions = sum(action_counts.values())
                        if total_actions > 0:
                            for action_type, count in action_counts.items():
                                writer.add_scalar(f'Actions/{action_type}_ratio/{self.name}', 
                                                count / total_actions, episode)
                        
                        # 策略质量指标
                        if len(episode_data['player_1']['action_probs']) > 0:
                            avg_action_prob = np.mean(episode_data['player_1']['action_probs'])
                            avg_entropy = np.mean(episode_data['player_1']['entropy'])
                            writer.add_scalar(f'Policy/AvgActionProb/{self.name}', avg_action_prob, episode)
                            writer.add_scalar(f'Policy/AvgEntropy/{self.name}', avg_entropy, episode)
                    
                    # 记录episode进度 - 修正总计算逻辑
                    if episode % 500 == 0:  # 每500个episode记录一次
                        with open(episode_progress_file, 'a', encoding='utf-8') as f:
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            # 修正：只记录当前Actor的进度，不计算"总计"
                            f.write(f"[{timestamp}] {self.name} completed {self.episode_count} episodes\n")
                else:
                    print(f"{self.name} Episode {episode} failed - no valid rewards")
                
                # postprocessing episode data for each agent
                for agent_name, agent_data in episode_data.items():
                    # 添加数据完整性检查
                    if (len(agent_data['action']) == 0 or 
                        len(agent_data['state']['observation']) == 0 or
                        len(agent_data['reward']) == 0):
                        continue
                    
                    # 调整数据长度一致性 - 麻将中action和reward的对应关系可能不同
                    min_len = min(len(agent_data['action']), len(agent_data['reward']), len(agent_data['value']))
                    if min_len == 0:
                        continue
                    
                    # 截断到最小长度
                    agent_data['action'] = agent_data['action'][:min_len]
                    agent_data['reward'] = agent_data['reward'][:min_len]
                    agent_data['value'] = agent_data['value'][:min_len]
                    agent_data['state']['observation'] = agent_data['state']['observation'][:min_len]
                    agent_data['state']['action_mask'] = agent_data['state']['action_mask'][:min_len]
                    
                    try:
                        obs = np.stack(agent_data['state']['observation'])
                        mask = np.stack(agent_data['state']['action_mask'])
                        actions = np.array(agent_data['action'], dtype = np.int64)
                        rewards = np.array(agent_data['reward'], dtype = np.float32)
                        values = np.array(agent_data['value'], dtype = np.float32)
                        next_values = np.array(agent_data['value'][1:] + [0], dtype = np.float32)
                        
                        td_target = rewards + next_values * self.config['gamma']
                        td_delta = td_target - values
                        advs = []
                        adv = 0
                        for delta in td_delta[::-1]:
                            adv = self.config['gamma'] * self.config['lambda'] * adv + delta
                            advs.append(adv)
                        advs.reverse()
                        advantages = np.array(advs, dtype = np.float32)
                        
                        # send samples to replay_buffer (per agent)
                        self.replay_buffer.push({
                            'state': {
                                'observation': obs,
                                'action_mask': mask
                            },
                            'action': actions,
                            'adv': advantages,
                            'target': td_target
                        })
                    except Exception as e:
                        continue
            
            except Exception as e:
                print(f"{self.name} episode {episode} failed: {e}")
                continue
                
        # 最终统计
        print(f"{self.name} finished all episodes. Final stats:")
        print(f"  - Total episodes: {self.episode_count}")
        print(f"  - Win rate: {win_count / max(1, self.episode_count):.3f}")
        print(f"  - Average reward: {total_reward / max(1, self.episode_count):.3f}")
        print(f"  - Average episode length: {np.mean(episode_lengths):.1f}")
        
        # 关闭 writer
        writer.close()