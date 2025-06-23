from multiprocessing import Process
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

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
                for reset_attempt in range(3):  # 最多重试3次
                    try:
                        obs = env.reset()
                        if obs and len(obs) > 0:
                            reset_failures = 0  # 重置失败计数
                            break
                        else:
                            print(f"{self.name} env.reset() returned empty obs, attempt {reset_attempt+1}")
                    except Exception as e:
                        print(f"{self.name} env.reset() failed, attempt {reset_attempt+1}: {e}")
                        if reset_attempt == 2:  # 最后一次尝试
                            reset_failures += 1
                            if reset_failures >= max_reset_failures:
                                print(f"{self.name} too many reset failures, recreating environment")
                                # 重新创建环境
                                env = MahjongGBEnv(config = {'agent_clz': FeatureAgent})
                                reset_failures = 0
                                try:
                                    obs = env.reset()
                                except:
                                    print(f"{self.name} even new environment failed to reset")
                                    obs = None
                
                if not obs or len(obs) == 0:
                    print(f"{self.name} skipping episode {episode} - reset failed")
                    continue
                
                episode_data = {agent_name: {
                    'state' : {
                        'observation': [],
                        'action_mask': []
                    },
                    'action' : [],
                    'reward' : [],
                    'value' : []
                } for agent_name in env.agent_names}
                done = False
                step_count = 0
                max_steps = 500  # 限制最大步数，防止无限循环
                
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
                            logits, value,_ = model(state)
                            action_dist = torch.distributions.Categorical(logits = logits)
                            action = action_dist.sample().item()
                            value = value.item()
                        actions[agent_name] = action
                        values[agent_name] = value
                        agent_data['action'].append(actions[agent_name])
                        agent_data['value'].append(values[agent_name])
                    # interact with env
                    try:
                        next_obs, rewards, done = env.step(actions)
                        
                        # 检查环境返回是否有效
                        if not next_obs or len(next_obs) == 0:
                            print(f"{self.name} env.step() returned empty obs at step {step_count}, ending episode")
                            done = True
                            # 创建默认奖励
                            rewards = {name: 0 for name in env.agent_names}
                        
                        for agent_name in rewards:
                            if agent_name in episode_data:  # 只为存在的agent添加奖励
                                episode_data[agent_name]['reward'].append(rewards[agent_name])
                        obs = next_obs
                        
                    except Exception as e:
                        print(f"{self.name} env.step() failed at step {step_count}: {e}")
                        done = True
                        rewards = {name: 0 for name in env.agent_names}
                        for agent_name in env.agent_names:
                            if agent_name in episode_data:  # 只为存在的agent添加奖励
                                episode_data[agent_name]['reward'].append(0)
                
                if step_count >= max_steps:
                    print(f"{self.name} episode {episode} reached max steps ({max_steps})")
                
                # 只有在成功完成episode时才记录奖励
                if step_count > 0 and 'rewards' in locals() and rewards:
                    # 记录奖励到 TensorBoard
                    # print(self.name, 'Episode', episode, 'Model', latest['id'], 'Reward', rewards['player_1'], 'Steps', step_count)
                    writer.add_scalar(f'Reward/{self.name}', rewards['player_1'], latest['id'])
                    avg_reward = sum(rewards.values()) / len(rewards)
                    writer.add_scalar('Reward/average_all_players', avg_reward, latest['id'])
                else:
                    print(f"{self.name} Episode {episode} failed - no valid rewards")
                
                # postprocessing episode data for each agent
                for agent_name, agent_data in episode_data.items():
                    # 添加数据完整性检查
                    if (len(agent_data['action']) == 0 or 
                        len(agent_data['state']['observation']) == 0 or
                        len(agent_data['reward']) == 0):
                        # print(f"{self.name} skipping {agent_name} due to empty data")  # 注释掉，这是正常的
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
                        # print(f"{self.name} error processing {agent_name}: {e}")  # 静默处理
                        continue
            
            except Exception as e:
                print(f"{self.name} episode {episode} failed: {e}")
                continue
                
        # 关闭 writer
        print(f"{self.name} finished all episodes. Closing TensorBoard writer.")
        writer.close()