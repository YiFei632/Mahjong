# 简单的调试脚本
from env import MahjongGBEnv
from feature import FeatureAgent
import traceback

# 测试环境reset
env = MahjongGBEnv(config={'agent_clz': FeatureAgent})

print("Testing environment reset...")
print(f"Environment agent_names: {env.agent_names}")

for i in range(3):
    try:
        print(f"\n--- Reset attempt {i+1} ---")
        obs = env.reset()
        print(f"Reset successful, obs keys: {list(obs.keys())}")
        print(f"Expected agent_names: {env.agent_names}")
        
        # 检查每个玩家的观测
        for agent_name, agent_obs in obs.items():
            print(f"  {agent_name}: obs shape = {agent_obs['observation'].shape}, mask sum = {agent_obs['action_mask'].sum()}")
        
        # 测试多步
        step_count = 0
        done = False
        while not done and step_count < 5:
            step_count += 1
            print(f"\n  Step {step_count}:")
            
            # 为每个在obs中的agent选择动作
            actions = {}
            for agent_name in obs:
                action_mask = obs[agent_name]['action_mask']
                valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
                if valid_actions:
                    actions[agent_name] = valid_actions[0]  # 选择第一个有效动作
                else:
                    actions[agent_name] = 0  # Pass
                print(f"    {agent_name}: action = {actions[agent_name]}, valid_actions = {len(valid_actions)}")
            
            next_obs, rewards, done = env.step(actions)
            print(f"    Step result: done = {done}, rewards = {rewards}")
            print(f"    Next obs keys: {list(next_obs.keys()) if next_obs else 'None'}")
            
            obs = next_obs
            
            if done:
                print(f"    Game ended after {step_count} steps")
                break
        
        if step_count >= 5:
            print(f"    Game continued beyond 5 steps")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        break