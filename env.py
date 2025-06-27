from agent import MahjongGBAgent

import random
from collections import defaultdict

try:
    from MahjongGB import MahjongFanCalculator
except:
    print('MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information.')
    raise

from MahjongGB import (
    MahjongShanten, 
    RegularShanten, 
    SevenPairsShanten, 
    ThirteenOrphansShanten,
    HonorsAndKnittedTilesShanten,
    KnittedStraightShanten
)

class Error(Exception):
    pass

class MahjongGBEnv():
    
    agent_names = ['player_%d' % i for i in range(1, 5)]
    
    def __init__(self, config):
        assert 'agent_clz' in config, "must specify agent_clz to process features!"
        self.agentclz = config['agent_clz']
        assert issubclass(self.agentclz, MahjongGBAgent), "ageng_clz must be a subclass of MahjongGBAgent!"
        self.duplicate = config.get('duplicate', True)
        self.variety = config.get('variety', -1)
        self.r = random.Random()
        self.normalizeReward = config.get('reward_norm', False)
        self.observation_space = self.agentclz.observation_space
        self.action_space = self.agentclz.action_space
        

        self.reward_config = {
            'tenpai_major_bonus': 15.0,      # 听牌重要奖励
            'draw_penalty': -5,           # 平局惩罚
            'shanten_rewards': {            
                0: 10.0,   # 听牌 - 大幅提高
                1: 3.0,    # 1向听
                2: 0.5,    # 2向听
                3: -1.0,   # 3向听开始惩罚
                4: -5.0,   # 4向听重惩罚
                5: -10.0,  # 5向听及以上严重惩罚
                6: -15.0,
                7: -20.0,
                8: -25.0
            },
            'shanten_improvement_bonus': {
                1: 5.0,    # 向听数减少1
                2: 12.0,   # 向听数减少2
                3: 25.0,   # 向听数减少3或更多
            },
            'shanten_regression_penalty': {
                1: -8.0,   # 向听数增加1
                2: -20.0,  # 向听数增加2
                3: -40.0,  # 向听数增加3或更多
            },
            'slow_progress_penalty': -0.5,   # 每轮没进步的小惩罚
            'stagnation_penalty': -2.0,     # 长时间停滞的惩罚
            'high_fan_bonus_threshold': 6, # 高番奖励阈值
            'high_fan_bonus_rate': 0.5,     # 高番额外奖励比例
            'self_draw_bonus': 2.0,         # 自摸额外奖励
            'progress_reward': 0.1,         # 向听数改善奖励
            

            'pair_bonus': 0.01,             # 从0.8降到0.01
            'triplet_bonus': 0.02,          # 从1.5降到0.02
            'sequence_bonus': 0.015,        # 从1.2降到0.015
            'potential_bonus': 0.005,       # 从0.4降到0.005
            

            'break_pair_penalty': -2.5,     # 拆对子惩罚
            'break_triplet_penalty': -4.0,  # 拆刻子惩罚
            'break_sequence_penalty': -3.0, # 拆顺子惩罚
            'break_potential_penalty': -1.0, # 拆潜在结构惩罚
            

            'missed_hu_penalty': -10.0,     # 错过胡牌重惩罚
            'hu_mega_bonus': 40.0,          # 胡牌巨额奖励
            'max_structure_reward': 0.15,   # 每轮最大结构奖励限制
            
            # 自立完成奖励
            'natural_completion_bonus': 2.0, # 自然完成面子奖励
            'hand_efficiency_reward': 0.5,   # 手牌效率奖励
        }

        # 记录玩家手牌结构历史
        self.player_hand_structures = [None] * 4
        self.prev_hand_structures = [None] * 4
        

        self.turn_count = 0
        self.last_can_hu = [False] * 4  # 记录上轮是否可胡牌
        self.player_stagnation_count = [0] * 4
        self.player_best_shanten = [8] * 4  # 记录每个玩家达到过的最好向听数
        
        # TensorBoard相关
        self.tensorboard_writer = config.get('tensorboard_writer', None)
        self.episode_count = 0
        
        # 统计数据
        self.global_stats = {
            'total_episodes': 0,
            'total_huang': 0,
            'total_wins': 0,
            'total_steps': 0,
            'win_by_fan': defaultdict(int),  # 按番数统计胜利
            'huang_tenpai_count': 0,  # 平局时听牌总人数
            'shanten_distribution': defaultdict(int),  # 向听数分布
        }
    
    def reset(self, prevalentWind=-1, tileWall=''):
        # Create agents to process features
        self.agents = [self.agentclz(i) for i in range(4)]
        self.reward = None
        self.done = False
        

        self.turn_count = 0
        self.last_can_hu = [False] * 4
        self.player_stagnation_count = [0] * 4
        self.player_best_shanten = [8] * 4
        
        # Episode统计初始化
        self.episode_count += 1
        self.global_stats['total_episodes'] += 1
        self.episode_stats = {
            'steps': 0,
            'final_shanten': [8, 8, 8, 8],  # 游戏结束时各玩家向听数
            'tenpai_turns': [0, 0, 0, 0],   # 各玩家听牌回合数
            'shanten_improvements': [0, 0, 0, 0],  # 向听数改善次数
            'final_rewards': [0, 0, 0, 0],  # 最终奖励
            'winner': -1,  # 胜利者
            'is_huang': False,  # 是否平局
            'fan_count': 0,  # 胜利番数
            'win_type': 'none',  # 胜利类型
        }
        
        # Init random seed
        if self.variety > 0:
            random.seed(self.r.randint(0, self.variety - 1))
        # Init prevalent wind
        self.prevalentWind = random.randint(0, 3) if prevalentWind < 0 else prevalentWind
        for agent in self.agents:
            agent.request2obs('Wind %d' % self.prevalentWind)
        # Prepare tile wall
        if tileWall:
            self.tileWall = tileWall.split()
        else:
            self.tileWall = []
            for j in range(4):
                for i in range(1, 10):
                    self.tileWall.append('W' + str(i))
                    self.tileWall.append('B' + str(i))
                    self.tileWall.append('T' + str(i))
                for i in range(1, 5):
                    self.tileWall.append('F' + str(i))
                for i in range(1, 4):
                    self.tileWall.append('J' + str(i))
            random.shuffle(self.tileWall)
        self.originalTileWall = ' '.join(self.tileWall)
        if self.duplicate:
            self.tileWall = [self.tileWall[i * 34 : (i + 1) * 34] for i in range(4)]
        self.shownTiles = defaultdict(int)
        # Deal cards
        self._deal()
        return self._obs()
    
    def step(self, action_dict):
            self.episode_stats['steps'] += 1
            self.turn_count += 1  
            self._ensure_reward_initialized()
            

            self._record_hand_structures()
            

            self._check_missed_hu_opportunities(action_dict)
            
            try:
                if self.state == 0:
                    # After Chi/Peng, prepare to Play
                    response = self.agents[self.curPlayer].action2response(action_dict[self.agent_names[self.curPlayer]]).split()
                    if response[0] == 'Play':
                        # 在打牌前检查拆牌惩罚
                        discard_penalty = self._calculate_discard_penalty(self.curPlayer, response[1])
                        self.reward[self.curPlayer] += discard_penalty
                        
                        self._discard(self.curPlayer, response[1])
                    else:
                        raise Error(self.curPlayer)
                    self.isAboutKong = False
                    
                elif self.state == 1:
                    # After Draw, prepare to Hu/Play/Gang/BuGang
                    response = self.agents[self.curPlayer].action2response(action_dict[self.agent_names[self.curPlayer]]).split()
                    if response[0] == 'Hu':
                        self.shownTiles[self.curTile] += 1
                        self._checkMahjong(self.curPlayer, isSelfDrawn=True, isAboutKong=self.isAboutKong)
                    elif response[0] == 'Play':
                        # 打牌前检查拆牌惩罚
                        discard_penalty = self._calculate_discard_penalty(self.curPlayer, response[1])
                        self.reward[self.curPlayer] += discard_penalty
                                            
                        self.hands[self.curPlayer].append(self.curTile)
                        self._discard(self.curPlayer, response[1])
                    elif response[0] == 'Gang' and not self.myWallLast and not self.wallLast:
                        self._concealedKong(self.curPlayer, response[1])
                    elif response[0] == 'BuGang' and not self.myWallLast and not self.wallLast:
                        self._promoteKong(self.curPlayer, response[1])
                    else:
                        raise Error(self.curPlayer)
                        
                elif self.state == 2:
                    # After Play, prepare to Chi/Peng/Gang/Hu/Pass
                    responses = {i: self.agents[i].action2response(action_dict[self.agent_names[i]]) for i in range(4) if i != self.curPlayer}
                    t = {i: responses[i].split() for i in responses}
                    
                    # Priority: Hu > Peng/Gang > Chi
                    for j in range(1, 4):
                        i = (self.curPlayer + j) % 4
                        if t[i][0] == 'Hu':
                            self._checkMahjong(i)
                            break
                    else:
                        for j in range(1, 4):
                            i = (self.curPlayer + j) % 4
                            if t[i][0] == 'Gang' and self._canDrawTile(i) and not self.wallLast:
                                self._kong(i, self.curTile)
                                break
                            elif t[i][0] == 'Peng' and not self.wallLast:
                                self._pung(i, self.curTile)
                                break
                        else:
                            i = (self.curPlayer + 1) % 4
                            if t[i][0] == 'Chi' and not self.wallLast:
                                self._chow(i, t[i][1])
                            else:
                                for j in range(1, 4):
                                    i = (self.curPlayer + j) % 4
                                    if t[i][0] != 'Pass': 
                                        raise Error(i)
                                if self.wallLast:
                                    # 平局处理
                                    self._handle_draw_game()
                                else:
                                    # Next player
                                    self.curPlayer = (self.curPlayer + 1) % 4
                                    self._draw(self.curPlayer)
                                    
                elif self.state == 3:
                    # After BuGang, prepare to Hu/Pass
                    responses = {i: self.agents[i].action2response(action_dict[self.agent_names[i]]) for i in range(4) if i != self.curPlayer}
                    for j in range(1, 4):
                        i = (self.curPlayer + j) % 4
                        if responses[i] == 'Hu':
                            self._checkMahjong(i, isAboutKong=True)
                            break
                    else:
                        for j in range(1, 4):
                            i = (self.curPlayer + j) % 4
                            if responses[i] != 'Pass': 
                                raise Error(i)
                        self._draw(self.curPlayer)
                

                if not self.done:
                    self._give_shanten_rewards()              # 保持向听数奖励
                    self._give_limited_structure_rewards()    # 限制结构奖励
                    self._collect_shanten_stats()
                
                # 游戏结束时记录统计
                if self.done:
                    self._log_episode_stats()
                    
            except Error as e:
                player = e.args[0]
                self.obs = {i: self.agents[i].request2obs('Player %d Invalid' % player) for i in range(4)}
                self.reward = [10] * 4
                self.reward[player] = -30
                self.done = True
                self._log_episode_stats()  # 即使出错也记录统计
                
            return self._obs(), self._reward(), self._done()


    def _promoteKong(self, player, tile):
        self.hands[player].append(self.curTile)
        idx = -1
        for i in range(len(self.packs[player])):
            if self.packs[player][i][0] == 'PENG' and self.packs[player][i][1] == tile:
                idx = i
        if idx < 0: raise Error(player)
        self.hands[player].remove(tile)
        offer = self.packs[player][idx][2]
        self.packs[player][idx] = ('GANG', tile, offer)
        self.shownTiles[tile] = 4
        self.state = 3
        self.curPlayer = player
        self.curTile = tile
        self.drawAboutKong = True
        self.isAboutKong = False
        self.agents[player].request2obs('Player %d BuGang %s' % (player, tile))
        self.obs = {i : self.agents[i].request2obs('Player %d BuGang %s' % (player, tile)) for i in range(4) if i != player}
        
    def _calculate_best_shanten(self, player):
        try:
            hand = tuple(self.hands[player])
            pack = tuple(self.packs[player])
            
            if not hand:
                return 8, []  # 空手牌，返回大向听数
            
            for tile in hand:
                if not isinstance(tile, str) or len(tile) < 2:
                    return 8, []  # 无效牌格式
            
            for pack_item in pack:
                if not isinstance(pack_item, tuple) or len(pack_item) != 3:
                    return 8, []  # 无效牌组格式
            
            # 如果手牌不足13张且没有牌组，只计算基本向听数
            total_tiles = len(hand) + sum(3 if pack_item[0] == 'CHI' or pack_item[0] == 'PENG' 
                                        else 4 if pack_item[0] == 'GANG' 
                                        else 0 for pack_item in pack)
            
            # 只有在合理的牌数情况下才计算向听数
            if total_tiles < 10 or total_tiles > 14:
                return 8, []
            
            # 如果手牌不是13张，使用基本向听数计算
            if len(hand) != 13:
                try:
                    shanten = MahjongShanten(pack=pack, hand=hand)
                    return shanten, []
                except Exception as e:
                    return 8, []
            
            # 13张手牌的情况，计算所有可能的向听数
            shanten_results = []
            
            try:
                # 1. 基本胡型
                regular_shanten, regular_useful = RegularShanten(hand=hand)
                shanten_results.append((regular_shanten, regular_useful, "regular"))
            except Exception as e:
                pass
            
            try:
                # 2. 七对子
                seven_pairs_shanten, seven_pairs_useful = SevenPairsShanten(hand=hand)
                shanten_results.append((seven_pairs_shanten, seven_pairs_useful, "seven_pairs"))
            except Exception as e:
                pass
            
            try:
                # 3. 十三幺
                thirteen_orphans_shanten, thirteen_orphans_useful = ThirteenOrphansShanten(hand=hand)
                shanten_results.append((thirteen_orphans_shanten, thirteen_orphans_useful, "thirteen_orphans"))
            except Exception as e:
                pass
            
            try:
                # 4. 全不靠
                honors_shanten, honors_useful = HonorsAndKnittedTilesShanten(hand=hand)
                shanten_results.append((honors_shanten, honors_useful, "honors"))
            except Exception as e:
                pass
            
            try:
                # 5. 组合龙
                knitted_shanten, knitted_useful = KnittedStraightShanten(hand=hand)
                shanten_results.append((knitted_shanten, knitted_useful, "knitted"))
            except Exception as e:
                pass
            
            if not shanten_results:
                return 8, []  # 如果都计算失败，返回一个很大的向听数
            
            # 返回最小的向听数
            best_result = min(shanten_results, key=lambda x: x[0])
            return best_result[0], best_result[1]
            
        except Exception as e:
            # 如果出现任何异常，返回安全值
            return 8, []  

    def _checkMahjong(self, player, isSelfDrawn=False, isAboutKong=False):
        try:
            fans = MahjongFanCalculator(
                pack=tuple(self.packs[player]),
                hand=tuple(self.hands[player]),
                winTile=self.curTile,
                flowerCount=0,
                isSelfDrawn=isSelfDrawn,
                is4thTile=(self.shownTiles[self.curTile] + isSelfDrawn) == 4,
                isAboutKong=isAboutKong,
                isWallLast=self.wallLast,
                seatWind=player,
                prevalentWind=self.prevalentWind,
                verbose=True
            )
            
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            
            if fanCnt < 8: 
                raise Exception('Not Enough Fans')
            
            self.obs = {i: self.agents[i].request2obs('Player %d Hu' % player) for i in range(4)}
            
            # 记录胜利统计
            self.episode_stats['winner'] = player
            self.episode_stats['fan_count'] = fanCnt
            self.episode_stats['win_type'] = 'self_draw' if isSelfDrawn else 'ron'
            self.global_stats['total_wins'] += 1
            self.global_stats['win_by_fan'][fanCnt] += 1

            # 超级胡牌奖励
            base_hu_reward = self.reward_config['hu_mega_bonus']
            
            # 番数奖励
            fan_bonus = fanCnt * 3.0  # 每番3分
            
            # 自摸奖励
            ziMo_bonus = self.reward_config['self_draw_bonus'] if isSelfDrawn else 0
            
            # 高番额外奖励
            high_fan_bonus = 0
            if fanCnt >= self.reward_config['high_fan_bonus_threshold']:
                excess_fan = fanCnt - self.reward_config['high_fan_bonus_threshold']
                high_fan_bonus = excess_fan * self.reward_config['high_fan_bonus_rate']
            
            # 速度奖励（早期胡牌额外奖励）
            speed_bonus = max(0, (100 - self.episode_stats['steps']) * 0.2)
            
            # 计算胜利者总奖励
            winner_total_reward = (base_hu_reward + fan_bonus + 
                                 ziMo_bonus + high_fan_bonus + speed_bonus)
            
            # 设置奖励
            if isSelfDrawn:
                self.reward = [-(fanCnt + 8)] * 4
                self.reward[player] = winner_total_reward
            else:
                self.reward = [-8] * 4
                self.reward[player] = winner_total_reward
                self.reward[self.curPlayer] -= fanCnt * 2  # 放炮者额外惩罚
            
            # 记录最终奖励
            self.episode_stats['final_rewards'] = self.reward.copy()
            
            self.done = True
            
        except Exception as e:
            raise Error(player)
    
    def _calculate_hand_quality_bonus(self, player):
        bonus = 0.0
        
        if self.player_hand_structures[player]:
            final_structure = self.player_hand_structures[player]
        else:
            final_structure = self._analyze_hand_structure(self.hands[player])
        
        # 1. 门清奖励(没有吃碰杠)
        exposed_melds = len(self.packs[player])
        if exposed_melds == 0:
            bonus += 8.0  # 门清大奖励
        
        # 2. 自然结构奖励
        natural_pairs = len(final_structure['pairs'])
        natural_triplets = len(final_structure['triplets']) 
        natural_sequences = len(final_structure['sequences'])
        
        natural_bonus = (natural_pairs * 1.0 + 
                        natural_triplets * 2.0 + 
                        natural_sequences * 1.5)
        bonus += natural_bonus
        
        # 3. 结构效率奖励
        total_useful_tiles = (natural_pairs * 2 + 
                            natural_triplets * 3 + 
                            natural_sequences * 3)
        if len(self.hands[player]) > 0:
            efficiency = total_useful_tiles / len(self.hands[player])
            bonus += efficiency * 3.0
        
        # 4. 自立完成vs依赖外部的对比奖励
        natural_melds = natural_triplets + natural_sequences
        if natural_melds > exposed_melds:
            self_reliance_ratio = natural_melds / max(natural_melds + exposed_melds, 1)
            bonus += self_reliance_ratio * 5.0  # 自立完成大奖励
        
        return bonus    
    
    def _calculate_bonus_rewards(self, winner, fanCnt, isSelfDrawn):
        bonus_rewards = [0.0] * 4
        
        if fanCnt >= self.reward_config['high_fan_bonus_threshold']:
            excess_fan = fanCnt - self.reward_config['high_fan_bonus_threshold']
            high_fan_bonus = excess_fan * self.reward_config['high_fan_bonus_rate']
            bonus_rewards[winner] += high_fan_bonus
        
        #自摸额外奖励
        if isSelfDrawn:
            bonus_rewards[winner] += self.reward_config['self_draw_bonus']
        
        return bonus_rewards
    

    def _give_shanten_rewards(self):
        for player in range(4):
            current_shanten, useful_tiles = self._calculate_best_shanten(player)
            
            #基础向听数奖励
            base_reward = self.reward_config['shanten_rewards'].get(current_shanten, -30.0)
            
            #有用牌数量调整
            if useful_tiles and current_shanten <= 2:
                useful_count = len(useful_tiles)
                useful_multiplier = 1.0 + min(useful_count / 15.0, 1.0)
                base_reward *= useful_multiplier
            
            self.reward[player] += base_reward
            
            #进步奖励
            if hasattr(self, 'previous_shanten') and player in self.previous_shanten:
                prev_shanten = self.previous_shanten[player]
                shanten_change = prev_shanten - current_shanten
                
                if shanten_change > 0:  # 向听数减少了
                    improvement_bonus = self.reward_config['shanten_improvement_bonus'].get(
                        min(shanten_change, 3), 
                        shanten_change * 5.0
                    )
                    self.reward[player] += improvement_bonus
                    self.episode_stats['shanten_improvements'][player] += 1
                    
                    # 重置停滞计数
                    self.player_stagnation_count[player] = 0
                    
                    # 更新最佳记录
                    if current_shanten < self.player_best_shanten[player]:
                        self.player_best_shanten[player] = current_shanten
                        # 创新最佳记录额外奖励
                        self.reward[player] += 3.0
                
                elif shanten_change < 0:  # 向听数增加了（退步）
                    regression_penalty = self.reward_config['shanten_regression_penalty'].get(
                        min(abs(shanten_change), 3),
                        abs(shanten_change) * -10.0
                    )
                    self.reward[player] += regression_penalty
                
                else: 
                    self.player_stagnation_count[player] += 1
                    
                    # 停滞惩罚
                    if self.player_stagnation_count[player] > 3:
                        self.reward[player] += self.reward_config['slow_progress_penalty']
                    
                    if self.player_stagnation_count[player] > 8:
                        self.reward[player] += self.reward_config['stagnation_penalty']
            
            # 听牌奖励
            if current_shanten == 0:
                tenpai_bonus = self.reward_config['tenpai_major_bonus']
                if useful_tiles:
                    # 根据有效牌数量给额外奖励
                    useful_count = len(useful_tiles)
                    if useful_count >= 8:  # 多面听
                        tenpai_bonus *= 1.5
                    elif useful_count >= 5:
                        tenpai_bonus *= 1.2
                
                self.reward[player] += tenpai_bonus
        
        # 更新previous_shanten
        self.previous_shanten = {}
        for player in range(4):
            shanten, _ = self._calculate_best_shanten(player)
            self.previous_shanten[player] = shanten

    def _collect_shanten_stats(self):
        """收集向听数统计数据"""
        for player in range(4):
            try:
                shanten, useful_tiles = self._calculate_best_shanten(player)
                
                # 更新最终向听数
                self.episode_stats['final_shanten'][player] = shanten
                
                # 统计听牌回合数
                if shanten == 0:
                    self.episode_stats['tenpai_turns'][player] += 1
                
                # 更新全局向听数分布
                self.global_stats['shanten_distribution'][shanten] += 1
                
            except:
                continue

    def _handle_draw_game(self):
        """处理平局情况"""
        self.obs = {i: self.agents[i].request2obs('Huang') for i in range(4)}
        self.reward = [0, 0, 0, 0]
        
        # 标记为平局
        self.episode_stats['is_huang'] = True
        self.global_stats['total_huang'] += 1
        
        # 统计平局时听牌情况
        huang_tenpai_count = 0
        huang_tenpai_details = []
        
        for i in range(4):
            self.reward[i] += self.reward_config['draw_penalty']
            
            try:
                shanten, useful_tiles = self._calculate_best_shanten(i)
                self.episode_stats['final_shanten'][i] = shanten
                
                if shanten == 0:
                    huang_tenpai_count += 1
                    useful_count = len(useful_tiles) if useful_tiles else 0
                    huang_tenpai_details.append({
                        'player': i,
                        'useful_count': useful_count
                    })
                    
                    # 听牌奖励
                    tenpai_bonus = 1.0
                    if useful_tiles:
                        tenpai_bonus *= (1.0 + min(useful_count / 10.0, 1.0))
                    self.reward[i] += tenpai_bonus
                    
                elif shanten in [1, 2]:
                    self.reward[i] += self.reward_config['shanten_rewards'].get(shanten, 0) * 0.5
                    
            except:
                self.episode_stats['final_shanten'][i] = 8
        
        # 记录平局听牌统计
        self.global_stats['huang_tenpai_count'] += huang_tenpai_count
        self.episode_stats['huang_tenpai_players'] = huang_tenpai_count
        self.episode_stats['huang_tenpai_details'] = huang_tenpai_details
        
        # 记录最终奖励
        self.episode_stats['final_rewards'] = self.reward.copy()
        
        self.done = True

    def _log_episode_stats(self):
        """记录episode统计到TensorBoard"""
        if not self.tensorboard_writer:
            return
            
        episode = self.episode_count
        
        # 基础游戏统计
        self.tensorboard_writer.add_scalar('Game/EpisodeLength', self.episode_stats['steps'], episode)
        
        # 胜负统计
        if self.episode_stats['is_huang']:
            self.tensorboard_writer.add_scalar('Game/Huang', 1, episode)
            self.tensorboard_writer.add_scalar('Game/Win', 0, episode)
            
            # 平局听牌统计
            self.tensorboard_writer.add_scalar('Huang/TenpaiPlayersCount', self.episode_stats['huang_tenpai_players'], episode)
            
            # 每个玩家的平局奖励
            for i in range(4):
                self.tensorboard_writer.add_scalar(f'Huang/Player{i}_FinalReward', self.episode_stats['final_rewards'][i], episode)
                
        else:
            self.tensorboard_writer.add_scalar('Game/Huang', 0, episode)
            self.tensorboard_writer.add_scalar('Game/Win', 1, episode)
            
            if self.episode_stats['winner'] != -1:
                # 胜利统计
                winner = self.episode_stats['winner']
                self.tensorboard_writer.add_scalar('Win/Winner', winner, episode)
                self.tensorboard_writer.add_scalar('Win/FanCount', self.episode_stats['fan_count'], episode)
                self.tensorboard_writer.add_scalar('Win/WinType', 1 if self.episode_stats['win_type'] == 'self_draw' else 0, episode)
                
                # 胜利奖励
                self.tensorboard_writer.add_scalar(f'Win/WinnerReward', self.episode_stats['final_rewards'][winner], episode)
        
        # 向听数统计
        for i in range(4):
            final_shanten = self.episode_stats['final_shanten'][i]
            tenpai_turns = self.episode_stats['tenpai_turns'][i]
            improvements = self.episode_stats['shanten_improvements'][i]
            
            self.tensorboard_writer.add_scalar(f'Shanten/Player{i}_FinalShanten', final_shanten, episode)
            self.tensorboard_writer.add_scalar(f'Shanten/Player{i}_TenpaiTurns', tenpai_turns, episode)
            self.tensorboard_writer.add_scalar(f'Shanten/Player{i}_Improvements', improvements, episode)
            
            # 听牌率统计
            tenpai_rate = tenpai_turns / max(self.episode_stats['steps'], 1)
            self.tensorboard_writer.add_scalar(f'Shanten/Player{i}_TenpaiRate', tenpai_rate, episode)
        
        # 整体向听数分布
        avg_final_shanten = sum(self.episode_stats['final_shanten']) / 4
        self.tensorboard_writer.add_scalar('Shanten/AverageFinalShanten', avg_final_shanten, episode)
        
        # 奖励统计
        total_reward = sum(self.episode_stats['final_rewards'])
        avg_reward = total_reward / 4
        max_reward = max(self.episode_stats['final_rewards'])
        min_reward = min(self.episode_stats['final_rewards'])
        
        self.tensorboard_writer.add_scalar('Reward/TotalReward', total_reward, episode)
        self.tensorboard_writer.add_scalar('Reward/AverageReward', avg_reward, episode)
        self.tensorboard_writer.add_scalar('Reward/MaxReward', max_reward, episode)
        self.tensorboard_writer.add_scalar('Reward/MinReward', min_reward, episode)
        
        # 每隔100个episode记录累积统计
        if episode % 100 == 0:
            self._log_cumulative_stats(episode)
    
    def _log_cumulative_stats(self, episode):
        """记录累积统计数据"""
        if not self.tensorboard_writer:
            return
            
        total_episodes = self.global_stats['total_episodes']
        if total_episodes == 0:
            return
        
        # 胜率和平局率
        win_rate = self.global_stats['total_wins'] / total_episodes
        huang_rate = self.global_stats['total_huang'] / total_episodes
        
        self.tensorboard_writer.add_scalar('Cumulative/WinRate', win_rate, episode)
        self.tensorboard_writer.add_scalar('Cumulative/HuangRate', huang_rate, episode)
        
        # 平局时平均听牌人数
        if self.global_stats['total_huang'] > 0:
            avg_huang_tenpai = self.global_stats['huang_tenpai_count'] / self.global_stats['total_huang']
            self.tensorboard_writer.add_scalar('Cumulative/AvgHuangTenpaiCount', avg_huang_tenpai, episode)
        
        # 番数分布
        total_wins = self.global_stats['total_wins']
        if total_wins > 0:
            for fan_count, count in self.global_stats['win_by_fan'].items():
                fan_rate = count / total_wins
                self.tensorboard_writer.add_scalar(f'Cumulative/FanDistribution/{fan_count}Fan', fan_rate, episode)
        
        # 向听数分布
        total_shanten_records = sum(self.global_stats['shanten_distribution'].values())
        if total_shanten_records > 0:
            for shanten, count in self.global_stats['shanten_distribution'].items():
                shanten_rate = count / total_shanten_records
                self.tensorboard_writer.add_scalar(f'Cumulative/ShantenDistribution/{shanten}Shanten', shanten_rate, episode)
        
        # 打印统计摘要
        if episode % 500 == 0:
            print(f"\n=== Episode {episode} 统计摘要 ===")
            print(f"总局数: {total_episodes}")
            print(f"胜率: {win_rate:.3f}")
            print(f"平局率: {huang_rate:.3f}")
            if self.global_stats['total_huang'] > 0:
                print(f"平局时平均听牌人数: {avg_huang_tenpai:.2f}")
            print("=" * 30)

    def get_stats_summary(self):
        """获取统计摘要（用于调试）"""
        total_episodes = self.global_stats['total_episodes']
        if total_episodes == 0:
            return "No episodes completed yet."
        
        win_rate = self.global_stats['total_wins'] / total_episodes
        huang_rate = self.global_stats['total_huang'] / total_episodes
        
        summary = {
            'total_episodes': total_episodes,
            'win_rate': win_rate,
            'huang_rate': huang_rate,
            'total_wins': self.global_stats['total_wins'],
            'total_huang': self.global_stats['total_huang'],
        }
        
        if self.global_stats['total_huang'] > 0:
            summary['avg_huang_tenpai'] = self.global_stats['huang_tenpai_count'] / self.global_stats['total_huang']
        
        return summary
    
    def _ensure_reward_initialized(self):
        if not hasattr(self, 'reward') or self.reward is None:
            self.reward = [0.0] * 4
        elif not isinstance(self.reward, list) or len(self.reward) != 4:
            self.reward = [0.0] * 4
    

    def _check_missed_hu_opportunities(self, action_dict):
        """检查长时间听牌惩罚"""
        try:
            for i in range(4):
                current_shanten, useful_tiles = self._calculate_best_shanten(i)
                
                # 如果当前是听牌状态
                if current_shanten == 0:
                    if not hasattr(self, 'tenpai_count'):
                        self.tenpai_count = [0] * 4
                    
                    self.tenpai_count[i] += 1
                    
                    # 长时间听牌递增惩罚
                    if self.tenpai_count[i] > 10:  
                        penalty_turns = self.tenpai_count[i] - 8
                        penalty = -0.2 * penalty_turns  # 每多一轮惩罚-0.2
                        self.reward[i] += penalty
                    
                    # 极长时间听牌重惩罚
                    if self.tenpai_count[i] > 15:
                        self.reward[i] += -2.0
                        
                else:
                    # 不是听牌状态，重置计数
                    if hasattr(self, 'tenpai_count'):
                        self.tenpai_count[i] = 0
                        
        except Exception as e:
            # 静默处理错误，不影响训练
            pass

    def _simple_can_hu_check(self, player):
        """简单的听牌状态检查"""
        try:
            current_shanten, useful_tiles = self._calculate_best_shanten(player)
            return current_shanten <= 3  # 只检查是否听牌
        except:
            return False
    
    def _record_hand_structures(self):
        """记录当前手牌结构"""
        self.prev_hand_structures = [struct.copy() if struct else None for struct in self.player_hand_structures]
        
        for i, agent in enumerate(self.agents):
            if hasattr(agent, 'hand'):
                self.player_hand_structures[i] = self._analyze_hand_structure(agent.hand)
    
    def _analyze_hand_structure(self, hand):
        """分析手牌结构"""
        if not hand:
            return {'pairs': [], 'triplets': [], 'sequences': [], 'potential_sequences': [], 'score': 0}
        
        # 统计牌数
        tile_counts = defaultdict(int)
        for tile in hand:
            tile_counts[tile] += 1
        
        structure = {
            'pairs': [],              # 对子
            'triplets': [],           # 刻子
            'sequences': [],          # 顺子
            'potential_sequences': [], # 潜在顺子
            'score': 0
        }
        
        # 识别对子和刻子
        for tile, count in tile_counts.items():
            if count == 2:
                structure['pairs'].append(tile)
            elif count >= 3:
                structure['triplets'].append(tile)
        
        # 识别顺子和潜在顺子(仅数牌)
        for suit in ['W', 'T', 'B']:
            nums = []
            for tile in tile_counts:
                if tile.startswith(suit) and len(tile) == 2:
                    nums.extend([int(tile[1])] * tile_counts[tile])
            nums.sort()
            
            # 检查连续的三张牌(顺子)
            i = 0
            while i < len(nums) - 2:
                if nums[i+1] == nums[i] + 1 and nums[i+2] == nums[i] + 2:
                    sequence = (f"{suit}{nums[i]}", f"{suit}{nums[i+1]}", f"{suit}{nums[i+2]}")
                    structure['sequences'].append(sequence)
                    i += 3  # 跳过已使用的牌
                else:
                    i += 1
            
            # 检查连续的两张牌(潜在顺子)
            unique_nums = sorted(set(num for num in nums))
            for i in range(len(unique_nums) - 1):
                if unique_nums[i+1] == unique_nums[i] + 1:
                    pot_seq = (f"{suit}{unique_nums[i]}", f"{suit}{unique_nums[i+1]}")
                    structure['potential_sequences'].append(pot_seq)
        
        # 计算结构分数
        structure['score'] = (len(structure['pairs']) * 2 + 
                            len(structure['triplets']) * 3 + 
                            len(structure['sequences']) * 3 + 
                            len(structure['potential_sequences']) * 1)
        
        return structure
    
    def _calculate_discard_penalty(self, player, discard_tile):
        """计算打牌的拆牌惩罚"""
        try:
            if not self.prev_hand_structures[player]:
                return 0.0
            
            penalty = 0.0
            prev_structure = self.prev_hand_structures[player]
            
            # 模拟打出这张牌后的手牌
            temp_hand = self.hands[player].copy()
            if discard_tile in temp_hand:
                temp_hand.remove(discard_tile)
                new_structure = self._analyze_hand_structure(temp_hand)
                
                # 检查结构损失
                pair_loss = len(prev_structure['pairs']) - len(new_structure['pairs'])
                triplet_loss = len(prev_structure['triplets']) - len(new_structure['triplets'])
                sequence_loss = len(prev_structure['sequences']) - len(new_structure['sequences'])
                potential_loss = len(prev_structure['potential_sequences']) - len(new_structure['potential_sequences'])
                
                # 应用惩罚
                penalty += pair_loss * self.reward_config['break_pair_penalty']
                penalty += triplet_loss * self.reward_config['break_triplet_penalty']
                penalty += sequence_loss * self.reward_config['break_sequence_penalty']
                penalty += potential_loss * self.reward_config['break_potential_penalty']
                
                # 额外检查：是否拆了特定的好牌型
                if self._is_breaking_valuable_structure(discard_tile, prev_structure):
                    penalty += -1.5  # 额外惩罚
            
            return penalty
        except:
            return 0.0
    
    def _is_breaking_valuable_structure(self, discard_tile, structure):
        """检查是否拆了有价值的结构"""
        try:
            # 检查是否是对子的一部分
            if discard_tile in structure['pairs']:
                return True
            
            # 检查是否是刻子的一部分
            if discard_tile in structure['triplets']:
                return True
            
            # 检查是否是顺子的一部分
            for seq in structure['sequences']:
                if discard_tile in seq:
                    return True
            
            # 检查是否是潜在顺子的一部分
            for pot_seq in structure['potential_sequences']:
                if discard_tile in pot_seq:
                    return True
            
            return False
        except:
            return False
    

    def _give_limited_structure_rewards(self):
        """给予限制的结构维持奖励"""
        try:
            for player in range(4):
                if not self.player_hand_structures[player]:
                    continue
                    
                current_structure = self.player_hand_structures[player]
                

                structure_reward = (
                    len(current_structure['pairs']) * self.reward_config['pair_bonus'] +
                    len(current_structure['triplets']) * self.reward_config['triplet_bonus'] +
                    len(current_structure['sequences']) * self.reward_config['sequence_bonus'] +
                    len(current_structure['potential_sequences']) * self.reward_config['potential_bonus']
                )
                

                structure_reward = min(structure_reward, self.reward_config['max_structure_reward'])
                
                self.reward[player] += structure_reward
                

                if self.prev_hand_structures[player]:
                    prev_structure = self.prev_hand_structures[player]
                    score_improvement = current_structure['score'] - prev_structure['score']
                    if score_improvement > 0:
                        improvement_reward = min(score_improvement * 0.05, 0.1)  # 大幅限制
                        self.reward[player] += improvement_reward
        except:
            pass
    
    def _obs(self):
        return {self.agent_names[k] : v for k, v in self.obs.items()}
    
    def _reward(self):
        if self.reward: return {self.agent_names[k] : self.reward[k] for k in self.obs}
        return {self.agent_names[k] : 0 for k in self.obs}
    
    def _done(self):
        return self.done
    
    def _drawTile(self, player):
        if self.duplicate:
            return self.tileWall[player].pop()
        return self.tileWall.pop()
    
    def _canDrawTile(self, player):
        if self.duplicate:
            return bool(self.tileWall[player])
        return bool(self.tileWall)
    
    def _deal(self):
        self.hands = []
        self.packs = []
        for i in range(4):
            hand = []
            while len(hand) < 13:
                tile = self._drawTile(i)
                hand.append(tile)
            self.hands.append(hand)
            self.packs.append([])
            self.agents[i].request2obs(' '.join(['Deal', *hand]))
        self.curPlayer = 0
        self.drawAboutKong = False
        self._draw(self.curPlayer)
    
    def _draw(self, player):
        tile = self._drawTile(player)
        self.myWallLast = not self._canDrawTile(player)
        self.wallLast = not self._canDrawTile((player + 1) % 4)
        self.isAboutKong = self.drawAboutKong
        self.drawAboutKong = False
        self.state = 1
        self.curTile = tile
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d Draw' % player)
        self.obs = {player : self.agents[player].request2obs('Draw %s' % tile)}
    
    def _discard(self, player, tile):
        if tile not in self.hands[player]: raise Error(player)
        self.hands[player].remove(tile)
        self.shownTiles[tile] += 1
        self.wallLast = not self._canDrawTile((player + 1) % 4)
        self.curTile = tile
        self.state = 2
        self.agents[player].request2obs('Player %d Play %s' % (player, tile))
        self.obs = {i : self.agents[i].request2obs('Player %d Play %s' % (player, tile)) for i in range(4) if i != player}
    
    def _kong(self, player, tile):
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 4: raise Error(player)
        for i in range(4): self.hands[player].remove(tile)
        # offer: 0 for self, 123 for up/oppo/down
        self.packs[player].append(('GANG', tile, (player + 4 - self.curPlayer) % 4))
        self.shownTiles[tile] = 4
        self.curPlayer = player
        self.drawAboutKong = True
        self.isAboutKong = False
        for agent in self.agents:
            agent.request2obs('Player %d Gang' % player)
        self._draw(player)
    
    def _pung(self, player, tile):
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 3: raise Error(player)
        for i in range(3): self.hands[player].remove(tile)
        # offer: 0 for self, 123 for up/oppo/down
        self.packs[player].append(('PENG', tile, (player + 4 - self.curPlayer) % 4))
        self.shownTiles[tile] += 2
        self.state = 0
        self.curPlayer = player
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d Peng' % player)
        self.obs = {player : self.agents[player].request2obs('Player %d Peng' % player)}
    
    def _chow(self, player, tile):
        self.hands[player].append(self.curTile)
        self.shownTiles[self.curTile] -= 1
        color = tile[0]
        num = int(tile[1])
        for i in range(-1, 2):
            t = color + str(num + i)
            if t not in self.hands[player]: raise Error(player)
            self.hands[player].remove(t)
            self.shownTiles[t] += 1
        # offer: 123 for which tile is offered
        self.packs[player].append(('CHI', tile, int(self.curTile[1]) - num + 2))
        self.state = 0
        self.curPlayer = player
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d Chi %s' % (player, tile))
        self.obs = {player : self.agents[player].request2obs('Player %d Chi %s' % (player, tile))}
    
    def _concealedKong(self, player, tile):
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 4: raise Error(player)
        for i in range(4): self.hands[player].remove(tile)
        # offer: 0 for self, 123 for up/oppo/down
        self.packs[player].append(('GANG', tile, (player + 4 - self.curPlayer) % 4))
        self.shownTiles[tile] = 4
        self.curPlayer = player
        self.drawAboutKong = True
        self.isAboutKong = False
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d AnGang' % player)
        self.agents[player].request2obs('Player %d AnGang %s' % (player, tile))
        self._draw(player)
    
