"""
!/usr/bin/python3
-*- coding: utf-8 -*-
@FileName: environment.py
@Time: 2024/4/15 下午3:54
@Author: ZhengtaoCao
@Description: None
"""

from harl.envs.battle5v5.env.env_runner import EnvRunner
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch as th
from time import sleep
from harl.envs.battle5v5.config import ADDRESS, config, POOL_NUM, ISHOST, XSIM_NUM
# from .multiagentenv import MultiAgentEnv
import math
import logging

BLUE_INFO = {
    '0': {'Name': '蓝有人机', 'ID': 6},
    '1': {'Name': '蓝无人机1', 'ID': 14},
    '2': {'Name': '蓝无人机2', 'ID': 15},
    '3': {'Name': '蓝无人机3', 'ID': 16},
    '4': {'Name': '蓝无人机4', 'ID': 17},
}
RED_INFO = {
    '0': {'Name': '红有人机', 'ID': 1},
    '1': {'Name': '红无人机1', 'ID': 2},
    '2': {'Name': '红无人机2', 'ID': 11},
    '3': {'Name': '红无人机3', 'ID': 12},
    '4': {'Name': '红无人机4', 'ID': 13},
}
BLUE_FIRE_INFO = {
    '0': {'Name': '空空导弹_1(蓝有人机_1_武器系统_1)', 'ID': 2147483668},
    '1': {'Name': '空空导弹_2(蓝有人机_1_武器系统_1)', 'ID': 2147483669},
    '2': {'Name': '空空导弹_3(蓝有人机_1_武器系统_1)', 'ID': 2147483670},
    '3': {'Name': '空空导弹_4(蓝有人机_1_武器系统_1)', 'ID': 2147483671},
    '4': {'Name': '空空导弹_1(蓝无人机_1_武器系统_1)', 'ID': 2147483657},
    '5': {'Name': '空空导弹_2(蓝无人机_1_武器系统_1)', 'ID': 2147483659},
    '6': {'Name': '空空导弹_1(蓝无人机_2_武器系统_1)', 'ID': 2147483652},
    '7': {'Name': '空空导弹_2(蓝无人机_2_武器系统_1)', 'ID': 2147483654},
    '8': {'Name': '空空导弹_1(蓝无人机_3_武器系统_1)', 'ID': 2147483660},
    '9': {'Name': '空空导弹_2(蓝无人机_3_武器系统_1)', 'ID': 2147483662},
    '10': {'Name': '空空导弹_1(蓝无人机_4_武器系统_1)', 'ID': 2147483648},
    '11': {'Name': '空空导弹_2(蓝无人机_4_武器系统_1)', 'ID': 2147483650},
}

class HuaRuBattleEnvWrapper(EnvRunner):
    def __init__(self, 
                 env_args = None,
                 port_num = 0,
                 config = config
        ):
        agents = config['agents']
        ADDRESS = env_args['ADDRESS']
        address_xsim=ADDRESS['ip'] + ":" + str(ADDRESS['port'] + port_num)
        EnvRunner.__init__(self, agents, address_xsim, env_args['mode'])
        # MultiAgentEnv.__init__(self)
        self.env_name = env_args['env_name']
        self.n_agents = env_args['num_agents']
        self.map_name = env_args['map_name']

        # 动作空间
        self.action_space = [spaces.Discrete(15), spaces.Discrete(15), spaces.Discrete(15), spaces.Discrete(15), spaces.Discrete(15)]
        
        # # 多个动作的动作空间
        # self.action_space = [spaces.MultiDiscrete([10,6]), spaces.MultiDiscrete([10,6]), spaces.MultiDiscrete([10,6]), spaces.MultiDiscrete([10,6]), spaces.MultiDiscrete([10,6])]
        # 修改 self.get_avail_actions()，单个Agent的get_avail_actions函数输出为各个动作的mask直接相加的一位数组

        # 状态空间
        self.observation_space = [spaces.Box(low=0, high=1, shape=(105,)), spaces.Box(low=0, high=1, shape=(105,)), spaces.Box(low=0, high=1, shape=(105,)), spaces.Box(low=0, high=1, shape=(105,)), spaces.Box(low=0, high=1, shape=(105,))]
        #   全局的状态
        self.share_observation_space  = [[105 * 5, [5, 105]], [105 * 5, [5, 105]], [105 * 5, [5, 105]], [105 * 5, [5, 105]], [105 * 5, [5, 105]]]

        # 初始化智能体，红方智能体是用于转换仿真的指令；蓝方智能体适用于利用代码规则
        # 记录红蓝方的每帧的states
        self.ori_message = None
        self.episode_limit = env_args['episode_limit']
        self.red_death = None # 存入红方死亡的实体ID
        self.blue_death = None # 存入蓝方死亡实体ID

        # 记录每一帧每个飞机实体的位置
        self.red_agent_loc = None

        # 存储上一帧的原始战场态势数据
        self.last_ori_message = None
        # 存取本episode训练的其他信息
        self.infos = None
        self.battles_won = 0
        self.battles_game = 0
        self.forward_step = None
        # self.logger = logging.getLogger('Battle_5v5')
        # self.logger.setLevel(logging.DEBUG)

        self.done = None
        self.reward = None

    def reset(self, if_test=False, args=None, cur_time=None):
        """重置仿真环境, 返回初始帧obs"""
        for side, agent in self.agents.items():
            agent.reset()

        super().reset()

        # self.logger.console_logger.info('Reset, Restart Engine!')
        self.forward_step = 0
        self.red_death = []
        self.blue_death = []
        self.red_agent_loc = {'0': None, '1': None, '2': None, '3': None, '4': None}
        self.infos = {'BattleWon': 1} # 0蓝方胜利，1代表是平局，2红方胜利
        self.obs = None
        self.ori_message = None

        # 重置奖励函数
        self.reward = 0.
        # 重置训练环境

        self.ori_message = super().step([])  # 推动拿到第一帧的obs信息
        # self.done, flag_no = super().get_done(self.ori_message)

        cmd_list = []
        cmd_list.extend(self.agents["red"].make_init_cmd())

        self.last_ori_message = self.ori_message
        self.ori_message = super().step(cmd_list)

        # 获取RL Obs & State
        self.obs = self.get_obs()
        self.state = self.get_state()

        return self.obs, self.state, self.get_avail_actions() # list, np, np

    def step(self, actions=None, if_test=False):
        """
            逐帧推进仿真引擎, 发送actions, 返回obs
            :param actions: [9, ...] 是一个一维数组，规模为5，表示五个Agent的动作
        """
        self.forward_step += 1
        if_done = False
        # 准备将自己已经知道的所有信息一起发送给Agent
        parse_msg_ = {'agent_pre_loc': self.red_agent_loc,
                     'blue_info': BLUE_INFO,
                     'red_info': RED_INFO
        }
        # 生成蓝方Agents的cmd list
        blue_cmd_list = self.get_blue_cmd(self.agents["blue"])
        # 将网络输出动作转换为仿真执行指令,给self.red_agents
        cmd_list = self.agents["red"].make_actions(actions, parse_msg_) # Agents的仿真指令
        cmd_list.extend(blue_cmd_list)

        # print(f'当前生成的指令包括：{cmd_list}')
        self.last_ori_message = self.ori_message
        # 将仿真指令，发送回仿真，并且拿到下一帧的状态
        self.ori_message = super().step(cmd_list)
        # print(super().get_done(self.ori_message))
        self.done, flag_no = super().get_done(self.ori_message)
        # 解析得到下一帧的Agents Obs
        self.obs = self.get_obs()
        self.state = self.get_state()  # 更新全局state

        if self.done[0] or self.forward_step >= self.episode_limit:  # 或者步数超过episode_limit
            if not self.done[0]:
                # 超过了步数
                # 说明当前episode的训练已经结束
                # self.logger.console_logger.info('当前Episode训练已经结束(超过指定训练步数)！')
                self.infos['BattleWon'] = 1
                # self.logger.console_logger.info("平  局")
                # 结束的时候不需要kill env，只是下个episode开始的时候，reset()一下Agents就可以了
                if_done = True
                # self.close()
                # super().reset()

            else:
                # 说明当前episode的训练已经结束
                # self.logger.console_logger.info('当前Episode训练已经结束！')
                if (self.done[1] == 1 and self.done[2] == 0):
                    # self.logger.console_logger.info("红 方 胜!")
                    self.infos['BattleWon'] = 2
                elif (self.done[1] == 0 and self.done[2] == 1):
                    # self.logger.console_logger.info("蓝 方 胜!")
                    self.infos['BattleWon'] = 0
                else:
                    pass
                    # self.logger.console_logger.info("平  局")
                # 结束的时候不需要kill env，只是下个episode开始的时候，reset()一下Agents就可以了
                if_done = True
                # self.close()
                # super().reset()

        # cur_reward = self.get_reward_adjust(self.last_ori_message, self.ori_message, if_done, flag_no)
        cur_reward = self.get_simple_reward(self.last_ori_message, self.ori_message, if_done, flag_no)
        # cur_reward, if_dones = self.get_reward_independent(self.last_ori_message, self.ori_message)
        # if if_done:
        #     if_dones = [True for _ in range(self.n_agents)]
        # else:
        #     pass
        # return self.obs, self.state, cur_reward, if_dones, [self.infos for _ in range(self.n_agents)], self.get_avail_actions()
 
        # self.reward计算的是累积奖励，没有什么用, cur_reward才是当前帧的奖励
        # self.reward += cur_reward

        # if self.done[0]:
        #     self.logger.console_logger.info(f'本Episode获得的最终奖励是: {cur_reward}')
        #     self.logger.console_logger.info(f'本Episode获得的累积奖励: {self.reward}')

        # if self.forward_step % 100 == 0:
        #     self.logger.console_logger.info('Forward step: %d' % self.forward_step)
        #     self.logger.console_logger.info(f'Cur frame reward: {cur_reward}')
        #     self.logger.console_logger.info(f'Total reward: {self.reward}')

        #   return local_obs, global_state, rewards, dones, infos, self.get_avail_actions()
        if if_done:
            self.battles_game += 1
        if self.infos['BattleWon'] == 2:
                self.battles_won += 1
        infos = [{} for i in range(self.n_agents)]
        for i in range(self.n_agents):
            infos[i] = {
                "battles_won": self.battles_won,
                "battles_game": self.battles_game,
                "BattleWon": self.infos['BattleWon'],
            }
        return self.obs, self.state, [[cur_reward] for _ in range(self.n_agents)], [if_done for _ in range(self.n_agents)], infos, self.get_avail_actions()

    def get_blue_cmd(self, blue_agents):
        """获取蓝方当前的cmd_list"""
        cmd_list = None
        cmd_list = super()._agent_step(blue_agents, self.ori_message["sim_time"], self.ori_message["blue"])
        # print(f'蓝方Agents的命令', cmd_list)
        return cmd_list

    def get_reward(self, last_obs=None, next_obs=None, if_done=False):
        """
            每帧计算reward值，使用团队奖励，当敌方的某个战机被击落后，产生正奖励，当己方的一个飞机被击落，产生负奖励；
            到最后，即if_done==True，说明episode结束，开始进行结局奖励计算。
            :param last_obs: 上一帧的战场局势(原始数据形式)
            :param next_obs: 下一帧的战场局势(原始数据形式)
            :param if_done: 本episode是否已经结束？如果结束的话，要进行战场判定
        """
        reward = 0.
        """战场没有结束，只需要统计占损奖励"""
        # 1. 统计上一帧中，红方战机的数量 & 存在的导弹剩余数量
        last_red_agent_num = len(last_obs['red']['platforminfos'])
        last_red_weapon_num = 0.
        for entity in last_obs['red']['platforminfos']:
            last_red_weapon_num += entity['LeftWeapon']
        # 2. 统计下一帧中，红方战机的数量 & 存在的导弹剩余数量
        next_red_agent_num = len(next_obs['red']['platforminfos'])
        next_red_weapon_num = 0.
        for entity in next_obs['red']['platforminfos']:
            next_red_weapon_num += entity['LeftWeapon']
        # 3. 统计上一帧中，蓝方战机的数量 & 存在的导弹剩余数量
        last_blue_agent_num = len(last_obs['blue']['platforminfos'])
        last_blue_weapon_num = 0.
        for entity in next_obs['blue']['platforminfos']:
            last_blue_weapon_num += entity['LeftWeapon']
        # 4. 统计下一帧中，蓝方战机的数量 & 存在的导弹剩余数量
        next_blue_agent_num = len(next_obs['blue']['platforminfos'])
        next_blue_weapon_num = 0.
        for entity in next_obs['blue']['platforminfos']:
            next_blue_weapon_num += entity['LeftWeapon']

        # 计算占损战耗奖励
        reward = (-1.2) * (10 * (last_red_agent_num - next_red_agent_num)) - 0.5 * (last_red_weapon_num - next_red_weapon_num) \
                 + 1.2 * (10 * (last_blue_agent_num - next_blue_agent_num) + 0.5 * (last_blue_weapon_num - next_blue_weapon_num))
        # reward = (-1.5) * (10 * (last_red_agent_num - next_red_agent_num)) \
        #          + 2.0 * (50 * (last_blue_agent_num - next_blue_agent_num) + 0.5 * (last_blue_weapon_num - next_blue_weapon_num))

        # print(f'reward1: {reward}')
        # if last_red_agent_num - next_red_agent_num == 0:
        #     reward += 1  # 如果数量能保持现状，不死那么+5
        # print(f'reward2: {reward}')
        distance_reward = 0.
        if not if_done:
            # 设置一个奖励：计算每一帧红方每个飞机离中心点的距离，距离中心点越近得分越高
            for agent_order in range(len(self.ori_message['red']['platforminfos'])):
                cur_id = self.ori_message['red']['platforminfos'][agent_order]['ID']
                if cur_id in self.red_death:
                    distance_reward += 0.
                else:
                    # 当前的坐标
                    cur_x = self.ori_message['red']['platforminfos'][agent_order]['X']
                    cur_y = self.ori_message['red']['platforminfos'][agent_order]['Y']
                    cur_z = self.ori_message['red']['platforminfos'][agent_order]['Alt']
                    cur_distance = math.sqrt(cur_x ** 2 + cur_y ** 2 + cur_z ** 2)
                    # 上一个坐标
                    last_x = self.last_ori_message['red']['platforminfos'][agent_order]['X']
                    last_y = self.last_ori_message['red']['platforminfos'][agent_order]['Y']
                    last_z = self.last_ori_message['red']['platforminfos'][agent_order]['Alt']
                    last_distance = math.sqrt(last_x ** 2 + last_y ** 2 + last_z ** 2)

                distance_reward += 0.002 if (last_distance - cur_distance) >= 0 else -0.002

        else:
            """战场已经结束, 需要额外统计结局奖励"""
            if self.infos['BattleWon'] == 2: # 红方胜利
                reward += 600
            elif self.infos['BattleWon'] == 0:
                reward -= 600
            else:
                reward += 0

        reward += distance_reward

        return reward

    def get_reward_adjust(self, last_obs=None, next_obs=None, if_done=False, flag_no=False):
        """
            每帧计算reward值，使用团队奖励，当敌方的某个战机被击落后，产生正奖励，当己方的一个飞机被击落，产生负奖励；
            到最后，即if_done==True，说明episode结束，开始进行结局奖励计算。
            :param last_obs: 上一帧的战场局势(原始数据形式)
            :param next_obs: 下一帧的战场局势(原始数据形式)
            :param if_done: 本episode是否已经结束？如果结束的话，要进行战场判定
        """
        reward = 0.  # 最终奖励
        distance_reward = 0.  # 距离奖励, 距离中心点越近越胜利，越来越低
        attack_reward = 0.  # 攻击奖励，攻击敌方越多伤害越高，毁伤奖励应该是越来越高的
        damaged_reward = 0.  # 毁伤奖励， 自己造成的毁伤越多得到的负奖励越高

        """战场没有结束，只需要统计占损奖励"""
        # 1. 统计上一帧中，红方战机的数量 & 存在的导弹剩余数量
        last_red_agent_num = len(last_obs['red']['platforminfos'])
        last_red_weapon_num = 0.
        for entity in last_obs['red']['platforminfos']:
            last_red_weapon_num += entity['LeftWeapon']
        # 2. 统计下一帧中，红方战机的数量 & 存在的导弹剩余数量
        next_red_agent_num = len(next_obs['red']['platforminfos'])
        next_red_weapon_num = 0.
        for entity in next_obs['red']['platforminfos']:
            next_red_weapon_num += entity['LeftWeapon']
        # 3. 统计上一帧中，蓝方战机的数量 & 存在的导弹剩余数量
        last_blue_agent_num = len(last_obs['blue']['platforminfos'])
        last_blue_weapon_num = 0.
        for entity in next_obs['blue']['platforminfos']:
            last_blue_weapon_num += entity['LeftWeapon']
        # 4. 统计下一帧中，蓝方战机的数量 & 存在的导弹剩余数量
        next_blue_agent_num = len(next_obs['blue']['platforminfos'])
        next_blue_weapon_num = 0.
        for entity in next_obs['blue']['platforminfos']:
            next_blue_weapon_num += entity['LeftWeapon']

        # 计算伤害奖励
        if last_red_agent_num - next_red_agent_num > 0:
            # 说明上一轮红方飞机更多，红方飞机有伤亡
            if RED_INFO['0']['ID'] in self.red_death:
                # 红方有人机被打掉，红方输掉了
                damaged_reward -= 1200
            else:
                # 红方无人机被打掉，减小分
                damaged_reward = damaged_reward - (200 * (last_red_agent_num - next_red_agent_num))
                # damaged_reward -= 200 * (last_red_agent_num - next_red_agent_num)  # 地方是不是不对，如果同一帧，死掉两个实体呢？这里应该是 *
        else:
            # 说明红方飞机数量没有变化
            damaged_reward -= 0.

        reward += damaged_reward

        # 计算攻击奖励
        if last_blue_agent_num - next_blue_agent_num > 0:
            # 说明上一帧蓝方飞机要更多
            if BLUE_INFO['0']['ID'] in self.blue_death:
                attack_reward += 1200
            else:
                attack_reward += 200 * (last_blue_agent_num - next_blue_agent_num)
        else:
            # 说明蓝方飞机数量没有变化
            attack_reward += 0.

        reward += attack_reward

        # 计算距离奖励
        for agent_order in range(len(self.ori_message['red']['platforminfos'])):
            cur_id = self.ori_message['red']['platforminfos'][agent_order]['ID']
            if cur_id in self.red_death:
                distance_reward += 0.
            else:
                # 当前的坐标
                cur_x = self.ori_message['red']['platforminfos'][agent_order]['X']
                cur_y = self.ori_message['red']['platforminfos'][agent_order]['Y']
                # cur_z = self.ori_message['red']['platforminfos'][agent_order]['Alt']
                cur_distance = math.sqrt(cur_x ** 2 + cur_y ** 2)
                # 上一个坐标
                last_x = self.last_ori_message['red']['platforminfos'][agent_order]['X']
                last_y = self.last_ori_message['red']['platforminfos'][agent_order]['Y']
                # last_z = self.last_ori_message['red']['platforminfos'][agent_order]['Alt']
                last_distance = math.sqrt(last_x ** 2 + last_y ** 2)
                # 这个值是很大的
                # distance_reward += 1 if (last_distance - cur_distance) > 0 else -1
                distance_reward += 0.2 if (last_distance - cur_distance) > 0 else -0.2

        reward += (distance_reward * self.decay_coefficient(step=self.forward_step))  # 1.0是衰减系数

        return reward

    def get_reward_independent(self, last_obs=None, next_obs=None):
        reward = []
        if_dones = [False for _ in range(self.n_agents)]
        for i in range(self.n_agents):
            distance_reward = 0.  # 距离奖励, 距离中心点越近越胜利，越来越低
            attack_reward = 0.  # 攻击奖励，攻击敌方越多伤害越高，毁伤奖励应该是越来越高的
            agent_id = RED_INFO[str(i)]['ID']
            if agent_id in self.red_death:
                if_dones[i] = True
                if agent_id == 1:
                    reward.append([-200])
                else:
                    reward.append([-50])
            else:
                """战场没有结束，只需要统计占损奖励"""
                # 1. 统计上一帧中，蓝方战机的数量
                last_blue_agent_num = len(last_obs['blue']['platforminfos'])
                # 2. 统计下一帧中，蓝方战机的数量
                next_blue_agent_num = len(next_obs['blue']['platforminfos'])
                # 计算攻击奖励
                if last_blue_agent_num - next_blue_agent_num > 0:
                    # 说明上一帧蓝方飞机要更多
                    if BLUE_INFO['0']['ID'] in self.blue_death:
                        attack_reward += 200
                    else:
                        attack_reward += 50 * (last_blue_agent_num - next_blue_agent_num)
                else:
                    # 说明蓝方飞机数量没有变化
                    attack_reward += 0.

                # 计算距离奖励
                # 当前的坐标
                cur_x = next_obs['red']['platforminfos'][i]['X']
                cur_y = next_obs['red']['platforminfos'][i]['Y']
                # cur_z = self.ori_message['red']['platforminfos'][agent_order]['Alt']
                cur_distance = math.sqrt(cur_x ** 2 + cur_y ** 2)
                # 上一个坐标
                last_x = last_obs['red']['platforminfos'][i]['X']
                last_y = last_obs['red']['platforminfos'][i]['Y']
                # last_z = self.last_ori_message['red']['platforminfos'][agent_order]['Alt']
                last_distance = math.sqrt(last_x ** 2 + last_y ** 2)
                # 这个值是很大的
                distance_reward += 0.03 if (last_distance - cur_distance) > 0 else -0.03
                distance_reward = distance_reward * self.decay_coefficient(step=self.forward_step)
                reward.append([attack_reward+distance_reward])

        return reward, if_dones  

    def decay_coefficient(self, step=None):
        total_steps = self.episode_limit / 2
        decay_rate = -np.log(0.01) / total_steps  # 衰减速率，这里假设最终值为0.01
        return np.exp(-decay_rate * step)

    def get_simple_reward(self, last_obs=None, next_obs=None, if_done=False, flag_no=False):
        """获取一个简单奖励，胜利：1，平局：0，失败：-1

        :param last_obs: 上一局的态势
        :param next_obs: 本局的态势
        :param if_done: 是否本episode已经结束
        :param flag_no: 用来判断是否红方是通过长时间占领中心区域获胜的

        :return reward: float
        """
        reward = 0.
        if not if_done:
            pass
        else:
            if self.infos['BattleWon'] == 2: # 红方胜利
                reward = 1
                if flag_no:
                    reward = 0
            elif self.infos['BattleWon'] == 0:
                reward = -1
            else:
                reward = 0

        return reward

    def get_obs(self):
        """获取所有Agent的obs = [obs1, obs2, ..., obs5]"""
        agent_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agent_obs

    def get_obs_agent(self, agent_order):
        """获取每个Agent自己的obs"""
        # 已经有一个全局属性self.ori_message
        assert self.ori_message is not None
        # 通过解析这个原始数据构造每个Agent的obs，先通过这个id来判断是否已经死亡，就是判断字典中是否有这个实体
        self.cur_agent_name = RED_INFO[str(agent_order)]['Name']  # 当前Agent的名字
        self.cur_agent_id = RED_INFO[str(agent_order)]['ID']  # 当前Agent的id
        # 查找返回的信息中是否包含当前Agent，如果没有的话说明当前Agent已经死亡
        cur_agent_exist = any(filter(lambda x: x['ID'] == self.cur_agent_id, self.ori_message['red']['platforminfos']))
        if not cur_agent_exist:
            """说明这个Agent已经不存在了"""
            self.red_agent_loc[str(agent_order)] = None
            self.red_death.append(self.cur_agent_id)
        """当前Agent的obs"""
        agent_obs = self.parse_msg(str(agent_order), cur_agent_exist)
        return np.array(agent_obs)

    def parse_msg(self, agent_order, exist=None):
        """对原始态势信息进行解析处理，构造状态"""
        cur_agent_obs = []
        platforminfos_list = self.ori_message['red']['platforminfos']
        if exist:
            #   获取当前Agent在['platforminfos']中的索引
            cur_index = next(((i, item) for i, item in enumerate(platforminfos_list) if item['ID'] == RED_INFO[str(agent_order)]['ID']), (-1,None))[0]
            #   拿这个Agent的信息
            type_own = platforminfos_list[cur_index]['Type'] / 3.0
            x_own = (platforminfos_list[cur_index]['X'] + 150000) / 300000
            y_own = (platforminfos_list[cur_index]['Y'] + 150000) / 300000
            z_own = platforminfos_list[cur_index]['Alt'] / 15000
            heading_own = platforminfos_list[cur_index]['Heading'] / 360
            pitch_own = (platforminfos_list[cur_index]['Pitch'] + 90) / 180
            speed_own = (platforminfos_list[cur_index]['Speed'] - 100) / 300
            leftweapon_own = platforminfos_list[cur_index]['LeftWeapon'] / 4

            own_list = [type_own, x_own, y_own,z_own,
                        heading_own, pitch_own, speed_own, leftweapon_own]
            cur_agent_obs.extend(own_list)

            self.red_agent_loc[str(agent_order)] = {'X': x_own, 'Y': y_own, 'Z': z_own, 'heading': heading_own, 'pitch': pitch_own}

            # 拿其他队友的信息，对于某个Agent，它的队友固定为1，2，3，4顺序, 除了agent_order以外
            team_list = []
            for i in range(self.n_agents):
                if i != int(agent_order):
                    team_ID = RED_INFO[str(i)] ['ID']
                    # 查询当前队友是否还存活
                    team_exist = any(filter(lambda x: x['ID'] == team_ID, platforminfos_list))
                    if team_exist:
                        team_index = next(((j, item) for j, item in enumerate(platforminfos_list) if item['ID'] == team_ID), (-1,None))[0]
                        type_team = platforminfos_list[team_index]['Type'] / 3.0
                        x_team = (platforminfos_list[team_index]['X'] + 150000) / 300000 - x_own
                        y_team =(platforminfos_list[team_index]['Y'] + 150000) / 300000 - y_own
                        z_team = (platforminfos_list[team_index]['Alt'] / 15000) - z_own
                        speed_team = (platforminfos_list[team_index]['Speed'] - 100) / 300 - speed_own
                        leftweapon_team = platforminfos_list[team_index]['LeftWeapon'] / 4
                        team_ones = [type_team, x_team, y_team, z_team, speed_team, leftweapon_team]
                        team_list.extend(team_ones)
                    else:
                        # 说明这个队友已经死亡，那么只需要在对应位置上设为0即可，同时将当前队友ID写入红方死亡列表
                        self.red_death.append(team_ID) if team_ID not in self.red_death else None
                        team_ones = [0., 0., 0., 0., 0., 0.]
                        team_list.extend(team_ones)
            cur_agent_obs.extend(team_list)

            #   拿敌人的状态信息
            trackinfos_list = self.ori_message['red']['trackinfos']
            enemy_list = []
            for i in range(5):
                enemy_ID = BLUE_INFO[str(i)]['ID']
                enemy_exist = any(filter(lambda x: x['ID'] == enemy_ID, trackinfos_list))
                if enemy_exist:
                    # 找到这个敌人在trackinfos_list中的索引位置
                    enemy_index = next(((j, item) for j, item in enumerate(trackinfos_list) if item['ID'] == enemy_ID), (-1, None))[0]
                    type_enemy = trackinfos_list[enemy_index]['Type'] / 3.0
                    x_enemy = (trackinfos_list[enemy_index]['X'] + 150000) / 300000 - x_own
                    y_enemy = (trackinfos_list[enemy_index]['Y'] + 150000) / 300000 - y_own
                    z_enemy = trackinfos_list[enemy_index]['Alt'] / 15000 - z_own
                    speed_enemy = (trackinfos_list[enemy_index]['Speed'] - 100) / 300 - speed_own
                    enemy_ones = [type_enemy, x_enemy, y_enemy, z_enemy, speed_enemy]
                    enemy_list.extend(enemy_ones)
                else:
                    # 说明这个敌人已经死掉了
                    self.blue_death.append(enemy_ID)
                    enemy_ones = [0., 0., 0., 0., 0.]
                    enemy_list.extend(enemy_ones)
            cur_agent_obs.extend(enemy_list)

            #   拿敌人的导弹信息，这里只取对自己有威胁的导弹
            missileinfos_list = self.ori_message['red']['missileinfos']
            cur_fire = []
            for i in range(12):
                # 一共是12枚导弹，分别查看这12枚导弹是否出现了
                cur_blue_fire_exist = any(filter(lambda x: x['Name'] == BLUE_FIRE_INFO[str(i)]['Name'], missileinfos_list))
                try:
                    cur_blue_fire_index = next(((j, item) for j, item in enumerate(missileinfos_list) if item['Name'] == BLUE_FIRE_INFO[str(i)]['Name']), (-1, None))[0]
                except:
                    print(missileinfos_list)
                    print(f'curr index : {i}')
                # 如果出现了，查看是否锁定了自己？
                if cur_blue_fire_exist:
                    # 说明这枚弹已经出现了，拿到这枚弹的信息
                    cur_blue_fire_info = missileinfos_list[cur_blue_fire_index]
                    if cur_blue_fire_info['EngageTargetID'] == RED_INFO[str(agent_order)]['ID']:
                        # 锁定了自己
                        cur_blue_fire_Z = cur_blue_fire_info['Alt']
                        cur_fire.extend([(cur_blue_fire_info['X'] + 150000) / 300000 - x_own,
                                         (cur_blue_fire_info['Y'] + 150000) / 300000 - y_own,
                                         (cur_blue_fire_Z - 2000) / 30000 - z_own,
                                         (cur_blue_fire_info['Speed'] - 400) / 1000 - speed_own]
                        )
                    else:
                        # 未锁定自己
                        cur_fire.extend([0., 0., 0., 0.])
                else:
                    # 说明这枚弹没有出现
                    cur_fire.extend([0., 0., 0., 0.])

            cur_agent_obs.extend(cur_fire)
            # 对每一个obs_value都进行归一化
            cur_agent_obs = cur_agent_obs - np.mean(cur_agent_obs)
            cur_agent_obs = cur_agent_obs / np.max(np.abs(cur_agent_obs))

        else:
            """当前Agent已经死亡"""
            cur_agent_obs.extend([0.0 for _ in range(105)])

        return cur_agent_obs

    def get_global_state(self):
        return [np.array(self.obs).flatten() for _ in range(self.n_agents)]

    def get_state(self):
        """获取全局状态"""
        return self.get_global_state()

    def get_avail_actions(self):
        """
            获得可执行动作列表
        """
        np1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=object)  # 9个方向
        np2 = np.array([1, 1, 1, 1, 1, 1], dtype=object)  # 5个攻击动作+1个不采用任何动作
        avai_cow = np.array([np1, np2], dtype=object)
        avai_mask = np.tile(avai_cow, (5, 1))
        # 分别返回 agents的actions1，actions2
        avai_dict = []
        for i in range(5):
            """分别遍历五个Agent给出available mask"""
            # 先判断这个Agent是否已经死亡
            if_death = RED_INFO[str(i)]['ID'] in self.red_death  # 查看这个Agent的ID是否在死亡列表self.red_death中
            if if_death:
                avai_mask[i][0] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=object)
                avai_mask[i][1] = np.array([0, 0, 0, 0, 0, 1], dtype=object)
            else:
                # 如果没有死亡，那么移动就可以全部是1,需要判断能不能攻击具体到某个敌方，要进行弹药数量判断和距离判断
                # cur_index = next(((j, item) for j, item in enumerate(self.ori_message['red']['platforminfos']) if RED_INFO[str(i)]['ID'] == self.ori_message['red']['platforminfos'][j])['ID'], (-1, None))[0]
                # next的正确用法。
                cur_index = next(((j, item) for j, item in enumerate(self.ori_message['red']['platforminfos']) if
                                  RED_INFO[str(i)]['ID'] == item.get('ID')), (-1, None))[0]

                left_weapon_num = self.ori_message['red']['platforminfos'][cur_index]['LeftWeapon']  # 先判断有没有剩余弹药
                cur_red_agent_loc = [self.ori_message['red']['platforminfos'][cur_index]['X'],
                                     self.ori_message['red']['platforminfos'][cur_index]['Y'],
                                     self.ori_message['red']['platforminfos'][cur_index]['Alt'],
                                     ]

                # print(f'agent ID: {RED_INFO[str(i)]["ID"]}, Leftweapon: {left_weapon_num}')
                if left_weapon_num == 0:
                    avai_mask[i][1] = np.array([0, 0, 0, 0, 0, 1], dtype=object)
                # else:
                elif left_weapon_num > 0:
                    # 进行地方距离判断
                    # avai_mask[i][1] 每个index上对应着敌方固定的实体
                    cur_ava = np.array([],dtype=object)
                    # 分别计算敌方实体的距离
                    for j in range(5):
                        # 查找蓝方实体的位置，要先判断这个蓝方是不是已经死掉了
                        blue_ID = BLUE_INFO[str(j)]['ID']
                        if blue_ID in self.blue_death:
                            cur_ava = np.append(cur_ava, 0)
                        else:
                            # 这个蓝方实体还没有死掉
                            enemy_index = next(((i_, item) for i_, item in enumerate(self.ori_message['red']['trackinfos']) if
                                 item.get('ID') == blue_ID), (-1, None))[0]
                            enemy_infos = self.ori_message['red']['trackinfos'][enemy_index]
                            enemy_loc = [
                                enemy_infos['X'],
                                enemy_infos['Y'],
                                enemy_infos['Alt']
                            ]
                            # 判断距离
                            if self.distance_computation(cur_red_agent_loc, enemy_loc) <= 80000:
                                cur_ava = np.append(cur_ava, 1)
                            else:
                                cur_ava = np.append(cur_ava, 0)

                    # print(f'original cur_ava: {cur_ava}')
                    new_cur_ava = np.zeros_like(cur_ava)
                    new_cur_ava[(cur_ava == 1)] = 1
                    # print(f'cur_ava', new_cur_ava)
                    # cur_ava.append(1)  # 最后一维是不执行任何攻击工作
                    cur_ava = np.append(new_cur_ava, 1)
                    # cur_ava = np.append(cur_ava, 1)
                    avai_mask[i][1] = cur_ava

        for i in range(5):
            avai_dict.append(list(avai_mask[i][0]) + list(avai_mask[i][1]))

        return avai_dict

    def distance_computation(self, point_1=None, point_2=None):
        """
            计算两个点之间的距离
            :param point_1 第一个点的 [X，Y，Z]
            :param point_2 第二个点的 [X，Y，Z]

            :return distance: int
        """
        return math.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2 + (point_1[2] - point_2[2])**2)

    def close(self):
        """关闭仿真引擎"""
        self._end()

    def seed(self, seed):
        self._seed = seed

    def render(self):
        """渲染视频"""
        pass

    def save_replay(self):
        pass

    def llh_to_ecef(self, lat, lon, alt) -> {}:
        # WGS84椭球参数
        a = 6378137.0  # 地球半径
        e = 8.1819190842622e-2  # 偏心率
        # 将度转换为弧度
        # lat = math.radians(lat)
        # lon = math.radians(lon)
        N = a / math.sqrt(1 - e ** 2 * math.sin(lat) ** 2)
        X = (N + alt) * math.cos(lat) * math.cos(lon)
        Y = (N + alt) * math.cos(lat) * math.sin(lon)
        Z = ((1 - e ** 2) * N + alt) * math.sin(lat)

        return {'X': X, 'Y': Y, 'Z': Z}

    def get_total_actions(self):
        pass

    def get_obs_size(self):
        return self.observation_space

    def get_state_size(self):
        return self.n_agents * self.get_obs_size()

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(), # 620
            "obs_shape": self.get_obs_size(), # 124
            "n_actions": self.get_total_actions(), # 15
            "n_agents": self.n_agents, # 5
            "episode_limit": self.episode_limit
        }

        return env_info

    def render(self):
        pass

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "win_rate": self.battles_won / self.battles_game,
        }
        return stats
