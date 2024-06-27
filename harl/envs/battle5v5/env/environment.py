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
from harl.envs.battle5v5.config import config  #, POOL_NUM, ISHOST, XSIM_NUM, ADDRESS
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
    '0': {'Name': '空空导弹_1(蓝有人机_武器系统_1)', 'ID': 2147483668},
    '1': {'Name': '空空导弹_2(蓝有人机_武器系统_1)', 'ID': 2147483669},
    '2': {'Name': '空空导弹_3(蓝有人机_武器系统_1)', 'ID': 2147483670},
    '3': {'Name': '空空导弹_4(蓝有人机_武器系统_1)', 'ID': 2147483671},
    '4': {'Name': '空空导弹_1(蓝无人机1_武器系统_1)', 'ID': 2147483657},
    '5': {'Name': '空空导弹_2(蓝无人机1_武器系统_1)', 'ID': 2147483659},
    '6': {'Name': '空空导弹_1(蓝无人机2_武器系统_1)', 'ID': 2147483652},
    '7': {'Name': '空空导弹_2(蓝无人机2_武器系统_1)', 'ID': 2147483654},
    '8': {'Name': '空空导弹_1(蓝无人机3_武器系统_1)', 'ID': 2147483660},
    '9': {'Name': '空空导弹_2(蓝无人机3_武器系统_1)', 'ID': 2147483662},
    '10': {'Name': '空空导弹_1(蓝无人机4_武器系统_1)', 'ID': 2147483648},
    '11': {'Name': '空空导弹_2(蓝无人机4_武器系统_1)', 'ID': 2147483650},
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
        self.action_space = [spaces.Discrete(17), spaces.Discrete(17), spaces.Discrete(17), spaces.Discrete(17), spaces.Discrete(17)]
        
        # # 多个动作的动作空间
        # self.action_space = [spaces.MultiDiscrete([10,6]), spaces.MultiDiscrete([10,6]), spaces.MultiDiscrete([10,6]), spaces.MultiDiscrete([10,6]), spaces.MultiDiscrete([10,6])]
        # 修改 self.get_avail_actions()，单个Agent的get_avail_actions函数输出为各个动作的mask直接相加的一位数组

        # 状态空间
        self.observation_space = [spaces.Box(low=0, high=1, shape=(105,)), spaces.Box(low=0, high=1, shape=(105,)), spaces.Box(low=0, high=1, shape=(105,)), spaces.Box(low=0, high=1, shape=(105,)), spaces.Box(low=0, high=1, shape=(105,))]
        #   全局的状态
        self.share_observation_space  = [[105 * 5, [5, 105]]]

        # 初始化智能体，红方智能体是用于转换仿真的指令；蓝方智能体适用于利用代码规则
        # 记录红蓝方的每帧的states
        self.ori_message = None
        self.episode_limit = env_args['episode_limit']
        self.red_death = None # 存入红方死亡的实体ID
        self.blue_death = None # 存入蓝方死亡实体ID

        # 记录每一帧每个飞机实体的位置
        self.red_agent_loc = None
        self.agents_speed = None

        # 存储上一帧的原始战场态势数据
        self.last_ori_message = None
        # 存取本episode训练的其他信息
        self.infos = None
        self.battles_won = 0
        self.battles_game = 0
        self.forward_step = None
        self.already_done = False
        self.shoot_interval = None  # 每个Agent的设计间隔
        self.shoot_interval_step = 0  # 50000米发射，导弹1000m/s

        self.done = None
        self.reward = None

        # 模拟修改想定步长
        self.change_time_step = 15  # 每次做出动作之后，要空推15帧

    def reset(self, if_test=False, args=None, cur_time=None):
        """重置仿真环境, 返回初始帧obs"""
        for side, agent in self.agents.items():
            agent.reset()
        super().reset()
        self.ori_message = super().step([])

        # self.logger.console_logger.info('Reset, Restart Engine!')
        self.forward_step = 0
        self.red_death = []
        self.blue_death = []
        self.red_agent_loc = {'0': None, '1': None, '2': None, '3': None, '4': None}
        self.agents_speed = {'0': None, '1': None, '2': None, '3': None, '4': None}
        self.infos = {'BattleWon': 1} # 0蓝方胜利，1代表是平局，2红方胜利
        self.obs = None
        self.shoot_interval = np.zeros([self.n_agents, 5])

        # 重置奖励函数
        self.reward = 0.
        self.already_done = False

        # 重置训练环境
        if self.ori_message["sim_time"] == 0.0:
            self.ori_message = super().step([])  # 推动拿到第一帧的obs信息

        self.obs = self.get_obs()
        self.state = self.get_state()
        # 准备将自己已经知道的所有信息一起发送给Agent
        parse_msg_ = {'agent_pre_loc': self.red_agent_loc,
                     'blue_info': BLUE_INFO,
                     'red_info': RED_INFO,
                     'sim_time': self.ori_message['sim_time'],
                     'agent_speed': self.agents_speed # 记录当前帧每个智能体的速度，如果已经死亡，则速度为-1
        }

        cmd_list = []
        cmd_list.extend(self.agents["red"].make_init_cmd())
        cmd_list.extend(self.get_blue_cmd(self.agents["blue"]))

        self.last_ori_message = self.ori_message
        self.ori_message = super().step(cmd_list)

        ## 将前150步的规则操作都移到reset()中，不输入到算法中，减小探索空间
        # 前150将动作为0传给alo_agent
        rule_actions = [[0], [0], [0], [0], [0]]  # 加一个强规则，前150帧像前飞，减小探索空间
        while self.ori_message["sim_time"] < 200:
            cmd_list = []
            if self.ori_message["sim_time"] % 20 == 0:
                cmd_list.extend(self.agents["red"].make_actions(rule_actions, parse_msg_))
            else:
                pass
            cmd_list.extend(self.get_blue_cmd(self.agents["blue"]))
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
        for index in range(len(actions)):
            # 拿取每个agent的攻击动作
            cur_shoot_action = int(actions[index][0])
            if cur_shoot_action >= 9 and cur_shoot_action < 14:  # 射击间隔
                # self.shoot_interval[index][cur_shoot_action] = self.shoot_interval_step
                # 修改为，朝着所有敌方发射都存在间隔
                for shoot_target in range(5):
                    self.shoot_interval[index][shoot_target] = self.shoot_interval_step
        non_zero_indices = self.shoot_interval != 0
        self.shoot_interval[non_zero_indices] -= 1
        self.forward_step += 1

        if_done = False
        # 准备将自己已经知道的所有信息一起发送给Agent
        parse_msg_ = {'agent_pre_loc': self.red_agent_loc,
                     'blue_info': BLUE_INFO,
                     'red_info': RED_INFO,
                     'sim_time': self.ori_message['sim_time'],
                     'agent_speed': self.agents_speed # 记录当前帧每个智能体的速度，如果已经死亡，则速度为-1
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

        if not self.done[0]:
            ## 如果修改想定步长，则是15帧之内空推，不产生任务指令
            forward_step = 1
            while forward_step % self.change_time_step != 0:
                blue_cmd_list = self.get_blue_cmd(self.agents["blue"])
                cmd_list = []
                cmd_list.extend(blue_cmd_list)
                self.ori_message = super().step(cmd_list)
                self.done, flag_no = super().get_done(self.ori_message)
                if self.done[0]:
                    break
                forward_step+=1

        # 解析得到下一帧的Agents Obs
        self.obs = self.get_obs()
        self.state = self.get_state()  # 更新全局state

        if self.done[0] or self.forward_step >= self.episode_limit:  # 或者步数超过episode_limit
            if not self.done[0]:
                # 超过了步数
                # 说明当前episode的训练已经结束
                self.infos['BattleWon'] = 1  # 平 局
                if_done = True
                # self.close()
                # super().reset()
            else:
                # 说明当前episode的训练已经结束
                if (self.done[1] == 1 and self.done[2] == 0):
                    self.infos['BattleWon'] = 2  # 红 方 胜
                elif (self.done[1] == 0 and self.done[2] == 1):
                    self.infos['BattleWon'] = 0  # 蓝 方 胜
                else:
                    pass
                if_done = True
                # self.close()
                # super().reset()

        # cur_reward = self.get_reward_adjust(self.last_ori_message, self.ori_message, if_done, flag_no)
        cur_reward = self.get_reward_new(self.last_ori_message, self.ori_message, if_done, flag_no)
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

    def get_reward_new(self, last_obs=None, next_obs=None, if_done=False, flag_no=False):

        reward = 0.  # 最终奖励
        distance_reward = 0.  # 距离奖励, 距离中心点越近越胜利，越来越低
        attack_reward = 0.  # 攻击奖励，攻击敌方越多伤害越高，毁伤奖励应该是越来越高的
        damaged_reward = 0.  # 毁伤奖励， 自己造成的毁伤越多得到的负奖励越高
        locked_reward = 0.   # 锁定奖励，锁定敌方飞机，获取奖励
        protect_reward = 0.  # 保护红方有人机

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
        for entity in last_obs['blue']['platforminfos']:
            last_blue_weapon_num += entity['LeftWeapon']
        # 4. 统计下一帧中，蓝方战机的数量 & 存在的导弹剩余数量
        next_blue_agent_num = len(next_obs['blue']['platforminfos'])
        next_blue_weapon_num = 0.
        for entity in next_obs['blue']['platforminfos']:
            next_blue_weapon_num += entity['LeftWeapon']

        # # 计算锁定奖励
        # last_red_locked_num = 0
        # for entity in last_obs['red']['platforminfos']:
        #     last_red_locked_num += 1 if entity['IsLocked'] else 0
        # next_red_locked_num = 0
        # for entity in next_obs['red']['platforminfos']:
        #     next_red_locked_num += 1 if entity['IsLocked'] else 0

        # last_blue_locked_num = 0
        # for entity in last_obs['blue']['platforminfos']:
        #     last_blue_locked_num +=1 if entity['IsLocked'] else 0

        # next_blue_locked_num = 0
        # for entity in next_obs['blue']['platforminfos']:
        #     next_blue_locked_num += 1 if entity['IsLocked'] else 0

        # if last_red_locked_num > 0:
        #     if next_red_locked_num > 0:
        #         locked_reward -= 1
        #     else:
        #         locked_reward += 10

        # if last_blue_locked_num > 0:
        #     if next_blue_locked_num > 0:
        #         locked_reward += 1
        #     else:
        #         locked_reward -= 10

        # # new
        # if last_blue_locked_num - next_blue_locked_num > 0:
        #     locked_reward -= 2
        #     self.logger.console_logger.info(f'蓝方脱离锁定! -2')
        # else:
        #     locked_reward += 1
        #     self.logger.console_logger.info(f'蓝方被持续锁定! +1')

        reward += locked_reward
        # # 计算保护奖励
        # last_red_x_positions = []
        # last_red_y_positions = []
        # last_red_av_x = None
        # last_red_av_y = None
        # last_red_av_x_protected = False
        # last_red_av_y_protected = False
        # last_red_av_protected = False

        # for entity in last_obs['red']['platforminfos']:
        #     if entity['ID'] == 1:
        #         last_red_av_x = entity['X']
        #         last_red_av_y = entity['Y']
        #     else:
        #         last_red_x_positions.append(entity['X'])
        #         last_red_y_positions.append(entity['Y'])

        # if last_red_av_x and last_red_av_y and len(last_red_x_positions) > 0 and len(last_red_y_positions):
        #     if last_red_av_x < max(last_red_x_positions) and last_red_av_x > min(last_red_x_positions):
        #         last_red_av_x_protected = True
        #     if last_red_av_y < max(last_red_y_positions) and last_red_av_y > min(last_red_y_positions):
        #         last_red_av_y_protected = True

        #     if last_red_av_x_protected and last_red_av_y_protected:
        #         last_red_av_protected = True

        # next_red_x_positions = []
        # next_red_y_positions = []
        # next_red_av_x = None
        # next_red_av_y = None
        # next_red_av_x_protected = False
        # next_red_av_y_protected = False
        # next_red_av_protected = False

        # for entity in next_obs['red']['platforminfos']:
        #     if entity['ID'] == 1:
        #         next_red_av_x = entity['X']
        #         next_red_av_y = entity['Y']
        #     else:
        #         next_red_x_positions.append(entity['X'])
        #         next_red_y_positions.append(entity['Y'])

        # if next_red_av_x and next_red_av_y and len(next_red_x_positions) > 0 and len(next_red_y_positions) > 0:
        #     if next_red_av_x < max(next_red_x_positions) and next_red_av_x > min(next_red_x_positions):
        #         next_red_av_x_protected = True
        #     if next_red_av_y < max(next_red_y_positions) and next_red_av_y > min(next_red_y_positions):
        #         next_red_av_y_protected = True

        #     if next_red_av_x_protected and next_red_av_y_protected:
        #         next_red_av_protected = True

        # if not last_red_av_protected and next_red_av_protected:
        #     protect_reward += 20
        #     self.logger.console_logger.info(f'保护红方有人机! +20')
        # if last_red_av_protected and not next_red_av_protected:
        #     protect_reward -= 50
        #     self.logger.console_logger.info(f'红方有人机脱离保护! -50')
        # if last_red_av_protected and next_red_av_protected:
        #     protect_reward += 0.1
        #     # self.logger.console_logger.info(f'持续保护红方有人机! +0.1')
        # if not last_red_av_protected and not next_red_av_protected:
        #     protect_reward -= 0.1
        #     # self.logger.console_logger.info(f'没有保护红方有人机! -0.1')

        # next_x_y_too_far = 0
        # if next_red_av_x:
        #     next_x_y_too_far += (np.abs(np.array(next_red_x_positions) - next_red_av_x) > 75000).sum()
        # if next_red_av_y:
        #     next_x_y_too_far += (np.abs(np.array(next_red_y_positions) - next_red_av_y) > 75000).sum()

        # if next_x_y_too_far > 0:
        #     protect_reward -= 0.1 * next_x_y_too_far
        #     self.logger.console_logger.info(f'无人机离有人机过远！{-0.5 * next_x_y_too_far}')
        # else:
        #     self.logger.console_logger.info(f'所有无人机不离有人机过远！')

        # reward += protect_reward

        # 计算伤害奖励
        if last_red_agent_num - next_red_agent_num > 0:
            # 说明上一轮红方飞机更多，红方飞机有伤亡
            if RED_INFO['0']['ID'] in self.red_death:
                # 红方有人机被打掉，红方输掉了
                # damaged_reward -= 600
                pass # 算在最终输赢奖励里
            else:
                # 红方无人机被打掉，减小分
                damaged_reward -= 200 * (last_red_agent_num - next_red_agent_num)
                # damaged_reward -= 200 * (last_red_agent_num - next_red_agent_num)  # 地方是不是不对，如果同一帧，死掉两个实体呢？这里应该是 *
        if last_blue_weapon_num - next_blue_weapon_num > 0:
            # 说明红方飞机数量没有变化, 躲蛋成功获得，奖励
            damaged_reward += 100 * (last_blue_weapon_num - next_blue_weapon_num)
            # self.logger.console_logger.info(f'红方躲蛋成功! +{damaged_reward}')

        reward += damaged_reward

        # 计算攻击奖励
        if last_blue_agent_num - next_blue_agent_num > 0:
            # 说明上一帧蓝方飞机要更多
            if BLUE_INFO['0']['ID'] in self.blue_death:
                # 蓝方有人机被打掉，红方胜利
                # attack_reward += 600
                pass # 算在最终输赢奖励里
            else:
                attack_reward += 200 * (last_blue_agent_num - next_blue_agent_num)
        if last_red_weapon_num - next_red_weapon_num > 0:
            # 说明蓝方飞机数量没有变化， 被蓝方躲避，浪费弹药
            attack_reward -= 100 * (last_red_weapon_num - next_red_weapon_num)
            # self.logger.console_logger.info(f'蓝方躲蛋成功! 红方浪费弹药 {attack_reward}')

        reward += attack_reward

        # 计算距离奖励
        for agent_order in range(len(self.ori_message['red']['platforminfos'])):
            cur_id = self.ori_message['red']['platforminfos'][agent_order]['ID']
            if cur_id in self.red_death:
                distance_reward += 0.
            else:
                # 当前的坐标
                cur_x = next_obs['red']['platforminfos'][agent_order]['X']
                cur_y = next_obs['red']['platforminfos'][agent_order]['Y']
                cur_distance = math.sqrt((cur_x / 10000) ** 2 + (cur_y / 10000) ** 2)
                # 上一个坐标
                last_x = last_obs['red']['platforminfos'][agent_order]['X']
                last_y = last_obs['red']['platforminfos'][agent_order]['Y']
                last_distance = math.sqrt((last_x / 10000) ** 2 + (last_y / 10000) ** 2)
                # 这个值是很大的
                # distance_reward += 1 if (last_distance - cur_distance) > 0 else -1
                distance_reward += 0.5 if (last_distance - cur_distance) > 0 else -0.5

        reward += (distance_reward * self.decay_coefficient(step=self.forward_step))  # 1.0是衰减系数

        # # 计算边缘惩罚
        # border_reward = 0
        # if next_obs['sim_time'] > 180:
        #     for agent_order in range(len(self.ori_message['red']['platforminfos'])):
        #         cur_id = self.ori_message['red']['platforminfos'][agent_order]['ID']
        #         if cur_id in self.red_death:
        #             border_reward += 0.
        #         else:
        #             # 当前的坐标
        #             cur_x = next_obs['red']['platforminfos'][agent_order]['X']
        #             cur_y = next_obs['red']['platforminfos'][agent_order]['Y']
        #             if abs(cur_x) > 120000.0:
        #                 if 150000 - abs(cur_x) <= 0:
        #                     print(f"红方飞机{agent_order}自杀了！")
        #                 # print(f"红方飞机{agent_order}距离x轴飞出边界还剩{150000 - abs(cur_x)}！")
        #                 border_reward += 3 * np.exp(1 - (abs(cur_x) - 120000) / 30000)
        #             if abs(cur_y) > 120000.0:
        #                 if 150000 - abs(cur_y) <= 0:
        #                     print(f"红方飞机{agent_order}自杀了！")
        #                 # print(f"红方飞机{agent_order}距离y轴飞出边界还剩{150000 - abs(cur_y)}！")
        #                 border_reward += 3 * np.exp(1 - (abs(cur_y) - 120000) / 30000)

        # reward -= border_reward

        # 胜负奖励
        winner = 0
        if if_done:
            if self.infos['BattleWon'] == 2:
                winner = 1
            elif self.infos['BattleWon'] == 0:
                winner = 2
        elif next_red_weapon_num == 0: # 无弹惩罚 （无弹时已经输了）
            missile_locking_human = None
            for missile in next_obs['red']['missileinfos']:
                if missile['Identification'] == '红方' and missile['EngageTargetID'] == BLUE_INFO['0']['ID']: # 场上没有正在追击敌方有人机的导弹 已经输了
                    missile_locking_human = missile
                    break
            if missile_locking_human is None:
                winner = 2
        elif next_blue_weapon_num == 0: # 无弹提前胜利
            missile_locking_human = None
            for missile in next_obs['blue']['missileinfos']:
                if missile['Identification'] == '蓝方' and missile['EngageTargetID'] == RED_INFO['0']['ID']:
                    missile_locking_human = missile
                    break
            if missile_locking_human is None:
                winner = 1

        if not self.already_done:
            if winner == 1: # 红方胜
                self.already_done = True
                reward += 2400
            elif winner == 2: # 蓝方胜 (极端情况： 红射4发导弹，击落蓝方全部无人机，蓝射光全部导弹，击落红方有人机，此时红方差值200*4+100*8=1600)
                self.already_done = True
                reward -= 2400
            else: # 尚未结束，或平局
                reward += 0

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
        total_steps = self.episode_limit
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
            self.agents_speed[str(agent_order)] = -1    # 此处如此操作是否欠妥
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
            heading_own = platforminfos_list[cur_index]['Heading'] / (2*math.pi) # [0, 2pi]
            pitch_own = (platforminfos_list[cur_index]['Pitch'] + (math.pi/2)) / (math.pi)
            speed_own = (platforminfos_list[cur_index]['Speed'] - 100) / 300
            leftweapon_own = platforminfos_list[cur_index]['LeftWeapon'] / 4

            own_list = [type_own, x_own, y_own,z_own,
                        heading_own, pitch_own, speed_own, leftweapon_own]
            cur_agent_obs.extend(own_list)

            self.red_agent_loc[str(agent_order)] = {'X': platforminfos_list[cur_index]['X'], 'Y': platforminfos_list[cur_index]['Y'], 
                                                    'Z': platforminfos_list[cur_index]['Alt'], 'heading': platforminfos_list[cur_index]['Heading'], 'pitch': platforminfos_list[cur_index]['Pitch']}
            self.agents_speed[str(agent_order)] = platforminfos_list[cur_index]['Speed'] # 存取当前速度

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
        from copy import deepcopy
        # np1 = np.array([1 for _ in range(20)], dtype=object)  # 9个方向
        # # np1[10:] = 0  # 不采用跟随动作
        # np2 = np.array([1 for _ in range(6)], dtype=object)  # 5个攻击动作+1个不采用攻击动作
        # np3 = np.array([1 for _ in range(3)], dtype=object)
        # avai_cow = np.array([np1, np2, np3], dtype=object)
        avai_cow = np.array([1 for _ in range(17)])
        # avai_mask = np.tile(avai_cow, (5, 1)) # 出现问题，会共享内存
        avai_mask = np.vstack([deepcopy(avai_cow) for _ in range(5)])
        # 分别返回 agents的actions1，actions2
        # avai_dict = {'0': [], '1': [], '2': []}
        avai_dict = []
        for i in range(5):
            """分别遍历五个Agent给出available mask"""
            # 先判断这个Agent是否已经死亡
            if_death = RED_INFO[str(i)]['ID'] in self.red_death  # 查看这个Agent的ID是否在死亡列表self.red_death中
            if if_death:
                # avai_mask[i][0] = np.array([0 for _ in range(20)], dtype=object)
                # avai_mask[i][0][9] = 1  # 10 动作是不执行任何指令，任何情况下都是1
                # avai_mask[i][1] = np.array([0, 0, 0, 0, 0, 1], dtype=object)
                # avai_mask[i][2] = np.array([0, 0, 1], dtype=object)
                avai_mask[i] = np.array([0 for _ in range(17)], dtype=object)
                avai_mask[i][16] = 1
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
                # add new 跟随指令，当选择跟随的实体已经死亡时，则不能跟随；只要自己没死，则可以跟随自己(区域巡逻)
                # for follow_order in range(10):
                #     if follow_order == i: 
                #         # 说明跟随的是自己，而且自己没有死亡
                #         # 如果巡逻的范围加上自己现在的坐标超出了边界范围
                #         area_length_limit = abs(1000) / 2.0 + abs(self.ori_message['red']['platforminfos'][cur_index]['X'])
                #         area_width_limit = abs(1000) / 2.0 + abs(self.ori_message['red']['platforminfos'][cur_index]['Y'])
                #         if area_length_limit <= 150000 and area_width_limit <= 150000:
                #             avai_mask[i][follow_order+16] = 1
                #         else:
                #             # 如果巡逻的范围加上自己现在的坐标超出了边界范围
                #             avai_mask[i][follow_order+16] = 0
                #         # 一直报区域巡逻边界出界，索性把区域巡逻全部设为0
                #         pass
                #     else:
                #         # 跟随的不是自己，要判断这个实体是否已经死亡
                #         if follow_order >=0 and follow_order < 5:
                #             # 红方队友
                #             team_id = RED_INFO[str(follow_order)]["ID"]
                #             avai_mask[i][follow_order+16] = 1 if team_id not in self.red_death else 0
                #         else:
                #             # 蓝方敌人
                #             enemy_id = BLUE_INFO[str(follow_order-5)]["ID"]
                #             avai_mask[i][follow_order+16] = 1 if enemy_id not in self.blue_death else 0
                #         # print(f'当前跟随的对象不是自己，是{follow_order}')
                #         pass
                
                if left_weapon_num == 0:
                    # avai_mask[i][1] = np.array([0, 0, 0, 0, 0, 1], dtype=object)
                    avai_mask[i][9:14] = 0  # 
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

                    # # 加入射击间隔
                    agent_shoot_interval = self.shoot_interval[i]
                    agent_shoot_legal = np.ones_like(agent_shoot_interval)
                    agent_shoot_legal[agent_shoot_interval != 0] = 0
                    # # print(f'agent_shoot_legal', agent_shoot_legal)
                    # # print(f'original cur_ava: {cur_ava}')
                    new_cur_ava = np.zeros_like(cur_ava)
                    new_cur_ava[(cur_ava == 1) & (agent_shoot_legal == 1)] = 1
                    # # print(f'cur_ava', new_cur_ava)
                    # # cur_ava.append(1)  # 最后一维是不执行任何攻击工作
                    # cur_ava = np.append(new_cur_ava, 1)
                    # # cur_ava = np.append(cur_ava, 1)
                    # avai_mask[i][1] = cur_ava
                    avai_mask[i][9:14] = new_cur_ava        
            
            avai_dict.append(avai_mask[i])
            # avai_mask[i][2] = np.array([1, 1, 1], dtype=object) # 加速，减速，速度不变永远都可以执行

        # for i in range(5):
        #     avai_dict['0'].append(list(avai_mask[i][0]))
        #     avai_dict['1'].append(list(avai_mask[i][1]))
        #     avai_dict['2'].append(list(avai_mask[i][2]))

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
