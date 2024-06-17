"""
@FileName：config.py
@Description：
@Time：2021/5/9 下午8:08
@Copyright：©2011-2021 北京华如科技股份有限公司
"""
from harl.envs.battle5v5.agent.agent import Agent
import sys
# sys.path.append("/home/gang/Hok_Marl_ppo(4)/src/envs/battle5v5")

from harl.envs.battle5v5.agent.alo_agent import AloAgent
from harl.envs.battle5v5.agent.blue_alo_agent import BlueAloAgent
from harl.envs.battle5v5.agent.demo_agent import DemoAgent
from harl.envs.battle5v5.agent.MyAgent.test_agent import TestAgent
# 蓝方超强规则Agents

# from harl.envs.battle5v5.agent.HR1 import HR1
# 是否启用host模式,host仅支持单个xsim
ISHOST = False

# 为态势显示工具域分组ID  1-1000
HostID = 1

IMAGE = "xsim:v7.0"

# 加速比 1-100
TimeRatio = 100

# 范围:0-100 生成的回放个数 (RTMNum + 2),后续的回放会把之前的覆盖掉.
RTMNum = 0

config = {
    "episode_time": 100,   # 训练次数
    "step_time": 1, # 想定步长
    'agents': {
            'red': AloAgent,
            'blue': TestAgent,
            # 'blue': HR1.HR1,
            'self_play_blue': BlueAloAgent
            }
}

# 进程数量
POOL_NUM = 10

# 启动XSIM的数量
XSIM_NUM = 6


ADDRESS = {
    "ip": "127.0.0.1",
    "port": 11112
}
