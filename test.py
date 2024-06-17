import numpy
import torch
# share_obs  = numpy.ones((5, 2,620))
# share_obs[0][0] = 0
# share_obs[1][0] = 0
# print(share_obs[:, 0])
# print(share_obs[:, 0].shape)

# dones  = numpy.ones((2))
# print(dones)
# done_env = numpy.all(dones, axis=0)
# print(done_env)

# a = torch.Tensor([0.1, 0.2, 0.7])
# a_mask = torch.Tensor([[0, 0, 1], [0, 1, 1]])
# print(a)
# print(a_mask)
# a[a_mask[0] == 0] = -1e10
# print(a)

# avai_mask = torch.Tensor([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1]], [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1]], [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1]], [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1]], [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1]]])
# print(avai_mask)
# print(avai_mask[0])

# avai_mask1 = torch.Tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
# print(avai_mask1)
# print(avai_mask1[0])

# from gymnasium import spaces
# act_space = spaces.MultiDiscrete([10,6])
# print(act_space.nvec)
# print(sum(act_space.nvec))

# available_actions = numpy.ones((2,16))
# available_actions[0][4] = 0
# available_actions[1][14] = 0
# print(available_actions[:, 0:act_space.nvec[0]])
# print(available_actions[:, act_space.nvec[0]:act_space.nvec[0]+act_space.nvec[1]])
# c = []
# for i in range(3):
#     a = 8 + i
#     b = 9 + i
#     d = [a,b]
#     c.extend(d)
# print(c)

# q = [55]
# c = []
# m = [1, 10, 20]
# n = [199, 100, 30]
# print(n[1:])
# # for i in range(3):
#     if i ==1:
#         a = m[i]
#         red1 = [a]
#     if i == 2:
#         a = n[i]
#         red2 = [a]
# print(red1,red2)
#     a = m[i]
#     b = n[i]
#     d = [a,b]
#     c.extend(d)
# q.extend(c)
# print(c)
# print(q)
