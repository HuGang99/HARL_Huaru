# echo "停止所有正在运行的容器"
# docker stop $(docker ps -a -q)
# echo "删除所有容器"
# docker rm $(docker ps -a -q)

# while read container_id
# do

#     # 删除容器
#     echo "del container $container_id..."
#     docker stop $container_id
#     docker rm $container_id
#     sleep 1

# done < container_ids.txt

# 创建或清空容器ID文件
# > container_ids.txt

python  harl/envs/battle5v5/train.py --algo mappo --env huarubattle --exp_name get_reward_new


# # python  train.py --algo mappo --env huarubattle --exp_name get_simple_reward

# # python  train.py --algo happo --env huarubattle --exp_name get_simple_reward
