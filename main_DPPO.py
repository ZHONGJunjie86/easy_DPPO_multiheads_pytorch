from cProfile import run
import random
from re import M
import sys
from pathlib import Path
#from torch import manager_path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

import torch.multiprocessing as mp
import torch
import numpy as np
from log_path import *
from common import *
from algo.DPPO import PPO, Shared_grad_buffers
import argparse
import datetime
from matplotlib.pyplot import get
import os
from env.chooseenv import make
import sys
import time
import copy

from collections import deque, namedtuple

import wandb
from multiprocessing.managers import BaseManager
class MyManager(BaseManager):
    pass

MyManager.register("PPO_Copy",PPO)

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
Memory_size = 4


def train(rank, args, device, main_device, log_dir, run_dir, shared_lock, shared_count, event, model_enemy_path,
          shared_model=None, experiment_share_1=None, K_epochs=3, shared_grad_buffer = None):

    
    print("rank ", rank)
    print(f'device: {device}')

    env = make(args.game_name, conf=None)

    ctrl_agent_index = int(args.controlled_player)
    
    model = PPO(args, device)
    #model_enemy = PPO(args, device)

    model.actor.load_state_dict(
        shared_model.get_actor().state_dict())  # sync with shared model
    #model_enemy.actor.load_state_dict(shared_model.get_actor().state_dict())

    history_success = []
    history_enemy = {}
    RENDER = True#args.render #

    total_step_reward = [0,0]
    c_loss, a_loss = 0, 0
    step_reward = 0

    memory = Memory()
    memory_enemy = Memory()


    count_down = 100
    episode = 0
    episode_enemy_update = 0
    success = 0
    select_pre = False
    distance_dict = {"our_turn":dict(),"enemy":dict()}
    pre_ball_nums = 0

    
    record_win = deque(maxlen=100)
    record_win_op = deque(maxlen=100)

    our_turn = False
    if rank == 0:
        wandb.config = {
            "learning_rate": 0.0004,
        }
        
        wandb.init(project="Curling", entity="zhongjunjie")

    while episode < args.max_episodes:

        step = 0
        Gt = 0
        state = env.reset()
        if RENDER and rank == 0 and episode % 10 == 0:
            env.env_core.render()

        obs = np.array(state[ctrl_agent_index]['obs'])/10
        obs_enemy = np.array(state[1-ctrl_agent_index]['obs'])/10

        for _ in range(Memory_size):
            if np.sum(obs) != -90:
                memory.m_obs.append(obs)
                our_turn = True

            if np.sum(obs_enemy) != -90:
                memory_enemy.m_obs.append(obs_enemy)
                our_turn = False

        if our_turn:
            obs = np.stack(memory.m_obs)
        else:obs = np.stack(memory_enemy.m_obs)

        while True:
            pre_ball_nums = sum(state[ctrl_agent_index]["throws left"])
            if np.sum(np.array(state[ctrl_agent_index]['obs'])/10) != -90:
                our_turn = True
            else: our_turn = False

            #TODO another features maybe 
            num_vector = np.array([
                                int(state[ctrl_agent_index]['release']), 
                                state[ctrl_agent_index]["throws left"][0]/10,
                                state[ctrl_agent_index]["throws left"][1]/10,
                                int(our_turn)
                                ])
            ################################# collect  action #############################
            if our_turn:
                action_opponent = [[1], [1]]
            else:
                action_ctrl = [[1], [1]]
            action_raw = model.choose_action(obs, num_vector, our_turn)
            if our_turn:
                action_ctrl = linear_transformer(action_raw)
            else:
                action_opponent = linear_transformer(action_raw)

            action = [action_opponent, action_ctrl] if ctrl_agent_index == 1 else [action_ctrl, action_opponent]
            
            # if episode>1:
            #     print("our_turn",our_turn,"action--------",action)
            ################################# env rollout ##########################################
            #self.all_observes, reward, self.done, info_before, info_after
            positions = env.env_core.agent_pos
            next_state, reward, done, _, info = env.step(action)

            next_obs = np.array(next_state[ctrl_agent_index]['obs'])/10
            next_obs_enemy = np.array(next_state[1-ctrl_agent_index]['obs'])/10

            # stack Memory
            if np.sum(next_obs) != -90:
                if len(memory.m_obs_next) == 0:
                    if len(memory.m_obs) == 0:
                        for _ in range(Memory_size):
                            memory.m_obs.append(next_obs)
                        memory.m_obs_next = copy.deepcopy(memory.m_obs)
                    else:
                        memory.m_obs_next = copy.deepcopy(memory.m_obs)
                        memory.m_obs_next[-1] = next_obs
                else:
                    del memory.m_obs_next[:1]
                    memory.m_obs_next.append(next_obs)
                    
                next_obs = np.stack(memory.m_obs_next)

            if np.sum(next_obs_enemy) != -90:
                if len(memory_enemy.m_obs_next) == 0:
                    if len(memory_enemy.m_obs) == 0:
                        for _ in range(Memory_size):
                            memory_enemy.m_obs.append(next_obs_enemy)
                        memory_enemy.m_obs_next = copy.deepcopy(memory_enemy.m_obs)
                    else:
                        memory_enemy.m_obs_next = copy.deepcopy(memory_enemy.m_obs)
                        memory_enemy.m_obs_next[-1] = next_obs_enemy
                else:
                    del memory_enemy.m_obs_next[:1]
                    memory_enemy.m_obs_next.append(next_obs_enemy)

                next_obs = np.stack(memory_enemy.m_obs_next)
            
            
            step += 1

            # ================================== reward shaping ========================================
            #done reward [0.0, 100.0]   post_reward[-100.0, 100.0]           
            #reward投完暂胜1 一局完10.0 结束100
            #350.0 距离开始，靠近环90左右,left and right nodes of red line is 200

            step_reward = compute_reward(state, ctrl_agent_index, positions,  distance_dict, our_turn, count_down, step_reward)

            count_down -= 1

            if our_turn:
                reward_index = 0
            else:
                reward_index = 1
            
            model.memory_our_enemy[reward_index].rewards.append(step_reward)
            #只看距离感觉奖励
            total_step_reward[reward_index] += step_reward
            
            #投完奖励
            if sum(reward) == 1 or sum(reward) == 10 or sum(reward) == 100:
                if sum(reward) == 100:
                    reward[reward.index(100)] = 10
                winner_index = reward.index(1) if sum(reward) == 1 else reward.index(10) 
                # our index 1
                if len(model.memory_our_enemy[winner_index].rewards) != 0:
                    model.memory_our_enemy[1-winner_index].rewards[-1] +=  0#max (0.5 * sum(reward) ,2)
                    total_step_reward[1-winner_index] +=  0#max (0.5 * sum(reward) ,2)
                    model.memory_our_enemy[winner_index].rewards[-1] -=  0#max (0.5 * sum(reward) ,2)
                    total_step_reward[winner_index] -=  0#max (0.5 * sum(reward) ,2)

                model.memory_our_enemy[reward_index].is_terminals.append(0)
                count_down = 100
            else: model.memory_our_enemy[reward_index].is_terminals.append(0)

            #球投完
            # if sum(reward) == 10:
            #     if reward[0] != reward[1]:
            #         post_reward = [reward[0]-100, reward[1]] if reward[0]<reward[1] else [reward[0], reward[1]-100]
            #     else:
            #         post_reward=[-1., -1.]
            

            if RENDER and rank == 0 and episode % 10 == 0:
                env.env_core.render()
            #?
            Gt += reward[ctrl_agent_index] if done else 0

            # ================================== collect data ========================================
            # Store transition in R
            state = next_state
            obs = next_obs
            #obs_enemy = next_obs_enemy
            
            if sum(reward) == 10 or done or sum(reward) == 100 or pre_ball_nums<sum(next_state[ctrl_agent_index]["throws left"]):
                distance_dict = {"our_turn":dict(),"enemy":dict()}
                model.memory_our_enemy[0].is_terminals.append(1)
                model.memory_our_enemy[1].is_terminals.append(1)
                
                win_is = 1 if reward[ctrl_agent_index]>reward[1-ctrl_agent_index] else 0
                win_is_op = 1 if reward[ctrl_agent_index]<reward[1-ctrl_agent_index] else 0
                record_win.append(win_is)
                record_win_op.append(win_is_op)
                if rank == 0:
                    print("Episode: ", episode, "controlled agent: ", ctrl_agent_index, "; Episode Return: ", Gt,
                        "; win rate(controlled & opponent): ", '%.2f' % (sum(record_win)/len(record_win)),
                        '%.2f' % (sum(record_win_op)/len(record_win_op)))


                #writer.add_scalar('training Gt', Gt, episode)
                # Training
                a_loss, c_loss = K_epochs_PPO_training(rank, event, 
                          None, model_enemy_path, model, shared_model,   #model_enemy 暂不用自博弈
                          shared_count, shared_grad_buffer, shared_lock, 
                          K_epochs, args, episode, run_dir, device )

                model.memory_our_enemy[0].clear_memory()
                model.memory_our_enemy[1].clear_memory()

                if win_is:
                    success = 1
                else:
                    success = 0
                history_success.append(success)

                if rank == 0:
                    wandb.log({"a_loss": a_loss, "c_loss": c_loss, 
                    "total_step_reward_0": total_step_reward[0],
                    "total_step_reward_1": total_step_reward[1],
                               })

                    #暂不用自博弈
                    # if len(history_success) >= 200 and sum(history_success[-200:]) > 110:
                    #     episode_enemy_update, select_pre = self_playing_update(model_enemy, model_enemy_path, model, shared_lock, 
                    #                                                             args, episode, episode_enemy_update,
                    #                                                             history_enemy, history_success, select_pre)
                    #     history_success = []

                total_step_reward = [0, 0]
                memory.clear_memory()
                memory_enemy.clear_memory()
                env.reset()
                step_reward = 0
                count_down = 100
                episode += 1
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default="olympics-curling", type=str)
    parser.add_argument('--algo', default="ddpg", type=str, help="bicnet/ddpg")
    parser.add_argument('--max_episodes', default=500000, type=int)  # 50000
    parser.add_argument('--episode_length', default=200, type=int)

    parser.add_argument('--gamma', default=0.99, type=float)  # 0.95
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--a_lr', default=0.0003, type=float)  # 0.0001
    
    parser.add_argument('--controlled_player', default=1, help="0(agent purple) or 1(agent green)")
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--save_interval", default=20, type=int)  # 1000
    parser.add_argument("--model_episode", default=0, type=int)
    parser.add_argument(
        '--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    # PPO
    parser.add_argument("--load_model", action='store_true')  # 加是true；不加为false
    parser.add_argument("--load_model_run", default=4, type=int)
    parser.add_argument("--load_model_run_episode", default=4000, type=int)
    parser.add_argument("--K_epochs", default=4, type=int)

    # Multiprocessing
    parser.add_argument('--processes', default=10, type=int,
                        help='number of processes to train with')

    args = parser.parse_args()

    if sys.version_info[0] > 2:
        mp.set_start_method('spawn')  # this must not be in global scope
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        # or else you get a deadlock in conv2d
        raise "Must be using Python 3 with linux!"

    manager = MyManager()
    manager.start()


    main_device = torch.device(
        "cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    model_share = manager.PPO_Copy(args, main_device)

    # 定义保存路径
    #/home/j-zhong/work_place/Competition_Olympics-Curling-main/rl_trainer/models/olympics-curling/run1/trained_model/actor_.pth
    model_enemy_path = "/home/j-zhong/work_place/Competition_Olympics-Curling-main/rl_trainer/models/snakes_3v3/run6/trained_model/enemy.pth"
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    print("run_dir-----------------------------",run_dir)
    if False:#True:#args.load_model:  # 
        load_dir = os.path.join(os.path.dirname(
            run_dir), "run" + str(args.load_model_run))
        model_share.load_model(load_dir, episode=args.load_model_run_episode)
    model_share.save_model(run_dir, 0)

    shared_lock = mp.Manager().Lock()
    event = mp.Event()
    shared_count = mp.Value("d", 0)
    list_1 = mp.Manager().list()
    list_2 = mp.Manager().list()
    experiment_share_1 = mp.Manager().dict(
        {"a_loss": list_1, "c_loss": list_2})
    shared_grad_buffer = Shared_grad_buffers(model_share.get_actor(), main_device)

    processes = []
    K_epochs = args.K_epochs

    for rank in range(args.processes):  # rank 编号
        if rank < args.processes*0.4 or rank==0:
            print("start", rank)
            p = mp.Process(target=train, args=(rank, args, main_device, main_device, log_dir, run_dir, shared_lock, shared_count,
                                               event, model_enemy_path, model_share, experiment_share_1, K_epochs, shared_grad_buffer))
        else:
            device = torch.device(
                "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            p = mp.Process(target=train, args=(rank, args, device, main_device, log_dir, run_dir, shared_lock, shared_count,
                                               event, model_enemy_path, model_share, experiment_share_1, K_epochs, shared_grad_buffer))

        p.start()
        processes.append(p)
    for p in processes: 
        p.join()