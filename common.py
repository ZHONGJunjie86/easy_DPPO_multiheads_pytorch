import math
import time
import copy
import torch
from algo.DPPO import PPO
import random
import numpy as np


cpu_device = torch.device("cpu")

class Memory:
    def __init__(self):
        self.m_obs = []
        self.m_obs_next = []
    
    def clear_memory(self):
        del self.m_obs[:]
        del self.m_obs_next[:]

        
def compute_distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx**2 + dy**2)

def compute_group_distance(origin, group_dict, positions):
    distance = 0
    for ball_dis_index in group_dict.keys():
        if group_dict[ball_dis_index] != None:
            dx = origin[0] - positions[group_dict[ball_dis_index]][0]
            dy = origin[1] - positions[group_dict[ball_dis_index]][1]
            distance += math.sqrt(dx**2 + dy**2)
        else:
            distance += 200
    return distance

def compute_reward(state, ctrl_agent_index, positions,  distance_dict, our_turn, count_down, step_reward):
    
    #距离奖励
    if state[ctrl_agent_index]['release']:
        if len(positions) == 1 and sum(positions[0])==450:
            return step_reward
        if our_turn:
            distance_dict["our_turn"][sum(state[ctrl_agent_index]["throws left"])] \
                                = len(positions)-1
            
            #对方有球在场上
            if len(distance_dict["enemy"]) != 0:
                step_reward = (compute_group_distance([300, 500], distance_dict["enemy"], positions)/len(distance_dict["enemy"])
                                - compute_group_distance([300, 500], distance_dict["our_turn"], positions)/len(distance_dict["our_turn"]))/1000
            else: step_reward = (125 - compute_distance([300, 500], positions[-1]))/1000
        else:
            distance_dict["enemy"][sum(state[ctrl_agent_index]["throws left"])] \
                                = len(positions)-1
            #对方有球在场上
            if len(distance_dict["our_turn"]) != 0:
                step_reward = (compute_group_distance([300, 500], distance_dict["our_turn"], positions) /len(distance_dict["our_turn"])
                                - compute_group_distance([300, 500], distance_dict["enemy"], positions)/len(distance_dict["enemy"]))/1000
            else: step_reward = (125 - compute_distance([300, 500], positions[-1]))/1000
        
    else:
        step_reward = 0
        if count_down < 50:
            step_reward = -0.02
        if our_turn:
            distance_dict["our_turn"][sum(state[ctrl_agent_index]["throws left"])] = None
        else:
            distance_dict["enemy"][sum(state[ctrl_agent_index]["throws left"])] = None

    return step_reward





#[Box(-100.0, 200.0, (1,), float32), Box(-30.0, 30.0, (1,), float32)]
actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
               7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
               14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
               21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
               28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
               35: [200, 30]}  


def self_playing_update(model_enemy, model_enemy_path, model, shared_lock, 
                        args, episode, episode_enemy_update,
                        history_enemy, history_success, select_pre):
    print(
        "-------------------------Update Enemy!!!----------------------------")
    print("Success Rate",
            (sum(history_success[-200:])/200.0)*100, "%")

    gap = episode - episode_enemy_update
    if len(history_enemy) < 5 and (not select_pre):
        history_model = PPO(args, cpu_device)
        history_model.actor.load_state_dict(
            model_enemy.actor.state_dict())
        history_enemy[history_model] = gap
    else:
        if gap > min(history_enemy.values()) and (not select_pre):  # 只会有第一个
            del history_enemy[min(
                history_enemy, key=history_enemy.get)]
            history_model = PPO(args, cpu_device)
            history_model.actor.load_state_dict(
                model_enemy.actor.state_dict())
            history_enemy[history_model] = gap

    if np.random.uniform() >= 0.2:
        model_enemy.actor.load_state_dict(
            model.actor.state_dict())
        select_pre = False
    else:
        print("选个以前牛逼的!!!!!")
        niubi_enemy = random.sample(history_enemy.keys(), 1)[0]
        model_enemy.actor.load_state_dict(
            niubi_enemy.actor.state_dict())
        select_pre = True

    shared_lock.acquire()
    torch.save(model_enemy.actor.state_dict(), model_enemy_path)
    shared_lock.release()

    episode_enemy_update = episode

    return episode_enemy_update, select_pre


def K_epochs_PPO_training(rank, event, 
                          model_enemy, model_enemy_path, model, shared_model, 
                          shared_count, shared_grad_buffer, shared_lock, 
                          K_epochs, args, episode, run_dir, device ):
    if rank == 0:
        print("---------------------------training!")
        training_time = 0
        shared_model.copy_memory(model.memory_our_enemy)
        while training_time < K_epochs:
            #
            a_loss, c_loss = model.compute_GAE(training_time)

            while shared_count.value < args.processes-1:
                time.sleep(0.01)
            time.sleep(0.01)
            #
            shared_lock.acquire()
            
            model.add_gradient(shared_grad_buffer)

            shared_count.value = 0
            shared_lock.release()
            #
            shared_model.update(copy.deepcopy(shared_grad_buffer.grads), args.processes)
            shared_grad_buffer.reset()

            c_loss, a_loss = model.get_loss()
            model.actor.load_state_dict(
                shared_model.get_actor().state_dict())

            event.set()
            event.clear()
            training_time += 1

        # torch.save(model_enemy.actor.state_dict(), model_enemy_path)
        model.reset_loss()
        shared_model.clear_memory()
        if episode % 20 == 0:
            shared_model.save_model(run_dir, episode)
        return a_loss, c_loss

    else:
        training_time = 0
        while training_time < K_epochs:
            a_loss, c_loss = model.compute_GAE(training_time)

            shared_lock.acquire()

            model.add_gradient(shared_grad_buffer)

            shared_count.value += 1
            shared_lock.release()

            event.wait()

            model.actor.load_state_dict(
                shared_model.get_actor().state_dict())

            training_time += 1

        # enemy_temp = torch.load(model_enemy_path , map_location = device)
        # model_enemy.load_state_dict(enemy_temp)
        return 0, 0


def linear_transformer(action):
    #[Box(-100.0, 200.0, (1,), float32), Box(-30.0, 30.0, (1,), float32)]
    return [[np.clip(action[0]*150 + 50, -100, 200)], [np.clip(action[1]*30, -30, 30)]]



