import random
import torch.multiprocessing as mp
import numpy as np
from Curve_ import cross_loss_curve
from log_path import *
from common import *
from algo.DPPO import PPO, Shared_grad_buffers
import argparse
import datetime
from matplotlib.pyplot import get
import os
import wandb
from tensorboardX import SummaryWriter
import sys
sys.path.append("/home/j-zhong/work_place/TD3_SAC_PPO_multi_Python/")
from env.chooseenv import make
from pathlib import Path
import sys
import time
import copy

from multiprocessing.managers import BaseManager
class MyManager(BaseManager):
    pass

MyManager.register("PPO_Copy",PPO)

#from torch import manager_path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
cpu_device = torch.device("cpu")

Memory_size = 4


def train(rank, args, device, main_device, log_dir, run_dir, shared_lock, shared_count, event, model_enemy_path,
          shared_model=None, experiment_share_1=None, K_epochs=3, shared_grad_buffer = None):

    if rank == 0:
        writer = SummaryWriter(str(log_dir))
        save_config(args, log_dir)
    print(f'device: {device}')

    env = make(args.game_name, conf=None)

    ctrl_agent_index = [0, 1, 2]
    ctrl_agent_num = len(ctrl_agent_index)

    obs_dim = 26
    width = env.board_width
    #print(f'Game board width: {width}')
    height = env.board_height

    if rank == 0:
        wandb.init(project="my-test-project", entity="zhongjunjie")
        wandb.config = {
            "learning_rate": 0.0001,
            "batch_size": args.batch_size
        }

    # PPO(obs_dim*Memory_size, act_dim, ctrl_agent_num, args,device)
    model = PPO(args, device)
    model_enemy = PPO(args, device)

    model.actor.load_state_dict(
        shared_model.get_actor().state_dict())  # sync with shared model
    model.critic.load_state_dict(
        shared_model.get_critic().state_dict())  # sync with shared model
    model_enemy.actor.load_state_dict(shared_model.get_actor().state_dict())

    history_reward = []
    history_step_reward = []
    history_a_loss = []
    history_c_loss = []
    history_success = []
    history_enemy = {}

    total_step_reward = 0
    c_loss, a_loss = 0, 0
    step_reward = [0, 0, 0]

    memory = Memory()
    memory_enemy = Memory()

    # torch.manual_seed(args.seed)

    episode = 0
    episode_enemy_update = 0
    success = 0
    select_pre = False

    while episode < args.max_episodes:

        punishiment_lock = [6, 6, 6]

        state = env.reset()

        state_to_training = state[0]
        obs = visual_ob(state[0])
        obs_enemy = copy.deepcopy(obs)
        obs = obs/10
        obs_enemy = get_enemy_obs(obs_enemy)

        for _ in range(Memory_size):
            memory.m_obs.append(obs)
            memory_enemy.m_obs.append(obs_enemy)
        obs = np.stack(memory.m_obs)
        obs_enemy = np.stack(memory_enemy.m_obs)

        episode += 1
        step = 0
        episode_reward = np.zeros(6)

        while True:
            num_vector = np.array([np.sum((episode_reward[:3]) - np.sum(episode_reward[3:]))/10,
                                  step/100])
            logits = model.choose_action(obs, num_vector )

            #logits_enemy = model_enemy.choose_action(obs_enemy)
            #actions = np.array([logits , logits_enemy]).reshape(6)
            actions = logits_AC(state_to_training, logits, height, width)

            next_state, reward, done, _, info = env.step(env.encode(actions))
            next_state_to_training = next_state[0]

            next_obs = visual_ob(next_state_to_training)
            next_obs_enemy = copy.deepcopy(next_obs)
            next_obs_enemy = get_enemy_obs(next_obs_enemy)
            next_obs = next_obs/10

            # Memory
            if len(memory.m_obs_next) != 0:
                del memory.m_obs_next[:1]
                del memory_enemy.m_obs_next[:1]
                memory.m_obs_next.append(next_obs)
                memory_enemy.m_obs_next.append(next_obs_enemy)
            else:
                memory.m_obs_next = memory.m_obs
                memory_enemy.m_obs_next = memory_enemy.m_obs
                memory.m_obs_next[Memory_size-1] = next_obs
                memory_enemy.m_obs_next[Memory_size-1] = next_obs_enemy

            next_obs = np.stack(memory.m_obs_next)
            next_obs_enemy = np.stack(memory_enemy.m_obs_next)

            # ================================== reward shaping ========================================
            reward = np.array(reward)
            episode_reward += reward
            """se:"""

            if done:  # 结束
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):  # AI赢
                    step_reward, enemy_reward = get_reward(
                        info, ctrl_agent_index, reward, punishiment_lock, score=1)
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):  # random赢
                    step_reward, enemy_reward = get_reward(
                        info, ctrl_agent_index, reward, punishiment_lock, score=2)
                else:  # 一样长
                    step_reward, enemy_reward = get_reward(
                        info, ctrl_agent_index, reward, punishiment_lock, score=0)
            elif np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):  # AI长
                step_reward, enemy_reward = get_reward(
                    info, ctrl_agent_index, reward, punishiment_lock, score=3)
            elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):  # random长
                step_reward, enemy_reward = get_reward(
                    info, ctrl_agent_index, reward, punishiment_lock, score=4)
            else:  # 一样长
                step_reward, enemy_reward = get_reward(
                    info, ctrl_agent_index, reward, punishiment_lock, score=0)

            total_step_reward += sum(step_reward)

            done = np.array([done] * ctrl_agent_num)

            model.memory.rewards.append(sum(step_reward))
            model.memory.is_terminals.append(done)

            # ================================== collect data ========================================
            # Store transition in R

            obs = next_obs
            obs_enemy = next_obs_enemy
            step += 1

            # Training
            if args.episode_length <= step:
                if rank == 0:
                    training_time = 0
                    shared_model.copy_memory(model.memory)
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

                    #torch.save(model_enemy.actor.state_dict(), model_enemy_path)
                    model.reset_loss()
                    shared_model.clear_memory()
                    if episode % 20 == 0:
                        shared_model.save_model(run_dir, episode)
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

                    #enemy_temp = torch.load(model_enemy_path , map_location=device)
                    # model_enemy.load_state_dict(enemy_temp)

                model.memory.clear_memory()

                if np.sum(episode_reward[0:3]) > np.sum(episode_reward[3:]):
                    success = 1
                else:
                    success = 0
                history_success.append(success)

                if rank == 0:
                    print(
                        f'[Episode {episode:05d}] total_reward: {np.sum(episode_reward[0:3]):} rank: {rank:.2f}')
                    print(
                        f'[Episode {episode:05d}] Enemy_reward: {np.sum(episode_reward[3:]):}')
                    print(f'\t\t\t\tsnake_1: {episode_reward[0]} '
                          f'snake_2: {episode_reward[1]} snake_3: {episode_reward[2]}')

                    reward_tag = 'reward'
                    loss_tag = 'loss'

                    writer.add_scalars(reward_tag, global_step=episode,
                                       tag_scalar_dict={'snake_1': episode_reward[0], 'snake_2': episode_reward[1],
                                                        'snake_3': episode_reward[2], 'total': np.sum(episode_reward[0:3])})

                    if c_loss and a_loss:
                        writer.add_scalars(loss_tag, global_step=episode,
                                           tag_scalar_dict={'actor': a_loss, 'critic': c_loss})

                    if c_loss and a_loss:
                        print(
                            f'\t\t\t\ta_loss {a_loss:.3f} c_loss {c_loss:.3f}')

                    history_reward.append(np.sum(episode_reward[0:3]))
                    history_a_loss.append(a_loss/100)
                    history_c_loss.append(c_loss/10)

                    history_step_reward.append(total_step_reward/100)
                    cross_loss_curve(history_reward, history_a_loss,
                                     history_c_loss, history_step_reward)
                    wandb.log({"a_loss": a_loss, "c_loss": c_loss,
                               "reward": np.sum(episode_reward[0:3]),
                               "relative_reward": (np.sum(episode_reward[0:3])-np.sum(episode_reward[3:])), "total_step_reward": total_step_reward,
                               })

                if rank == 0 and len(history_success) >= 200 and sum(history_success[-200:]) > 110:
                    print("-------------------------rank",
                          rank, "----------------------------")
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
                    #torch.save(model_enemy.actor.state_dict(), model_enemy_path)
                    shared_lock.release()

                    episode_enemy_update = episode
                    history_success = []

                total_step_reward = 0
                memory.clear_memory()
                memory_enemy.clear_memory()
                env.reset()
                step_reward = [0, 0, 0]
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default="snakes_3v3", type=str)
    parser.add_argument('--algo', default="ddpg", type=str, help="bicnet/ddpg")
    parser.add_argument('--max_episodes', default=500000, type=int)  # 50000
    parser.add_argument('--episode_length', default=200, type=int)
    parser.add_argument('--output_activation',
                        default="softmax", type=str, help="tanh/softmax")

    parser.add_argument('--tau', default=0.01, type=float)  # 0.005
    parser.add_argument('--gamma', default=0.99, type=float)  # 0.95
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--a_lr', default=0.0003, type=float)  # 0.0001
    parser.add_argument('--c_lr', default=0.0001, type=float)  # 0.0003
    parser.add_argument('--batch_size', default=512,
                        type=int)  # 32768  16384 8192 4096
    parser.add_argument('--epsilon', default=0.5, type=float)
    parser.add_argument('--epsilon_speed', default=0.993,
                        type=float)  # 0.99998

    parser.add_argument("--save_interval", default=20, type=int)  # 1000
    parser.add_argument("--model_episode", default=0, type=int)
    parser.add_argument(
        '--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    # PPO
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')

    parser.add_argument("--load_model", action='store_true')  # 加是true；不加为false
    parser.add_argument("--load_model_run", default=1, type=int)
    parser.add_argument("--load_model_run_episode", default=4000, type=int)

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
        "cuda:2") if torch.cuda.is_available() else torch.device("cpu")
    model_share = manager.PPO_Copy(args, main_device)

    # 定义保存路径
    model_enemy_path = "/home/j-zhong/work_place/TD3_SAC_PPO_multi_Python/rl_trainer/models/snakes_3v3/run5/trained_model/enemy.pth"
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    if args.load_model:  # False:#True:#
        load_dir = os.path.join(os.path.dirname(
            run_dir), "run" + str(args.load_model_run))
        model_share.load_model(load_dir, episode=args.load_model_run_episode)

    shared_lock = mp.Manager().Lock()
    event = mp.Event()
    shared_count = mp.Value("d", 0)
    list_1 = mp.Manager().list()
    list_2 = mp.Manager().list()
    experiment_share_1 = mp.Manager().dict(
        {"a_loss": list_1, "c_loss": list_2})
    shared_grad_buffer = Shared_grad_buffers(model_share.get_actor(), main_device)

    processes = []
    K_epochs = 4

    for rank in range(args.processes):  # rank 编号
        if rank < args.processes*0.4 or rank==0:
            p = mp.Process(target=train, args=(rank, args, main_device, main_device, log_dir, run_dir, shared_lock, shared_count,
                                               event, model_enemy_path, model_share, experiment_share_1, K_epochs, shared_grad_buffer))
        else:
            device = torch.device(
                "cuda:3") if torch.cuda.is_available() else torch.device("cpu")
            p = mp.Process(target=train, args=(rank, args, device, main_device, log_dir, run_dir, shared_lock, shared_count,
                                               event, model_enemy_path, model_share, experiment_share_1, K_epochs, shared_grad_buffer))

        p.start()
        processes.append(p)
    for p in processes:
        p.join()


# print("---run_dir",run_dir)
# model_share.save_model(run_dir, 0)
