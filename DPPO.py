from distutils import log
import os
from socketserver import ThreadingUnixDatagramServer
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
import torch.nn.functional as F

from torch.distributions import Normal #Multivariate
import torch.nn as nn
import torch.multiprocessing as mp
from torch.autograd import Variable
import copy

torch.set_default_tensor_type(torch.DoubleTensor)
hidden_size = 64


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.hidden_state = []
        self.num_vectors = []
        self.value = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.hidden_state[:]
        del self.num_vectors[:]
        del self.value[:]

class Actor(nn.Module):
    def __init__(self):  #(n+2p-f)/s + 1 
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(4,16, kernel_size=3, stride=1, padding=1) # 30 -> 15
        self.maxp1 = nn.MaxPool2d(4, stride=2 , padding= 1)
        self.conv2 = nn.Conv2d(16,16, kernel_size=4, stride=1, padding=0) # 15 -> 12
        self.conv3 = nn.Conv2d(16,8, kernel_size=4, stride=1, padding=0) # 12 -> 9
        self.self_attention = nn.MultiheadAttention(652, 4)

        self.gru = nn.GRU(652, 64, 1) 
        self.critic_linear = nn.Linear(64, 1)
        self.linear_mu = nn.Linear(64, 2)
        self.linear_sigma = nn.Linear(64, 2)

        self.normal =  torch.distributions.Normal
        self.num_vector_length = 4

    def forward(self, tensor_cv, num_vector, h_old,  old_action = None): #,batch_size
        # CV
        self.batch_size = tensor_cv.size()[0]
        i_1 = tensor_cv
        # CV
        x = F.relu(self.maxp1(self.conv1(i_1)))
        #i_2 = i_1 + x
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)).reshape(1,self.batch_size,648)#(1,self.batch_size,640)
        
        step = num_vector.reshape(1,self.batch_size,self.num_vector_length)

        x = torch.cat([x, step], -1)
        x = self.self_attention(x,x,x)[0] + x
        x,h_state = self.gru(x, h_old)
        
        value = self.critic_linear(x)

        #[Box(-100.0, 200.0, (1,), float32), Box(-30.0, 30.0, (1,), float32)]
        mu = torch.tanh(self.linear_mu(x)).reshape(self.batch_size, 2, 1)
        sigma = torch.relu(self.linear_sigma(x)).reshape(self.batch_size, 2, 1) + 1e-6

        dist = self.normal(mu,sigma)  
        entropy = dist.entropy().mean()

        if old_action == None:
            action = dist.sample().reshape(self.batch_size, 2,1)
        else: 
            action = old_action.reshape(self.batch_size, 2,1)

        selected_log_prob = dist.log_prob(action)   
        
        return action, selected_log_prob, entropy, h_state.data, value.reshape(self.batch_size,1,1)


class PPO:
    def __init__(self,  args, device):
        self.device = device
        self.a_lr = args.a_lr
        self.gamma = args.gamma

        # Initialise actor network 
        self.actor = Actor().to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.SmoothL1Loss = torch.nn.SmoothL1Loss()

        #
        self.memory_our_enemy = [Memory(), Memory()]
        self.hidden_state = torch.zeros(1,1,64).to(self.device)
        #
        self.c_loss = 0
        self.a_loss = 0
        
        self.eps_clip = 0.1
        self.vf_clip_param = 0.5
        self.lam = 0.95
        self.K_epochs = args.K_epochs
        self.old_value_1, self.old_value_2 = 0,0
        
        #
        self.shared_loss = 0
        self.loss_dic = [0,0]
        self.advantages = []
        self.target_value = []
        self.num_vector_length = 4

    # Random process N using epsilon greedy
    def choose_action(self, obs, num_vector, our_turn = False):
        memery_index = 0 if our_turn else 1

        self.memory_our_enemy[memery_index].states.append(obs)
        num_vector = torch.tensor(num_vector, device = self.device) #np.full(1, step/100)
        obs = torch.Tensor(obs).to(self.device).reshape(1,4,30,30)
        if len(self.memory_our_enemy[memery_index].hidden_state)==0:
            self.memory_our_enemy[memery_index].hidden_state.append(self.hidden_state.cpu().detach().numpy())

        action,action_logprob,_,self.hidden_state, value = self.actor(obs, num_vector, self.hidden_state)
        
        self.memory_our_enemy[memery_index].actions.append(action.cpu().detach().numpy())
        self.memory_our_enemy[memery_index].logprobs.append(action_logprob.cpu().detach().numpy()) #[0]
        self.memory_our_enemy[memery_index].hidden_state.append(self.hidden_state.cpu().detach().numpy())
        self.memory_our_enemy[memery_index].num_vectors.append(num_vector.cpu().detach().numpy())
        self.memory_our_enemy[memery_index].value.append(value.cpu().detach().numpy())
        return action.reshape(1,2).cpu().detach().numpy()[0]


    def compute_GAE(self, training_time, main_process = False):
        
        if training_time ==0:
            batch_size_1 = torch.tensor(self.memory_our_enemy[0].logprobs).view(-1, 2, 1).size()[0]
            batch_size_2 = torch.tensor(self.memory_our_enemy[1].logprobs).view(-1, 2, 1).size()[0]
            self.old_states = torch.cat( 
                            [torch.tensor(self.memory_our_enemy[0].states).view(-1,4,30,30),
                            torch.tensor(self.memory_our_enemy[1].states).view(-1,4,30,30)
                            ], 0).to(self.device).detach() 
            self.old_logprobs = torch.cat(
                            [torch.tensor(self.memory_our_enemy[0].logprobs).view(-1, 2, 1),
                            torch.tensor(self.memory_our_enemy[1].logprobs).view(-1, 2, 1)
                            ], 0).to(self.device).detach()
            self.old_actions = torch.cat( 
                            [torch.tensor(self.memory_our_enemy[0].actions).view(-1, 2, 1),
                            torch.tensor(self.memory_our_enemy[1].actions).view(-1, 2, 1)
                            ], 0).to(self.device).detach() 

            self.old_num_vector = torch.cat(
                            [torch.tensor(self.memory_our_enemy[0].num_vectors).view( -1,1, self.num_vector_length),
                            torch.tensor(self.memory_our_enemy[1].num_vectors).view(-1,1, self.num_vector_length)
                            ], 0).to(self.device).detach() 
            self.old_value = torch.cat(
                            [torch.tensor(self.memory_our_enemy[0].value).view(-1, 1, 1),
                            torch.tensor(self.memory_our_enemy[1].value).view(-1, 1, 1)
                            ], 0).to(self.device).detach()    
            self.old_hidden = torch.cat( 
                            [torch.tensor(self.memory_our_enemy[0].hidden_state[:-1]).view(-1, 1, 64),
                            torch.tensor(self.memory_our_enemy[1].hidden_state[:-1]).view(-1, 1, 64)
                            ], 0).to(self.device).detach() 
            compute_rewards = torch.cat( 
                            [torch.tensor(self.memory_our_enemy[0].rewards).view(-1, 1, 1),
                            torch.tensor(self.memory_our_enemy[1].rewards).view(-1, 1, 1)
                            ], 0).to(self.device).detach() 
            compute_termi = torch.cat( 
                            [torch.tensor(self.memory_our_enemy[0].is_terminals).view(-1, 1, 1)[-batch_size_1:],
                            torch.tensor(self.memory_our_enemy[1].is_terminals).view(-1, 1, 1)[-batch_size_2:]
                            ], 0).to(self.device).detach()
            # Monte Carlo estimate of rewards:
            rewards = []
            GAE_advantage = []
            target_value = []
            #
            discounted_reward = 0
            values_pre = 0
            advatage = 0

            for reward, is_terminal,values in zip(reversed(compute_rewards), reversed(compute_termi),
                                        reversed(self.old_value)): #反转迭代
                
                values = values.cpu().detach().numpy()
                reward = reward.cpu().detach().numpy()
                is_terminal = is_terminal.cpu().detach().numpy()

                discounted_reward = reward +  self.gamma *discounted_reward 
                rewards.insert(0, discounted_reward) #插入列表

                
                delta = reward + (1-is_terminal)*self.gamma*values_pre - values  
                advatage = delta + self.gamma*self.lam*advatage * (1-is_terminal)
                GAE_advantage.insert(0, advatage) #插入列表
                target_value.insert(0,float(values + advatage))
                values_pre = values
            
            # Normalizing the rewards:
            rewards = torch.tensor(rewards).to(self.device).view(-1,1,1)
            self.target_value = torch.tensor(target_value).to(self.device).view(-1,1,1)
            GAE_advantage = torch.tensor(GAE_advantage).to(self.device).view(-1,1,1)
            self.advantages = (GAE_advantage- GAE_advantage.mean()) / (GAE_advantage.std() + 1e-6) 
            
        #compute
        batch_size = self.target_value.size()[0]
        batch_sample = int(batch_size / self.K_epochs)
        indices = torch.randint(batch_size, size=(batch_sample,), requires_grad=False)#, device=self.device

        old_states = self.old_states[indices]
        old_num_vector = self.old_num_vector.reshape(batch_size,1,self.num_vector_length)[indices].view(1,-1, self.num_vector_length)
        old_hidden = self.old_hidden.reshape(batch_size,1,64)[indices].view(1,-1, 64)
        old_actions = self.old_actions[indices]
        old_logprobs = self.old_logprobs[indices]
        advantages = self.advantages[indices].detach()
        old_value = self.old_value[indices]
        target_value = self.target_value[indices]


        _, logprobs, dist_entropy, _, value = self.actor(old_states, old_num_vector, old_hidden, old_actions)
 
        ratios = torch.exp(logprobs.view(-1,2,1).sum(1,keepdim = True) - 
                           old_logprobs.view(-1,2,1).sum(1,keepdim = True).detach())

        surr1 = ratios*advantages
        surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip)*advantages 
        #Dual_Clip
        surr3 = torch.min(surr1, surr2)#torch.max(torch.min(surr1, surr2),3*advantages.detach())
        #
        
        value_pred_clip = old_value.detach() +\
            torch.clamp(value -old_value.detach(), -self.vf_clip_param, self.vf_clip_param)#self.vf_clip_param
        critic_loss1 = (value - target_value.detach()).pow(2)
        critic_loss2 = (value_pred_clip - target_value.detach()).pow(2)
        critic_loss = 0.5 * torch.max(critic_loss1 , critic_loss2).mean()
        #critic_loss = torch.nn.SmoothL1Loss()(state_values_1, target_value) + torch.nn.SmoothL1Loss()(state_values_2, target_value)

        actor_loss = -surr3.mean() - 0.02*dist_entropy + 0.5 * critic_loss
        
        # do the back-propagation...
        self.actor.zero_grad()
        actor_loss.backward()

        if main_process:
            self.loss_dic = [actor_loss, critic_loss]
        else:
            self.hidden_state = torch.zeros(1,1,64).to(self.device)

        self.a_loss += float(actor_loss.cpu().detach().numpy())
        self.c_loss += float(critic_loss.cpu().detach().numpy())
        return actor_loss.detach(), critic_loss.detach()

    def add_gradient(self, shared_grad_buffer):
        # add the gradient to the shared_buffer...
        shared_grad_buffer.add_gradient(self.actor)


    def update(self, shared_grad_buffer_grads, worker_num):     

        for n, p in self.actor.named_parameters():
            p.grad = Variable(shared_grad_buffer_grads[n + '_grad'])
        self.actor_optimizer.step()

        self.hidden_state = torch.zeros(1,1,64).to(self.device)

        return self.c_loss, self.a_loss

    def get_loss(self):
        return self.c_loss, self.a_loss

    def get_actor(self):
        return self.actor

    def reset_loss(self):
        self.a_loss = 0
        self.c_loss = 0

    def copy_memory(self, sample_mem):
        self.memory_our_enemy = sample_mem
    
    def clear_memory(self):
        self.memory_our_enemy[0].clear_memory()
        self.memory_our_enemy[1].clear_memory()

    def load_model(self, run_dir, episode):
        print(f'\nBegin to load model: ')
        base_path = os.path.join(run_dir, 'trained_model')
        print("base_path",base_path)

        model_actor_path = os.path.join(base_path, "actor_"  + ".pth")
        print(f'Actor path: {model_actor_path}')

        if  os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=self.device)
            self.actor.load_state_dict(actor)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')

    def save_model(self, run_dir, episode):
        print("---------------save-------------------")
        base_path = os.path.join(run_dir, 'trained_model')
        print("new_lr: ",self.a_lr)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_"  + ".pth") #+ str(episode)
        torch.save(self.actor.state_dict(), model_actor_path)



#this is used to accumulate the gradients
class Shared_grad_buffers:
    def __init__(self, models, main_device):
        self.device = main_device
        self.grads = {}
        for name, p in models.named_parameters():
            self.grads[name + '_grad'] = torch.zeros(p.size()).share_memory_().to(self.device)


    def add_gradient(self, models):
        for name, p in models.named_parameters():
            #print("name, p",name,p)
            self.grads[name + '_grad'] += p.grad.data.to(self.device)

    def reset(self):
        for name, grad in self.grads.items():
            self.grads[name].fill_(0)