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
from replay_buffer import ReplayBuffer
from common import soft_update, hard_update
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
        self.step = []
        self.value = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.hidden_state[:]
        del self.step[:]
        del self.value[:]

class Actor(nn.Module):
    def __init__(self):  #(n+2p-f)/s + 1 
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(4,8, kernel_size=(6,3), stride=1, padding=1) # 20104 -> 20104
        self.conv2 = nn.Conv2d(8,8, kernel_size=(6,3), stride=1, padding=1) # 20104 -> 20104
        self.conv3 = nn.Conv2d(8,8, kernel_size=(6,3), stride=1, padding=1) # 20104 -> 20104
        self.conv4 = nn.Conv2d(8,8, kernel_size=(6,3), stride=1, padding=1) # 20104 -> 20104
        self.self_attention = nn.MultiheadAttention(642, 2)

        self.gru = nn.GRU(642, 64, 1) 
        self.critic_linear = nn.Linear(64, 1)
        self.linear = nn.Linear(64, 12)
        # self.linear_1 = nn.Linear(64, 4)
        # self.linear_2 = nn.Linear(64, 4)
        # self.linear_3 = nn.Linear(64, 4)
        #
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical

    def forward(self, tensor_cv, step, h_old): #,batch_size
        # CV
        self.batch_size = tensor_cv.size()[0]
        i_1 = tensor_cv
        # CV
        x = F.elu(self.conv1(i_1))
        #i_2 = i_1 + x
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x)).reshape(1,self.batch_size,640)#(1,self.batch_size,640)
        

        step = step.reshape(1,self.batch_size,2)

        x = torch.cat([x, step], -1)
        x = self.self_attention(x,x,x)[0] + x
        x,h_state = self.gru(x, h_old)
        
        value = self.critic_linear(x)
        #y.detach()
        #x = x.reshape(self.batch_size,642)
        x = self.linear(x)
        # a = self.linear_1(x)
        # b = self.linear_2(x)
        # c = self.linear_3(x)
        #x = torch.stack([a,b,c],1)
        

        action_probs = torch.softmax( x.reshape(self.batch_size,3,4) ,dim =-1)
        dis = self.Categorical(action_probs)
        action = dis.sample()#.view(-1,1)
        z = (action_probs == 0.0).float()*1e-8
        
        log_prob = torch.log(action_probs+z)# * action_probs 
        entropy = -torch.sum(action_probs.reshape(self.batch_size,3,4)*log_prob.reshape(self.batch_size,3,4),dim=-1,keepdim=True)
        
        return action,log_prob,entropy, h_state.data, value.reshape(self.batch_size,1,1)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.act_dim = 3

        # Q1 architecture
        self.conv1_1 = nn.Conv2d(4,8, kernel_size=(6,3), stride=1, padding=1) # 20104 -> 17108
        self.conv2_1 = nn.Conv2d(8,8, kernel_size=(6,3), stride=1, padding=1) # 17108 -> 141016
        self.conv3_1 = nn.Conv2d(8,8, kernel_size=(6,3), stride=1, padding=1) # 141016-> 111032
        self.conv4_1 = nn.Conv2d(8,8, kernel_size=(6,3), stride=1, padding=1) # 16632 -> 14464
        self.self_attention_1 = nn.MultiheadAttention(642, 2)
        self.linear_1 = nn.Linear(642,1)
        #self.lstm_1 = nn.LSTM(640,12, 1) 
        ##########################################
        # Q2 architecture
        self.conv1_2 = nn.Conv2d(4,8, kernel_size=(6,3), stride=1, padding=1) # 20104 -> 17108
        self.conv2_2 = nn.Conv2d(8,8, kernel_size=(6,3), stride=1, padding=1) # 17108 -> 141016
        self.conv3_2 = nn.Conv2d(8,8, kernel_size=(6,3), stride=1, padding=1) # 141016-> 111032
        self.conv4_2 = nn.Conv2d(8,8, kernel_size=(6,3), stride=1, padding=1) # 16632 -> 14464
        self.self_attention_2 = nn.MultiheadAttention(642, 2)
        self.linear_2 = nn.Linear(642,1)

    def forward(self, tensor_cv, step):
        # CV
        batch_size = tensor_cv.size()[0]
        i = tensor_cv
        x = F.relu(self.conv1_1(i))
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv4_1(x))
        #
        x=  x.reshape(1,batch_size,640)

        step = step.reshape(1,batch_size,2)
        x = torch.cat([x, step], -1)
        #x = self.self_attention_1(x,x,x)[0] + x
        x = x.reshape(batch_size,642)

        out_1 = torch.tanh(self.linear_1(x)).reshape(batch_size,1,1)

        ###################################################################
        # CV
        i = tensor_cv
        x = F.relu(self.conv1_2(i))
        #i = i + x
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv4_2(x))
        #
        x =  x.reshape(1,batch_size,640)
        

        step = step.reshape(1,batch_size,2)
        x = torch.cat([x, step], -1)
        #x = self.self_attention_2(x,x,x)[0] + x
        x = x.reshape(batch_size,642)

        out_2 = torch.tanh(self.linear_2(x)).reshape(batch_size,1,1)

       
        return out_1,out_2 

class PPO:
    def __init__(self,  args, device):
        self.device = device
        self.a_lr = args.a_lr
        self.c_lr = args.c_lr
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.model_episode = args.model_episode
        self.eps = args.epsilon
        self.decay_speed = args.epsilon_speed
        self.output_activation = args.output_activation

        # Initialise actor network and critic network with ξ and θ
        self.actor = Actor().to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.critic = Critic().to(self.device)
        #self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)
        self.SmoothL1Loss = torch.nn.SmoothL1Loss()

        #
        self.memory = Memory()
        self.hidden_state = torch.zeros(1,1,64).to(self.device)
        #
        self.c_loss = 0
        self.a_loss = 0
        
        self.eps_clip = 0.1
        self.vf_clip_param = 10
        self.lam = 0.95
        self.batch_sample = 60
        self.old_value_1, self.old_value_2 = 0,0
        
        
        #
        self.shared_loss = 0
        self.loss_dic = [0,0]
        self.advantages = []
        self.target_value = []

    # Random process N using epsilon greedy
    def choose_action(self, obs, step, evaluation=False):
        self.memory.states.append(obs)
        step = torch.tensor(step, device = self.device) #np.full(1, step/100)
        obs = torch.Tensor([obs]).to(self.device)
        if len(self.memory.hidden_state)==0:
            self.memory.hidden_state.append(self.hidden_state.cpu().detach().numpy())

        action,action_logprob,_,self.hidden_state, value = self.actor(obs, step, self.hidden_state)

        self.memory.actions.append(action.cpu().detach().numpy())
        self.memory.logprobs.append(action_logprob.cpu().detach().numpy()) #[0]
        self.memory.hidden_state.append(self.hidden_state.cpu().detach().numpy())
        self.memory.step.append(step.cpu().detach().numpy())
        self.memory.value.append(value.cpu().detach().numpy())
        return action.cpu().detach().numpy()[0]


    def compute_GAE(self, training_time, main_process = False):
        batch_size = len(self.memory.actions)
        old_states = Variable( torch.tensor(self.memory.states).reshape(batch_size,4,20,10)).to(self.device).detach() #torch.squeeze(, 1)
        old_logprobs = Variable(torch.tensor(self.memory.logprobs).reshape(batch_size,3,4).to(self.device)).detach()
        old_actions = Variable(torch.tensor(self.memory.actions).reshape(batch_size,3,1).to(self.device)).detach()
        old_logprobs = Variable(old_logprobs.gather(-1,old_actions.long()) )

        old_hidden = Variable(torch.tensor(self.memory.hidden_state[:-1]).reshape(1,batch_size,64)).to(self.device).detach()
        old_step = Variable(torch.tensor(self.memory.step).reshape(1,batch_size,2).to(self.device)).detach()
        old_value = Variable(torch.tensor(self.memory.value).reshape(batch_size, 1, 1).to(self.device)).detach()


        if training_time ==0:
            #state_values_1,state_values_2 = self.critic(old_states, old_step)
            self.old_value_1 = old_value

            # Monte Carlo estimate of rewards:
            rewards = []
            GAE_advantage = []
            target_value = []
            #
            discounted_reward = 0#np.minimum(last_state_values_1[0].cpu().detach().numpy(),
                                #last_state_values_1[0].cpu().detach().numpy() )
            values_pre = 0#np.minimum(last_state_values_1[0].cpu().detach().numpy(),
                        #      last_state_values_1[0].cpu().detach().numpy() )
            advatage = 0
            for reward, is_terminal,values_1,values_2 in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals),
                                        reversed(self.old_value_1),reversed(self.old_value_1)): #反转迭代

                discounted_reward = reward +  self.gamma *discounted_reward 
                rewards.insert(0, discounted_reward) #插入列表

                values_1,values_2 = values_1.cpu().detach().numpy(),values_2.cpu().detach().numpy()
                #values = (values_1+values_2)/2
                values =values_1#
                #values = np.maximum(values_1,values_2)
                
                #delta = reward + values_pre - values
                #delta = discounted_reward - values 
                delta = reward + self.gamma*values_pre - values  #(1-is_terminal)*
                advatage = delta + self.gamma*self.lam*advatage 
                GAE_advantage.insert(0, advatage) #插入列表
                target_value.insert(0,float(values + advatage))#insert(0,float(reward + values_pre))
                values_pre = values
                #values_pre = np.minimum(values_1,values_2)
            
            # Normalizing the rewards:
            rewards = Variable(torch.tensor(rewards).to(self.device).reshape(batch_size,1,1))
            self.target_value = Variable(torch.tensor(target_value)).to(self.device).reshape(batch_size,1,1)
            GAE_advantage = Variable(torch.tensor(GAE_advantage)).to(self.device).reshape(batch_size,1,1)
            self.advantages = Variable((GAE_advantage- GAE_advantage.mean()) / (GAE_advantage.std() + 1e-6) )
            
            #GAE_advantage#

        #compute
        indices = torch.randint(batch_size, size=(self.batch_sample,), requires_grad=False)#, device=self.device

        old_states = old_states[indices]
        old_step = old_step.reshape(batch_size,1,2)[indices].reshape(1,self.batch_sample, 2)
        old_hidden = old_hidden.reshape(batch_size,1,64)[indices].reshape(1,self.batch_sample, 64)
        old_actions = old_actions[indices]
        old_logprobs = old_logprobs[indices]
        advantages = self.advantages[indices]
        old_value = self.old_value_1[indices]
        target_value = self.target_value[indices]

        _, logprobs, dist_entropy, _, value = self.actor(old_states, old_step, old_hidden)
        logprobs = logprobs.gather(-1,old_actions.long()) 
        #state_values_1,state_values_2 = self.critic(old_states, old_step)
        #value = state_values_1

        ratios = torch.exp(logprobs.reshape(self.batch_sample,3,1).sum(1,keepdim = True) - 
                           old_logprobs.reshape(self.batch_sample,3,1).sum(1,keepdim = True).detach())

        surr1 = ratios*advantages.detach()
        surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip)*advantages.detach() 
        #Dual_Clip
        surr3 = torch.min(surr1, surr2)#torch.max(torch.min(surr1, surr2),3*advantages.detach())
        #
        
        value_pred_clip = old_value.detach() +\
            torch.clamp(value -old_value.detach(), -self.eps_clip, self.eps_clip)#self.vf_clip_param
        critic_loss1 = (value - target_value.detach()).pow(2)
        critic_loss2 = (value_pred_clip - target_value.detach()).pow(2)
        critic_loss = 0.5 * torch.max(critic_loss1 , critic_loss2).mean()
        
        #critic_loss = (torch.nn.SmoothL1Loss()(state_values_1, rewards[indices]))/(rewards[indices].std() + 1e-6) 
        #critic_loss = torch.nn.SmoothL1Loss()(state_values_1, target_value) + torch.nn.SmoothL1Loss()(state_values_2, target_value)

        actor_loss = -surr3.mean() - 0.02*dist_entropy.mean() + 0.5 * critic_loss
        

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
    
    def get_critic(self):
        return self.critic

    def reset_loss(self):
        self.a_loss = 0
        self.c_loss = 0

    def copy_memory(self, sample_mem):
        self.memory = sample_mem
    
    def clear_memory(self):
        self.memory.clear_memory()

    def load_model(self, run_dir, episode):
        print(f'\nBegin to load model: ')
        base_path = os.path.join(run_dir, 'trained_model')
        print("base_path",base_path)

        model_actor_path = os.path.join(base_path, "actor_"  + ".pth")
        model_critic_path = os.path.join(base_path, "critic_"  + ".pth")
        print(f'Actor path: {model_actor_path}')
        print(f'Critic path: {model_critic_path}')

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=self.device)
            critic = torch.load(model_critic_path, map_location=self.device)
            self.actor.load_state_dict(actor)
            self.critic.load_state_dict(critic)
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

        model_critic_path = os.path.join(base_path, "critic_" + ".pth") #+ str(episode) 
        torch.save(self.critic.state_dict(), model_critic_path)


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
