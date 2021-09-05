import gym
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from random import sample, random
from collections import deque
import os
import ipdb
import wandb

class Net(nn.Module):
    def __init__(self, num_actions, lr=2.5e-4):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=4, padding=0),
            nn.ReLU(),
             # nn.Linear(256, num_of_actions)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(32*9*9, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        self.optim = optim.Adam(self.parameters(), lr=lr)

    def forward(self, input):
        output = self.conv_layers(input)
        output = output.view(-1, 32*9*9)
        output = self.linear_layers(output)
        return output


class PreProcess:
    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.prev_frames = deque(maxlen=stack_size)

    def stack_frames(self, frame):
        if len(self.prev_frames) == 0:
            self.prev_frames.extend([frame] * (self.stack_size - 1))
        self.prev_frames.append(frame)
        return np.stack(self.prev_frames, 0)

    def pre_process(self, frame, dim=(84,84)):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        frame = frame / 255
        return frame

class BufferReplay:
    def __init__(self, capacity=100000, device='cuda:0'):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.index = 0
        self.device = device

    def add(self, exp):
        if self.index >= self.capacity:
            self.index = 0
        self.buffer[self.index] = exp
        self.index += 1

    def sample(self, size):
        size = min(size, self.index if self.buffer[-1] == None else self.capacity)
        return sample(self.buffer[:size], size)

    def discretized_sample(self, size):
        size = min(size, self.index if self.buffer[-1] == None else self.capacity)
        _sample = sample(self.buffer[:(self.index if self.buffer[-1] == None else self.capacity)], size)
        states = torch.Tensor([s[0] for s in _sample]).to(self.device)
        actions = torch.tensor([s[1] for s in _sample]).to(self.device)
        rewards = torch.Tensor([s[2] for s in _sample]).to(self.device)
        next_states = torch.Tensor([s[3] for s in _sample]).to(self.device)
        terminals = torch.Tensor([s[4] for s in _sample]).to(self.device)
        return [states, actions, rewards, next_states, terminals]

class ExplorationStrategy:
    def __init__(self, eps=1.0, eps_end=0.1, eps_decay=0.999997):
        self.eps = eps
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def epsilon_greedy(self, step):
        self.eps = self.eps_decay ** step
        return self.eps


class Agent:
    def __init__(self, env, gamma=0.99, batch_size=2500, buffer_replay_capacity=100000, frame_skip=4, device='cuda:0'):
        self.env = env
        self.pp = PreProcess()
        self.br = BufferReplay(buffer_replay_capacity, device)
        self.es = ExplorationStrategy()
        self.value_net = Net(env.action_space.n).to(device)
        self.target_net = Net(env.action_space.n).to(device)
        self.update_target_net()
        self.target_net.eval()
        self.prev_obs = self.env.reset()
        self.prev_obs = self.pp.pre_process(self.prev_obs)
        self.prev_obs = self.pp.stack_frames(self.prev_obs)
        self.gamma = gamma
        self.batch_size = batch_size
        self.frame_skip = frame_skip
        self.device = device

    def step(self, step):
        if random() < self.es.epsilon_greedy(step):
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                input = torch.Tensor(self.prev_obs).to(self.device)
                input = torch.unsqueeze(input, 0)
                action =  torch.argmax(self.value_net(input)).detach().item()
        for i in range(self.frame_skip + 1):
          obs, reward, done, _ = self.env.step(action)
        obs = self.pp.pre_process(obs)
        obs = self.pp.stack_frames(obs)
        self.br.add([self.prev_obs, action, reward, obs, int(done)])
        self.prev_obs = obs
        return obs, reward, done


    def update_target_net(self):
        self.target_net.load_state_dict(self.value_net.state_dict())

    def env_reset(self):
        self.prev_obs = self.env.reset()
        self.prev_obs = self.pp.pre_process(self.prev_obs)
        self.prev_obs = self.pp.stack_frames(self.prev_obs)

    def learn(self):
        sample = self.br.discretized_sample(self.batch_size)
        # loss => R(s, a) + gamma * maxQ(s`, a*) - Q(s, a)
        q_sa = self.value_net(sample[0]).to(self.device)
        acts = sample[1].view(-1, 1)
        q_sa = q_sa.gather(1, acts)
        q_sa = torch.squeeze(q_sa)
        with torch.no_grad():
            max_q = self.target_net(sample[3]).to(self.device)
        max_q, _ = torch.max(max_q, dim=1)
        loss = torch.mean(((sample[2] + (1 - sample[4]) * self.gamma * max_q) -  q_sa)**2)

        self.value_net.optim.zero_grad()
        loss.backward()
        self.value_net.optim.step()
        return loss.detach().item()

def main(render=False):
    process = psutil.Process(os.getpid())
    BATCH_SIZE = 32
    BUFF_REPLAY_CAP = 150000
    TRAIN_FRQ = 4
    TARG_NET_UPDATE_FRQ = 1000
    FRAME_SKIP = 3
    agent = Agent(gym.make('BreakoutDeterministic-v4'), batch_size=BATCH_SIZE, buffer_replay_capacity=BUFF_REPLAY_CAP, frame_skip=FRAME_SKIP, device='cuda:0')
    episodes_reward = deque([0.0], maxlen=100)
    rolling_reward = 0
    global_step = 0
    wandb.init()
    while True:
        obs, reward, done = agent.step(global_step)
        rolling_reward += reward
        if done:
            agent.env_reset()
            episodes_reward.append(rolling_reward)
            rolling_reward = 0

        global_step += 1
        #if global_step > BUFF_REPLAY_CAP:
        if global_step % TRAIN_FRQ == 0:
            loss = agent.learn()
            print(len(episodes_reward), len(agent.br.buffer), process.memory_info().rss)
            wandb.log({'loss':loss, 'avg-reward': np.mean(episodes_reward), 'eps':agent.es.eps}, step=global_step)
        if global_step  % TARG_NET_UPDATE_FRQ == 0:
            agent.update_target_net()
        if render:
            agent.env.render()
            time.sleep(0.005)



if __name__ == '__main__':
    main()
