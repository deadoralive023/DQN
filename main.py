import gym
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from random import sample
import random
from collections import deque
import os
import ipdb
import wandb
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self, num_actions, lr=0.00025):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),

             # nn.Linear(256, num_of_actions)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)

        )
        self.optim = optim.Adam(self.parameters(), lr=lr)

    def forward(self, input):
        input = input / 255
        output = self.conv_layers(input)
        output = output.view(-1, 64*7*7)
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
        (thresh, frame) = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)
        frame = frame[30:frame.shape[0], 0:frame.shape[1]]
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        #frame = frame / 255
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
    def __init__(self, eps=1.0, eps_end=0.1, eps_decay=0.0000009):
        self.eps = eps
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def epsilon_greedy(self, step):
        self.eps = max(self.eps_end, self.eps - self.eps_decay)
        return self.eps


class Agent:
    def __init__(self, env, gamma=0.99, batch_size=2500, buffer_replay_capacity=100000, frame_skip=4, start_train_after=1, no_op=30, device='cuda:0'):
        self.env = env
        self.pp = PreProcess()
        self.br = BufferReplay(buffer_replay_capacity, device)
        self.es = ExplorationStrategy()
        self.value_net = Net(env.action_space.n).to(device)
        self.target_net = Net(env.action_space.n).to(device)
        self.update_target_net()
        self.target_net.eval()
        self.gamma = gamma
        self.batch_size = batch_size
        self.frame_skip = frame_skip
        self.start_train_after = start_train_after
        self.no_op = no_op
        self.device = device
        self.prev_obs = None
        self.env_reset()


    def step(self, step):
        prob_rand_act = random.uniform(0, 1)
        if step > self.start_train_after:
            if prob_rand_act < self.es.epsilon_greedy(step):
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    input = torch.Tensor(self.prev_obs).to(self.device)
                    input = torch.unsqueeze(input, 0)
                    action =  torch.argmax(self.value_net(input)).detach().item()
        else:
            action = self.env.action_space.sample()
        for i in range(self.frame_skip + 1):
            obs, reward, done, _ = self.env.step(action)
            obs = self.pp.pre_process(obs)
            obs = self.pp.stack_frames(obs)
            #reward = self.clip_reward(reward)
            self.br.add([self.prev_obs, action, reward, obs, int(done)])
            self.prev_obs = obs
        return obs, reward, done

    def clip_reward(self, reward):
        if reward > 0:
            return 1
        elif reward < 0:
            return -1
        return 0


    def update_target_net(self):
        self.target_net.load_state_dict(self.value_net.state_dict())

    def env_reset(self):
        self.env.reset()
        for _ in range(self.no_op):
            self.prev_obs, _, _, _ = self.env.step(0)
        self.prev_obs = self.pp.pre_process(self.prev_obs)
        self.prev_obs = self.pp.stack_frames(self.prev_obs)

    def save(self, name):
        torch.save(self.value_net, 'models/' + str(name) + '.pt')

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
    BATCH_SIZE = 256
    BUFF_REPLAY_CAP = 150000
    TRAIN_FRQ = 1
    TARG_NET_UPDATE_FRQ = 10000
    FRAME_SKIP = 3
    NO_OP = 70
    START_TRAIN_AFTER = 50000
    SAVE_MODEL_AFTER = 500000
    agent = Agent(gym.make('PongNoFrameskip-v4'), batch_size=BATCH_SIZE, buffer_replay_capacity=BUFF_REPLAY_CAP, frame_skip=FRAME_SKIP, start_train_after=START_TRAIN_AFTER, no_op=NO_OP, device='cpu')
    episodes_reward = deque([0.0], maxlen=100)
    rolling_reward = 0
    global_step = 0
    episode = 0
    wandb.init()
    while True:
        obs, reward, done = agent.step(global_step)
        rolling_reward += reward
        if done:
            episode += 1
            agent.env_reset()
            episodes_reward.append(rolling_reward)
            rolling_reward = 0

        global_step += 1
        if global_step > START_TRAIN_AFTER:
            if global_step % TRAIN_FRQ == 0:
                loss = agent.learn()
                wandb.log({'loss':loss, 'avg-reward': np.mean(episodes_reward), 'eps':agent.es.eps, 'episode':episode}, step=global_step-START_TRAIN_AFTER)
            if global_step  % TARG_NET_UPDATE_FRQ == 0:
                agent.update_target_net()
            if global_step % SAVE_MODEL_AFTER == 0:
                agent.save(global_step)
        if render:
            agent.env.render()
            time.sleep(0.005)

if __name__ == '__main__':
    main()
