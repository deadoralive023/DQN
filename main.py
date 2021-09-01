import gym
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from random import sample
from collections import deque
import ipdb
import keyboard as k


class Net(nn.Module):
    def __init__(self, num_actions, lr=0.0001):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=4, padding=0),
            nn.ReLU(),
             # nn.Linear(256, num_of_actions)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(32*10*10, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        self.optim = optim.Adam(self.parameters(), lr=lr)

    def forward(self, input):
        output = self.conv_layers(input)
        output = output.view(-1, 32*10*10)
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
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, exp):
        self.buffer.append(exp)

    def sample(self, size):
        size = min(size, len(self.buffer))
        return sample(self.buffer, size)

    def discretized_sample(self, size):
        size = min(size, len(self.buffer))
        indexes = sample(range(0, len(self.buffer)), size)
        states = torch.Tensor([self.buffer[i][0] for i in indexes])
        actions = torch.tensor([self.buffer[i][1] for i in indexes])
        rewards = torch.Tensor([self.buffer[i][2] for i in indexes])
        next_states = torch.Tensor([self.buffer[i][3] for i in indexes])
        terminals = torch.tensor([self.buffer[i][4] for i in indexes])
        return [states, actions, rewards, next_states, terminals]

class Agent:
    def __init__(self, env, gamma=1, batch_size=3):
        self.env = env
        self.pp = PreProcess()
        self.br = BufferReplay()
        self.value_net = Net(env.action_space.n)
        self.target_net = Net(env.action_space.n)
        self.copy_weights(self.target_net, self.value_net)
        self.target_net.eval()
        self.prev_obs = self.env.reset()
        self.prev_obs = self.pp.pre_process(self.prev_obs)
        self.prev_obs = self.pp.stack_frames(self.prev_obs)
        self.gamma = gamma
        self.batch_size = batch_size

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        obs = self.pp.pre_process(obs)
        obs = self.pp.stack_frames(obs)
        self.br.add([self.prev_obs, action, reward, obs, int(done)])
        self.prev_obs = obs
        if done:
            self.prev_obs = self.env.reset()
            self.prev_obs = self.pp.preprocess(self.prev_obs)
            self.prev_obs = self.pp.stack_frames(self.prev_obs)
        return obs, reward, done


    def copy_weights(self, net1, net2):
        net1.load_state_dict(net2.state_dict())


    def learn(self):
        sample = self.br.discretized_sample(self.batch_size)
        # loss => R(s, a) + gamma * maxQ(s`, a*) - Q(s, a)
        q_sa = self.value_net(sample[0])
        acts = sample[1].view(-1, 1)
        q_sa = q_sa.gather(1, acts)
        q_sa = torch.squeeze(q_sa)
        with torch.no_grad():
            max_q = self.target_net(sample[3])
        ipdb.set_trace()
        max_q, _ = torch.max(max_q, dim=1)
        loss = (((sample[2] + (1 - sample[4]) * self.gamma * max_q) - q_sa)**2).mean()

        self.value_net.optim.zero_grad()
        loss.backward()
        self.value_net.optim.step()
        return loss.detach().item()

def main(render=False):
    env = gym.make('Breakout-v0')
    agent = Agent(env)
    index = 0
    while True:
        action = env.action_space.sample()
        obs, reward, done = agent.step(action)
        #ipdb.set_trace()

        sample = agent.br.discretized_sample(10)
        output = agent.value_net(sample[0])
        index += 1
        if index == 10:
            agent.learn()
        if render:
            env.render()
            time.sleep(0.005)

        #if k.is_pressed('q'):
        #    break



if __name__ == '__main__':
    main()
    #env = gym.make('Breakout-v0')
    #pp = Preprocess()
    #obs = pp.pre_process(env.reset())
    #ipdb.set_trace()

    stacks = pp.stack_frames(obs)

    #obs_t = torch.Tensor(obs)
    #obs_t = torch.cat([obs_t, obs_t]).unsqueeze(1)
    #net = Net(4)
    #output = net(obs_t)
