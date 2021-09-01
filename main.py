import gym
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import sample
from collections import deque
import ipdb
import keyboard as k


class Net(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=0),
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

    def forward(self, input):
        output = self.conv_layers(input)
        output = output.view(-1, 32*10*10)
        output = self.linear_layers(output)
        return output


class Preprocess:
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
        frame = frame[np.newaxis, :, :]
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



def main():
    env = gym.make('Breakout-v0')
    obs = env.reset()

    while True:
        env.render()
        time.sleep(0.005)
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()
        if k.is_pressed('q'):
            break
        index += 1



if __name__ == '__main__':
    #main()
    env = gym.make('Breakout-v0')
    #pp = Preprocess()
    #obs = pp.pre_process(obs)
    #ipdb.set_trace()
    #obs_t = torch.Tensor(obs)
    #obs_t = torch.cat([obs_t, obs_t]).unsqueeze(1)
    #net = Net(4)
    #output = net(obs_t)
