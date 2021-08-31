import gym
import time
import cv2
import numpy as np
from collections import deque
import ipdb
import keyboard as k


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
        #frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        #frame = frame / 255
        return frame


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
    main()

