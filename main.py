import gym
import time
import keyboard as k


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
        brea

