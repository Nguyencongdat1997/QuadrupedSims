import gym
from quadruped_env import QuadrupedEnvironment
import time

env = QuadrupedEnvironment()
for ep in range(1):
    observation = env.reset()
    total_reward = 0
    for time_step in range(1000):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            env.reset()
            total_reward = 0
    print("Episode {} finished after {} timesteps with reward {}".format(ep+1, time_step+1, total_reward))
env.close()