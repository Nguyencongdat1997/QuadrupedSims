import gym
from quadruped_env import QuadrupedEnvironment
import time

env = QuadrupedEnvironment()
for ep in range(2):
    observation = env.reset()
    total_reward = 0
    done = False
    while not done:
        env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print("Episode {} finished after {} timesteps with reward {}".format(ep+1, env.run_steps+1, total_reward))
            break
env.close()