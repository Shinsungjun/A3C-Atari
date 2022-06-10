import gym
from multiprocessing import Process, Queue, Value
import torch
from models.model import A3C
def train(rank=None, shared_model = None):
    env = gym.make("Tennis-v4")
    model = A3C(env.observation_space.shape[2], env.action_space)
    while True:
        #* Load Updated Model
        model.load_state_dict(shared_model.state_dict())
        
        observation = env.reset()
        while True:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print(f"Episode finished rank : {rank}")
                break