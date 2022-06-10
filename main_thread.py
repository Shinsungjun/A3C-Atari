import gym
from multiprocessing import Process, Queue, Value
import torch
from models.model import A3C
import time

def main_thread(env, shared_model):
    best_reward = -10000
    episode = 0
    
    model = A3C(env.observation_space.shape[2], env.action_space)
    
    while True:
        observation = env.reset()
        model.load_state_dict(shared_model.state_dict())
        model.eval()
        episode += 1
        step = 0
        episode_reward = 0
        while True:
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                print("Episode finished after {} timesteps".format(step+1))
                break
            step += 1
            time.sleep(0.01)
        
        if episode % 10 == 0:
            print(f"Episode {episode} finished with {episode_reward} reward")
            
        if best_reward < episode_reward:
            best_reward = episode_reward
            print(f"New Best Reward Episode ! Episode : {episode} Reward : {episode_reward}")
            print(f"Saving Model .... ")
            #* save model
            
        
            
            
            