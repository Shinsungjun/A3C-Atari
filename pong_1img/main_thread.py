import gym
from multiprocessing import Process, Queue, Value
import torch
import torch.nn.functional as F
from models.model import A3C
import time
from utils.util import *
import wandb

def main_thread(args, env, shared_model):
    best_reward = -10000
    episode = 0
    
    model = A3C(env.observation_space.shape[2], env.action_space)
    observation = env.reset()
    state = encoding_observation(observation)
    done = True
    step = 0
    episode_reward = 0
    early_stop = False

    while True:
        model.load_state_dict(shared_model.state_dict())
        model.eval()
        
        if done:
            hx = torch.zeros((1, 256), requires_grad=False)
            
        else:
            hx = hx.clone().detach()
        
        while True:
            env.render()
            value, logit, hx = model(state, hx)
            prob = F.softmax(logit, dim=-1)
            # print("prob shape : ", prob.shape)
            action = prob.max(1)[1].detach().numpy() + 1 #PingPong 1 ~ 3
            # action = prob.max(1)[1].detach().numpy() + 10 # Tennis 0 ~ 17
            # action = prob.multinomial(num_samples=1).detach() + 10
            
            # print("action : ", action)
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            
            # if episode_reward < -5:
            #     done = True
            #     early_stop = True
            if done:
                print("Episode finished after {} timesteps".format(step+1))
                if early_stop:
                    print(f"early stopping")
                print(f"Episode {episode} finished with {episode_reward} reward")
                wandb.log({'main_model_reward' : episode_reward})
                
                observation_reset = env.reset()
                state = encoding_observation(observation_reset)
                
                if episode > 5 and episode % 5 == 0 :
                    print(f"Checkpoint Saving Model .... ")
                    torch.save(model.state_dict(), 'check_point.pth.tar')
                    
                if best_reward < episode_reward and early_stop == False:
                    best_reward = episode_reward
                    print(f"New Best Reward Episode ! Episode : {episode} Reward : {episode_reward}")
                    print(f"Saving Model .... ")
                    torch.save(model.state_dict(), 'best_reward.pth.tar')
                    #* save model
                    
                episode += 1
                episode_reward = 0
                step = 0
                early_stop = False
                break
            
            state = encoding_observation(observation)
            step += 1
            time.sleep(0.01)
        
            
        
            
        
            
            
            
            
'''
PingPong
    action 0 -> stay
    action 1 -> stay
    action 2 -> up
    action 3 -> down
    action 4 -> up
    aciton 5 -> down
    
    ----
    
    3 action define.
    action 1, action 2, action 3
'''

'''
Tennis
    action 0 -> stay
    action 1 -> hit
    action 2 -> move up
    action 3 -> move right
    action 4 -> move left
    action 5 -> move down
    ...
    
'''