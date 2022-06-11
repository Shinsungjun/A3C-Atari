import gym
from multiprocessing import Process, Queue, Value
import torch
import torch.nn.functional as F
from models.model import A3C
import time
from utils.util import *
def main_thread(args, env, shared_model):
    best_reward = -10000
    episode = 0
    
    model = A3C(env.observation_space.shape[2], env.action_space)
    observation = env.reset()
    state = encoding_observation(observation)
    done = True
    step = 0
    episode_reward = 0

    while True:
        

        model.load_state_dict(shared_model.state_dict())
        model.eval()
        
        if done:
            hx = torch.zeros((1, 256), requires_grad=False)
            
        else:
            hx = hx.clone().detach()
        

        for t in range(args.train_step):
            env.render()
            value, logit, hx = model(state, hx)
            prob = F.softmax(logit, dim=-1)
            action = prob.multinomial(num_samples=1).detach()

            observation, reward, done, info = env.step(action.numpy())
            episode_reward += reward
            if done:
                print("Episode finished after {} timesteps".format(step+1))
                print(f"Episode {episode} finished with {episode_reward} reward")
                observation_reset = env.reset()
                state = encoding_observation(observation_reset)
                
                if episode > 5 and episode % 5 == 0 :
                    print(f"Checkpoint Saving Model .... ")
                    torch.save(model.state_dict(), 'check_point.pth.tar')
                    
                if episode > 5 and best_reward < episode_reward:
                    best_reward = episode_reward
                    print(f"New Best Reward Episode ! Episode : {episode} Reward : {episode_reward}")
                    print(f"Saving Model .... ")
                    torch.save(model.state_dict(), 'best_reward.pth.tar')
                    #* save model
                    
                episode += 1
                episode_reward = 0
                step = 0
                break
            
            state = encoding_observation(observation)
            step += 1
            
        
            
        
            
        
            
            
            