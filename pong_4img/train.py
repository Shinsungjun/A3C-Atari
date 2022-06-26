import gym
from multiprocessing import Process, Queue, Value
import torch
import torch.nn.functional as F
from models.model import A3C
import torch.optim as optim
from utils.util import *
import wandb

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad
        
def train(args, rank=None, shared_model = None):
    env = gym.make("Pong-v4")
    env.seed(8967 + rank * 5)

    states = []
    model = A3C(3*4, env.action_space)
    optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)    
    episode = 0
    
    # * Environment Setting before training * #
    observation = env.reset()
    state = encoding_observation(observation) #1, C, H, W    
    done = True
    episode_reward = 0
    lives = 3
    
    states = []
    states.append(state)
    
    while True:
        #* Load Updated Model
        model.load_state_dict(shared_model.state_dict())
        episode += 1
        if done:
            states = []
            hx = torch.zeros((1, 256), requires_grad=True)
            
        else:
            hx = hx.clone().detach().requires_grad_(True)
        
        values = []
        log_probs = []
        rewards = []
        entropies = []
        
        for step in range(args.train_step):
            if len(states) >= 4:
                if len(states) > 4:
                    _ = states.pop(0)
                state_input = torch.cat(states, dim=1)
                
                # * Inference model to get value and action * #
                value, policy, hx = model(state_input, hx)
                prob_policy = F.softmax(policy, dim=-1)
                log_prob_policy = F.log_softmax(policy, dim=-1)
                
                entropy = -(log_prob_policy * prob_policy).sum(1, keepdim=True)
                
                action = prob_policy.multinomial(num_samples=1).detach()   
                    
                log_prob_policy = log_prob_policy.gather(1, action)
                
                #action = env.action_space.sample()
                observation, reward, done, info = env.step(action.numpy() + 1) #PingPong
                
                # if lives != info['ale.lives']:
                #     lives = info['ale.lives']
                #     reward -= 1
                
                #* Clip Reward -1 ~ +1 *#
                reward = max(min(reward, 1), -1) #pong : -1 or 1 #Space -1 or 0.1
                episode_reward += reward
                entropies.append(entropy)
                values.append(value)
                log_probs.append(log_prob_policy)
                rewards.append(reward)
            
            else:
                observation, reward, done, info = env.step(1) #PingPong
                
            
            if done:
                observation_reset = env.reset()
                state = encoding_observation(observation_reset)
                lives = 3
            
            else:
                state = encoding_observation(observation)
                states.append(state)
            
            if done:
                print(f"Episode finished rank : {rank}")
                agent_name = 'agent_' + str(rank) + '_reward'
                wandb.log({agent_name : episode_reward})
                episode_reward = 0
                break
        
        R = torch.zeros(1, 1)
        
        if not done:
            if len(states) > 4:
                    _ = states.pop(0)
            state_input = torch.cat(states, dim=1)
            value, _, _ = model(state_input, hx)
            R = value.detach()
        
        values.append(R)
        policy_loss = 0
        value_loss = 0
        
        for i in reversed(range(len(rewards))):
            if (rank == 0 or rank == 1) and i == 0 :{
                print(f"rank : {rank}, {rewards}")
            }
            
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            
            Q_value = R - values[i]
            
            policy_loss = policy_loss - log_probs[i] * Q_value.detach() - 0.01 * entropies[i]
        
        optimizer.zero_grad()
        
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 30)

        ensure_shared_grads(model, shared_model)
        optimizer.step()