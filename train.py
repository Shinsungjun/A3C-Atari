import gym
from multiprocessing import Process, Queue, Value
import torch
import torch.nn.functional as F
from models.model import A3C
import torch.optim as optim
from utils.util import *

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad
        
def train(args, rank=None, shared_model = None):
    env = gym.make("PongDeterministic-v4")
    env.seed(8967 + rank * 5)

    model = A3C(env.observation_space.shape[2], env.action_space)
    optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)    
    episode = 0
    
    # * Environment Setting before training * #
    observation = env.reset()
    state = encoding_observation(observation) #1, C, H, W    
    done = True
    while True:
        #* Load Updated Model
        model.load_state_dict(shared_model.state_dict())
        episode += 1
        if done:
            hx = torch.zeros((1, 256), requires_grad=True)
            
        else:
            hx = hx.clone().detach().requires_grad_(True)
        
        values = []
        log_probs = []
        rewards = []
        entropies = []
        
        for step in range(args.train_step):
            # * Inference model to get value and action * #
            value, logit, hx = model(state, hx)
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            if rank %2 == 0:
                action = prob.multinomial(num_samples=1).detach()
            else:
                action = prob.max(1)[1].detach()
                action = action.unsqueeze(0)                
                
            log_prob = log_prob.gather(1, action)
            
            #action = env.action_space.sample()
            observation, reward, done, info = env.step(action.numpy() + 1)
            
            # Clip Reward -1 ~ +1
            reward = max(min(reward, 1), -1)
            
            if done:
                observation_reset = env.reset()
                state = encoding_observation(observation_reset)
            
            else:
                state = encoding_observation(observation)
            
            entropies.append(entropy)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            
            if done:
                print(f"Episode finished rank : {rank}")
                break
        
        R = torch.zeros(1, 1)
        
        if not done:
            value, _, _ = model(state, hx)
            R = value.detach()
        
        values.append(R)
        policy_loss = 0
        value_loss = 0
        GAE = torch.zeros(1,1)
        
        for i in reversed(range(len(rewards))):
            if (rank == 0 or rank == 1) and i == 0 :{
                print(f"rank : {rank}, {rewards}")
            }
            
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            
            delta_t = rewards[i] + args.gamma * values[i+1] - values[i]
            GAE = GAE * args.gamma * args.tau + delta_t
            
            policy_loss = policy_loss - log_probs[i] * GAE.detach() - 0.01 * entropies[i]
        
        optimizer.zero_grad()
        
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 30)

        ensure_shared_grads(model, shared_model)
        optimizer.step()