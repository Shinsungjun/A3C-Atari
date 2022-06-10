import gym
from multiprocessing import Process, Queue, Value
import torch
from models.model import A3C
from utils.util import *
from main_thread import *
from train import train
import argparse

def parsing__arguments():
    parser = argparse.ArgumentParser(description='KAU - Reinforcement Learning')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                        help='parameter for GAE (default: 1.00)')
    parser.add_argument('--num_processes', type=int, default=4, metavar='N',
                        help='how many training processes to use (default: 4')
    
    return parser.parse_args()

def main(args):
    ## Initial Model, Buffer 
    ## threading 
    print_title()
    
    env = gym.make("Tennis-v4")
    
    print_info(env)
    
    # * Make Shared Model (pytorch share_memory)
    shared_model = A3C(env.observation_space.shape[2], env.action_space)
    shared_model.share_memory()
    
    sharedcheck = Value('d', 0)
    
    processes = []
    for rank in range(args.num_processes):
        p = Process(target=train, args = (rank, shared_model))
        p.start()
        processes.append(p)
    
    main_thread(env, shared_model)
    
    for p in processes:
        p.join()
    
if __name__ == "__main__":
    args = parsing__arguments()
    main(args)