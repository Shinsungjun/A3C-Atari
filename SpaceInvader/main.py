import gym
from multiprocessing import Process, Queue, Value
import torch
from models.model import A3C
from utils.util import *
from main_thread import *
from train import train
import argparse
import wandb
wandb.login()


def parsing__arguments():
    parser = argparse.ArgumentParser(description='KAU - Reinforcement Learning')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--num_processes', type=int, default=4, metavar='N',
                        help='how many training processes to use (default: 4')
    parser.add_argument('--train_step', type=int, default=20, 
                        help='number of forward steps before update model once (default: 20)')
    
    return parser.parse_args()

def main(args):
    ## Initial Model, Buffer 
    ## threading 
    wandb.init(project='A3C-Atari-Sungjun',name='SpaceInvaders-v4', config=args, sync_tensorboard=True, settings=wandb.Settings(start_method='thread', console="off"))
    
    print_title()
    torch.manual_seed(8967)
    env = gym.make("SpaceInvaders-v4")
    
    print_info(env)
    
    # * Make Shared Model (pytorch share_memory)
    shared_model = A3C(env.observation_space.shape[2], env.action_space)
    shared_model.share_memory()
    
    processes = []
    for rank in range(args.num_processes):
        p = Process(target=train, args = (args, rank, shared_model))
        p.start()
        processes.append(p)
    
    main_thread(args, env, shared_model)
    
    for p in processes:
        p.join()
    
if __name__ == "__main__":
    args = parsing__arguments()
    main(args)