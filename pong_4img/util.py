import torch
import cv2 as cv
import wandb
from PIL import Image

count=0
def print_title():
    print("*************************************")
    print("          Training for Atari          ")
    print("*************************************\n")
    
    
def print_info(env):
    print("Environment Information")
    print(f"Observation Space Shape : {env.observation_space.shape}")
    print(f"Action Space : {env.action_space.n}")

def encoding_observation(observation):
    global count
    
    state = observation[35:-16, :]
    
    #* code for wandb image logging *#
    if count == 5:
        vis_obs = Image.fromarray(observation)
        vis_state = Image.fromarray(state)
    
        vis_obs = wandb.Image(vis_obs, caption="Observation")
        vis_state = wandb.Image(vis_state, caption="State")
    
        wandb.log({"Observation": vis_obs})
        wandb.log({"State": vis_state})
        
    count += 1
    #********************************#
    
    
    state = cv.resize(state, dsize=(160,160), interpolation=cv.INTER_AREA)
    
    state = torch.from_numpy(state) # H, W, C
    state = state.permute(2, 0, 1) # C, H, W (Origin Size : 3, 210, 160)
    state = state / 255.0 #Normalize
    state = state.unsqueeze(0) # B, C, H, W
    return state