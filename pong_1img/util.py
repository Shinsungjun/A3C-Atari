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
    
    
# def encoding_observation(observation):
#     # cv.imwrite('oberv.png', observation)
#     observation = observation[30:-20, 20:-20]
#     # cv.imwrite('oberv2.png', observation)
    
#     observation = cv.resize(observation, dsize=(160,160), interpolation=cv.INTER_AREA)
#     # cv.imwrite('oberv3.png', observation)
    
#     observation = torch.from_numpy(observation) # H, W, C
#     observation = observation.permute(2, 0, 1) # C, H, W (Tennis : 3, 210, 160)
#     observation = observation / 255.0
#     observation = observation.unsqueeze(0)
#     return observation

def encoding_observation(observation):
    global count
    if count == 10:
        vis_obs = Image.fromarray(observation)
    
    # cv.imwrite('oberv.png', observation)
    state = observation[35:-16, :]
    if count == 10:
        vis_state = Image.fromarray(state)
    #images_array.append(observation)
        vis_obs = wandb.Image(vis_obs, caption="Observation")
        vis_state = wandb.Image(vis_state, caption="State")
    
        wandb.log({"Observation": vis_obs})
        wandb.log({"State": vis_state})
        
    count += 1

    # cv.imwrite('oberv2.png', observation)
    
    #observation = cv.resize(observation, dsize=(0,0), fx = 0.5, fy = 0.5, interpolation=cv.INTER_AREA)
    state = torch.from_numpy(state) # H, W, C
    state = state.permute(2, 0, 1) # C, H, W (Tennis : 3, 210, 160)
    state = state / 255.0
    state = state.unsqueeze(0)
    return state