import torch
import cv2 as cv
def print_title():
    print("*************************************")
    print("   Training for Atari - Tennis v4")
    print("*************************************\n")
    
    
def print_info(env):
    print("Environment Information")
    print(f"Observation Space Shape : {env.observation_space.shape}")
    print(f"Action Space : {env.action_space.n}")
    
    
def encoding_observation(observation):
    # cv.imwrite('oberv.png', observation)
    observation = observation[35:-16, :]
    # cv.imwrite('oberv2.png', observation)
    
    #observation = cv.resize(observation, dsize=(0,0), fx = 0.5, fy = 0.5, interpolation=cv.INTER_AREA)
    observation = torch.from_numpy(observation) # H, W, C
    observation = observation.permute(2, 0, 1) # C, H, W (Tennis : 3, 210, 160)
    observation = observation / 255.0
    observation = observation.unsqueeze(0)
    return observation