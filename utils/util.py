def print_title():
    print("*************************************")
    print("   Training for Atari - Tennis v4")
    print("*************************************\n")
    
    
def print_info(env):
    print("Environment Information")
    print(f"Observation Space Shape : {env.observation_space.shape}")
    print(f"Action Space : {env.action_space.n}")