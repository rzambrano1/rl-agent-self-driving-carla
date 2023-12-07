#!C:/Users/rzamb/anaconda3/envs/.carla/bin/python

#########
# Notes #
#########

######################
# Python Boilerplate #
######################

import os
import sys
import time

import math
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from torch import Tensor

from collections import namedtuple, deque
from itertools import count

import argparse

###################
# Pytorch Modules #
###################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale

#####################
# CARLA Environment #
#####################

import carla_gym_v6 # Custom environment created for this project
from carla_gym_v6 import manual_episode_end

#################
# Session Setup #
#################

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Forcing CPU
#device = torch.device("cpu")

#########################################    
# Custom Functions for image processing #
#########################################

# Note: Since the goal of this simulation is driving around without hitting obstacles (including other moving cars),
#       grayscale images are being used. To overcome temporal limitation problem four consecutive images are passed to
#       the neural network as part of the state; going from 12 channels to 4 reduces the computation adn thus reduces
#       runtime and hardware resources. Therefore, this agent ignores traffic lights.

def stack_img(img1:Tensor, img2:Tensor,img3:Tensor,img4:Tensor)->Tensor:
    """
    Input: Four images from CARLA camera sensor 
    Output: The four images stacked into a simgle tensor
    """
    return torch.cat((img1,img2,img3,img4),dim=0) 

def preprocess_img_stack(stacked_img:Tensor)->Tensor:
    """
    Input: A tensor with a stack of four 3-channel images
    Output: A tensor with a stack of four grayscale images
    """
    assert stacked_img.shape[0] == 4, "The image stack must be equal to 4, received {} images in the stack".format(stacked_img.shape[0])
    disagregated_imgs = []
    grayscale_imgs = []
    for img in range(4):
        disagregated_imgs.append(stacked_img[img])
    for img in disagregated_imgs:
        grayscale_imgs.append(rgb_to_grayscale(img, num_output_channels = 1))
    return torch.cat((grayscale_imgs[0],grayscale_imgs[1],grayscale_imgs[2],grayscale_imgs[3]),dim=0) 

######################################################################
# Deep Q-Network with capacity to handle temporal limitation problem #
######################################################################

class DQN_TempLimit(nn.Module):
    """This DQN is designed to take as an input four gray-scale images, each with 1 channel. 
    It aims to address the temporal limitation problem."""
    def __init__(self, n_actions):
        super(DQN_TempLimit, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=4, out_channels=12, kernel_size=4, stride=4)
        self.layer2 = nn.Conv2d(in_channels=12, out_channels=48, kernel_size=4, stride=4)
        self.layer3 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=2, stride=2)
        self.fc1 = nn.Linear(((96 * 20 * 15)+2), 128) 
        self.fc2 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, y):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        if len(y.shape)==1:
            x = torch.flatten(x)
            x = torch.cat((x,y))
        else:
            x = torch.flatten(x, start_dim=1)                              # Mod
            y = y.view(-1, 2)  # reshape y to match the dimensions of x    # Mod
            x = torch.cat((x, y), dim=1)                                   # Mod
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output

#################
# Main Function #
#################

def main(
        SHOW_PREVIEW,
        mode
):

    ############
    # Training #
    ############

    # Policy and Target Network instanciation

    # Get number of actions for the CARLA environment. Hardcoded item.
    n_actions = 8 # Equals len(carla_env.action_spac) # Originally 13, but all brake options were removed

    policy_net = DQN_TempLimit(n_actions).to(device)
    target_net = DQN_TempLimit(n_actions).to(device)
    #target_net.load_state_dict(policy_net.state_dict())

    # Next four lines load the latest weights for both policy and target network
    # Because of CARLA's 'unkown socket error' training stops unexpectedly. With these I can load the latest saved model
    # and continue training from there.
    STATE_DICT_PATH_POLICY = "C:/Users/rzamb/Documents/UMD/642_Robotics/finalProject/state_dict_experiment_5/policy_rgb_ep100.pth"
    STATE_DICT_PATH_TARGET = "C:/Users/rzamb/Documents/UMD/642_Robotics/finalProject/state_dict_experiment_5/target_rgb_ep100.pth"
    print("Previous weights loaded!")
    policy_net.load_state_dict(torch.load(STATE_DICT_PATH_POLICY))
    target_net.load_state_dict(torch.load(STATE_DICT_PATH_TARGET))

    #steps_done = 0

    def select_action(state):
        #nonlocal steps_done
        nonlocal policy_action_exploitation
        nonlocal random_action_exploration

        #steps_done += 1
        #policy_action_exploitation += 1
        with torch.no_grad():
            # t.max(0) will return the largest column value the probability vector (1-D Tensor).
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state[0].to(device),state[1].to(device)).max(0).indices.view(1, 1)

    episode_durations = []

    def plot_durations(show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    # Sensor 'mode' selection

    mode = mode # default = 'rgb' # Options: 'rgb', 'depth', 'segment'

    def state_gen(mode):
        """Input: Sensor mode
        Output: A valid state matching the sensor mode"""
        if mode == 'rgb':
            stacked = stack_img(
                torch.unsqueeze(torch.permute(torch.tensor(rgb_obs[0].astype('f')),(2, 0, 1)),0),
                torch.unsqueeze(torch.permute(torch.tensor(rgb_obs[1].astype('f')),(2, 0, 1)),0),
                torch.unsqueeze(torch.permute(torch.tensor(rgb_obs[2].astype('f')),(2, 0, 1)),0),
                torch.unsqueeze(torch.permute(torch.tensor(rgb_obs[3].astype('f')),(2, 0, 1)),0)
            )
            return preprocess_img_stack(stacked)
        elif mode == 'depth':
            return stack_img(
                torch.tensor(depth_obs[0].astype('f')),
                torch.tensor(depth_obs[1].astype('f')),
                torch.tensor(depth_obs[2].astype('f')),
                torch.tensor(depth_obs[3].astype('f'))
            )
        elif mode == 'segment':
            stacked = stack_img(
                torch.unsqueeze(torch.permute(torch.tensor(segment_obs[0].astype('f')),(2, 0, 1)),0),
                torch.unsqueeze(torch.permute(torch.tensor(segment_obs[1].astype('f')),(2, 0, 1)),0),
                torch.unsqueeze(torch.permute(torch.tensor(segment_obs[2].astype('f')),(2, 0, 1)),0),
                torch.unsqueeze(torch.permute(torch.tensor(segment_obs[3].astype('f')),(2, 0, 1)),0)
            )
            return preprocess_img_stack(stacked)
        else:
            raise ValueError('Pass a valid sensor mode')
        
    if torch.cuda.is_available():
        num_episodes = 50
    else:
        num_episodes = 50

    print("Initializing CARLA...")
    carla_env = carla_gym_v6.CarlaEnv(mode=mode)

    if SHOW_PREVIEW == True: # default = False
        carla_env.SHOW_CAM = True

    cum_rewards = []

    for i_episode in range(num_episodes):

        print("\nEpisode %d\n-------" % (i_episode + 1))
        
        # Initialize the environment and get it's state
        print("Starting new episode")

        start_time =  time.time()
        rewards = []
        policy_action_exploitation = 0
        random_action_exploration = 0

        rgb_obs, depth_obs, segment_obs, motion, step_reward, end_state, _ = carla_env.reset() 

        state = state_gen(mode=mode),torch.tensor([motion[0],motion[1]])
        
        for t in count():
            action = select_action(state)
            rgb_obs, depth_obs, segment_obs, motion, step_reward, end_state, _ = carla_env.step(action.item())
            rewards.append(step_reward)
            reward = torch.tensor([step_reward], device=device)
            done = end_state

            if done:
                # next_state = (None,None) # Experiment: Since the final state has the collision reward, it may have to be included in the states the DQN sees 
                
                end_time = time.time() - start_time
                # Getting sensor data after the action
                motion = carla_env.get_motion()
                rgb_obs = carla_env.cam_images[-4:]
                depth_obs = carla_env.depth_maps[-4:]
                segment_obs = carla_env.segmentation_maps[-4:]

                # Recording next_state
                next_state = state_gen(mode=mode),torch.tensor([motion[0],motion[1]])

            else:
                # Getting sensor data after the action
                motion = carla_env.get_motion()
                rgb_obs = carla_env.cam_images[-4:]
                depth_obs = carla_env.depth_maps[-4:]
                segment_obs = carla_env.segmentation_maps[-4:]

                # Recording next_state
                next_state = state_gen(mode=mode),torch.tensor([motion[0],motion[1]])

            # Move to the next state
            state = next_state

            if done:
                print("Episode ended")
                episode_reward = sum(rewards)
                cum_rewards.append(episode_reward)
                print("Episode Cumulative Reward: ", episode_reward)
                episode_durations.append(end_time)
                #print("Episode steps taken: ", steps_done)
                print("Episode duration: ", episode_durations[-1])
                #print("Exploration Steps - Random Action: ", random_action_exploration)
                #print("Exploitation Steps - Policy Action: ", policy_action_exploitation)
                #plot_durations()
                manual_episode_end(carla_env) # Clean the environment actors
                break

    print('Trajectory complete!')

    #plot_durations(show_result=True)
    #plt.ioff()
    #plt.show()

    # Saving the Agent

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Options for running the deep q-network training script for a CARLA self-driving agent')
    
    parser.add_argument('--SHOW_PREVIEW', default=True, action='store_true', help="If True it will open an OpenCV window displaying the signal from the RGB camera of the agent")
    parser.add_argument('--mode', default='rgb', action='store_true', help="Selects the sensor whose signal will be passed through the neural network: 'rgb', 'depth', 'segment'")

    arguments = parser.parse_args()

    print(arguments)

    main(
        arguments.SHOW_PREVIEW,
        arguments.mode
        )