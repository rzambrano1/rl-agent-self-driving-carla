#!C:/Users/rzamb/anaconda3/envs/.carla/bin/python

#########
# Notes #
#########

# Parts of the coode below are adeapted from 
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

######################
# Python Boilerplate #
######################

import os
import sys

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

#######################
# Replay Memory Class #
#######################

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward')) 

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

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
        BATCH_SIZE, 
        GAMMA,
        EPS_START,
        EPS_END,
        EPS_DECAY,
        TAU,
        LR,
        SHOW_PREVIEW,
        mode,
        POLICY_FILE_NAME, 
        TARGET_FILE_NAME
):

    ############
    # Training #
    ############

    # Policy and Target Network instanciation

    BATCH_SIZE = BATCH_SIZE # dafault = 128   # The number of transitions sampled from the replay buffer
    GAMMA = GAMMA           # default = 0.99  # The discount factor as mentioned in the previous section
    EPS_START = EPS_START   # default = 0.9   # Starting value of epsilon
    EPS_END = EPS_END       # default = 0.05  # Final value of epsilon
    EPS_DECAY = EPS_DECAY   # default = 1000  # Controls the rate of exponential decay of epsilon, higher means a slower decay
    TAU = TAU               # default = 0.005 # The update rate of the target network
    LR = LR                 # default = 1e-4  # The learning rate of the ``AdamW`` optimizer

    # Get number of actions for the CARLA environment. Hardcoded item.
    n_actions = 8 # Equals len(carla_env.action_spac) # Originally 13, but all brake options were removed

    policy_net = DQN_TempLimit(n_actions).to(device)
    target_net = DQN_TempLimit(n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    steps_done = 0

    def select_action(state):
        nonlocal steps_done
        nonlocal policy_action_exploitation
        nonlocal random_action_exploration
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            policy_action_exploitation += 1
            with torch.no_grad():
                # t.max(0) will return the largest column value the probability vector (1-D Tensor).
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state[0].to(device),state[1].to(device)).max(0).indices.view(1, 1)
        else:
            random_action_exploration += 1
            return torch.tensor([[carla_env.action_space_sample()]], device=device, dtype=torch.long)

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

    # Training Loop

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        
        transitions = memory.sample(BATCH_SIZE) # transitions[0] is an instance of a class with a state, an action, a next state, and a reward 
        
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        # batch[0] are states, batch[1] are actions, ... , batch[3] are rewards
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s != (None,None), 
                                            batch.next_state)), device=device, dtype=torch.bool)
        
        non_final_next_states_0, non_final_next_states_1 = zip(*batch.next_state)
        non_final_next_states_00 = [torch.unsqueeze(s0,0) for s0 in non_final_next_states_0 if s0 is not None]
        non_final_next_states_11 = [torch.unsqueeze(s1,0) for s1 in non_final_next_states_1 if s1 is not None]
        non_final_next_states_img = torch.cat(non_final_next_states_00,dim=0)
        non_final_next_states_vel = torch.cat(non_final_next_states_11)  
        
        state_batch_0,state_batch_1 = zip(* batch.state)
        state_batch_00 = [torch.unsqueeze(s,0) for s in state_batch_0]
        state_batch_11 = [torch.unsqueeze(v,0) for v in state_batch_1] 
        state_batch_img = torch.cat(state_batch_00,dim=0)
        state_batch_vel = torch.cat(state_batch_11)
        
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch_img.to(device),state_batch_vel.to(device)).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            target_output = target_net(non_final_next_states_img.to(device),non_final_next_states_vel.to(device)).max(1).values
            next_state_values[non_final_mask] = target_output
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

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
        num_episodes = 251 #1500
    else:
        num_episodes = 251 #500

    print("Initializing CARLA...")
    carla_env = carla_gym_v6.CarlaEnv(mode=mode)

    if SHOW_PREVIEW == True: # default = False
        carla_env.SHOW_CAM = True

    cum_rewards = []

    for i_episode in range(num_episodes):

        print("\nEpoch %d\n-------" % (i_episode + 1))
        
        # Initialize the environment and get it's state
        print("Starting new episode")

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
                #next_state = (None,None) # Experiment: Since the final state has the collision reward, it may have to be included in the states the DQN sees

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

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                #print("Episode ended")
                episode_reward = sum(rewards)
                cum_rewards.append(episode_reward)
                print("Episode Cumulative Reward: ", episode_reward)
                episode_durations.append(t + 1)
                print("Episode steps taken: ", steps_done)
                print("Episode duration: ", episode_durations[-1])
                print("Exploration Steps - Random Action: ", random_action_exploration)
                print("Exploitation Steps - Policy Action: ", policy_action_exploitation)
                #plot_durations()
                manual_episode_end(carla_env) # Clean the environment actors
                break

        if int(i_episode + 1) % 50 == 0:
            print("Saving current policy and target network...")
            WORKING_DIRECTORY_PATH = "C:/Users/rzamb/Documents/UMD/642_Robotics/finalProject/"
            POLICY_FILE_NAME = "policy"
            POLICY_FILE_NAME = POLICY_FILE_NAME + "_" + mode + "_ep" + str(i_episode + 1) + ".pth"
            TARGET_FILE_NAME = "target"
            TARGET_FILE_NAME = TARGET_FILE_NAME + "_" + mode + "_ep" + str(i_episode + 1) + ".pth"
            POLICY_PATH = os.path.join(WORKING_DIRECTORY_PATH, POLICY_FILE_NAME)
            TARGET_PATH = os.path.join(WORKING_DIRECTORY_PATH, TARGET_FILE_NAME)

            torch.save(policy_net.state_dict(), POLICY_PATH)
            torch.save(target_net.state_dict(), TARGET_PATH)


    print('Training complete!')

    print("Initial Cumulative Rewards: ", cum_rewards[0:10])
    print("- Cumulative Rewards Towards End of Training: ", cum_rewards[-10:])

    #plot_durations(show_result=True)
    #plt.ioff()
    #plt.show()

    # Saving the Agent

    print("Saving final policy and target network...")
    WORKING_DIRECTORY_PATH = "C:/Users/rzamb/Documents/UMD/642_Robotics/finalProject/"
    POLICY_FILE_NAME = POLICY_FILE_NAME # default = "policy"
    POLICY_FILE_NAME = POLICY_FILE_NAME + "_" + mode + "_final_" + ".pth"
    TARGET_FILE_NAME = TARGET_FILE_NAME # default = "target"
    TARGET_FILE_NAME = TARGET_FILE_NAME + "_" + mode + "_final_" + ".pth"
    POLICY_PATH = os.path.join(WORKING_DIRECTORY_PATH, POLICY_FILE_NAME)
    TARGET_PATH = os.path.join(WORKING_DIRECTORY_PATH, TARGET_FILE_NAME)

    torch.save(policy_net.state_dict(), POLICY_PATH)
    torch.save(target_net.state_dict(), TARGET_PATH)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Options for running the deep q-network training script for a CARLA self-driving agent')
    
    parser.add_argument('--BATCH_SIZE', default=128, action='store_true', help="Batch size to process from the replay memory")
    parser.add_argument('--GAMMA', default=0.99, action='store_true', help="Discount factor: a constant between 0 and 1. Ensures discounted cummulative reward sum converges")
    parser.add_argument('--EPS_START', default=0.9, action='store', help="Start value of epsilon. An epsilon-greedy policy uses epsilon to manage exploit-exploration problem")
    parser.add_argument('--EPS_END', default=0.05,action='store_true', help="End value of epsilon")
    parser.add_argument('--EPS_DECAY', default=5000, action='store_true', help="Controls the rate of exponential decay of epsilon, higher means a slower decay") # Tested 1k and 5k
    parser.add_argument('--TAU', default=0.001, action='store_true', help="The update rate of the target network") # Originally 0.005. Tested 0.01 - 0.005 and 0.001.
    parser.add_argument('--LR', default=1e-4, action='store_true', help="The learning rate of the ``AdamW`` optimizer")
    parser.add_argument('--SHOW_PREVIEW', default=False, action='store_true', help="If True it will open an OpenCV window displaying the signal from the RGB camera of the agent")
    parser.add_argument('--mode', default='rgb', action='store_true', help="Selects the sensor whose signal will be passed through the neural network: 'rgb', 'depth', 'segment'")
    parser.add_argument('--POLICY_FILE_NAME', default='policy', action='store_true', help="The name for the file with the network's weights. Ignore the extension .pth")
    parser.add_argument('--TARGET_FILE_NAME', default='target', action='store_true', help="The name for the file with the network's weights. Ignore the extension .pth")

    arguments = parser.parse_args()

    print(arguments)

    main(
        arguments.BATCH_SIZE, 
        arguments.GAMMA,
        arguments.EPS_START,
        arguments.EPS_END,
        arguments.EPS_DECAY,
        arguments.TAU,
        arguments.LR,
        arguments.SHOW_PREVIEW,
        arguments.mode,
        arguments.POLICY_FILE_NAME, 
        arguments.TARGET_FILE_NAME
        )
