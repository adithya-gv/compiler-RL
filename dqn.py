# Credit: Compiler Gym for the training environment
# Link to Compiler Gym: https://github.com/facebookresearch/CompilerGym

# Portions of this code are sourced from: https://compilergym.com/getting_started.html
# (specifically parts of the "Training a DQN agent" section)

import gym
from compiler_gym.wrappers import TimeLimit
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy
import math
import graph as graph
import random
import numpy as np

from itertools import count

import torch
from collections import namedtuple, deque

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
import torch_geometric.utils as utils

def make_env(timesteps=5):
  env = gym.make("llvm-ic-v0", observation_space='Programl', reward_space="IrInstructionCountOz")

  # This step caps the total number of steps per episode (we're in a continuous space)
  env = TimeLimit(env, max_episode_steps=timesteps)
  return env

env = make_env()

ds = env.datasets["cbench-v1"]

training_benchmarks = [
        "adpcm",
        "blowfish",
        "crc32",
        # "ghostscript",
        "ispell",
        "jpeg-d",
        "patricia",
        "rijndael",
        "stringsearch2",
        "susan",
        "tiff2rgba",
        "tiffmedian",
        "bitcount",
        "bzip2",
        "dijkstra",
        "gsm",
        "jpeg-c",
        "lame",
        "qsort",
        "sha",
        "stringsearch",
        "tiff2bw",
        "tiffdither"
    ]


# turn on interactive mode


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sets up transition
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

converter = graph.GraphConverter(8000)

# sets up replay memory for experience replay
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    # provides an action for each item in the batch
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

output_size = env.action_space.n
input_size = 1

class Network(nn.Module):
  def __init__(self, converter):
    super().__init__()
    self.conv1 = GCNConv(input_size, 2048)
    self.dropout1 = nn.Dropout(0.5)
    self.conv2 = GCNConv(2048, 1024)
    self.dropout2 = nn.Dropout(0.5)
    self.norm1 = BatchNorm(1024)
    self.linear1 = nn.Linear(1024, 256)
    self.relu3 = nn.LeakyReLU()
    self.linear2 = nn.Linear(256, output_size)

    self.converter = converter

  def forward(self, data):

    data = self.converter.to_pyg(data)

    x, edge_index = data.x, data.edge_index
    x = x.float()
    x = self.conv1(x, edge_index)
    x = self.dropout1(x)
    x = self.conv2(x, edge_index)
    x = self.dropout2(x)
    x = self.norm1(x)
    x = self.linear1(x)
    x = self.relu3(x)
    x = self.linear2(x)

    out = F.log_softmax(x, dim=1)
    return out

# parameters
BATCH_SIZE = 1
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# inital variables
rb = random.choice(training_benchmarks)
init_state = env.reset(benchmark="benchmark://npb-v0/46")
n_actions = output_size

# policy net computes policy 
policy_net = Network(converter).to(device)

# target net computes state optimality
target_net = Network(converter).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.01) # RMS prop optimzer
memory = ReplayMemory(10000) # 10000 frame size replay memory

steps_done = 0

# selecting action with epsilon greedy policy
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():	
            act = policy_net(state)
            rewards = torch.max(act, dim=1)[0]
            positions = torch.max(act, dim=1)[1]
            max_reward = torch.argmax(rewards)
            ret_val = torch.tensor([positions[max_reward].item()])
            return ret_val.view(1, 1)
    else:
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        return action



episode_durations = []

# single optimization step
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)

    non_final_next_states = batch.next_state[0]

    state_batch = batch.state[0]
    action_batch = batch.action[0]
    reward_batch = batch.reward[0]


    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if (non_final_next_states is not None):
        res = target_net(non_final_next_states)
        rewards = torch.max(res, dim=1)[0]
        max_reward = torch.argmax(rewards)
        next_state_values[non_final_mask] = rewards[max_reward].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))


    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss


# actual training loop
print("Starting Training...")
num_episodes = 65
for i_episode in range(num_episodes):
    total_loss = 0.0
    # Initialize the environment and state
    env.reset()
    state = env.observation["Programl"]
    for t in count():
        # Select and perform an action
        action = select_action(state)
        # print(action)
        next_state, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        loss = optimize_model()
        if loss is not None:
            total_loss += loss.item()
        if done:
            episode_durations.append(t + 1)
            break

        if t % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    print(i_episode)

def my_policy(env):
    env.reset()
    total_reward = 0.0
    for i in range(50):
        state = env.observation["Programl"]
        act = policy_net(state)
        rewards = torch.max(act, dim=1)[0]
        positions = torch.max(act, dim=1)[1]
        max_reward = torch.argmax(rewards)
        action = torch.tensor([positions[max_reward].item()])
        action = action.view(1, 1)
        _, reward, done, _ = env.step(action.item())
        total_reward += reward
        if done:
            return total_reward / i


def random_policy(env) -> None:
    env.reset()
    total_reward = 0.0
    for i in range(50):
       _, reward, done, _ = env.step(env.action_space.sample())
       total_reward += reward
       if done:
        return total_reward / i

for i in range(10):
    q_wins = 0
    r_wins = 0
    ties = 0
    for i in range(150):
        r1 = my_policy(env)
        print(r1)

        env.reset()
        r2 = random_policy(env)
        # print(r2)
        
        if r1 > r2:
            q_wins += 1
        else:
            r_wins += 1

    print("Agent's Record vs. Random: " + str(q_wins) + "-" + str(r_wins))

env.close()