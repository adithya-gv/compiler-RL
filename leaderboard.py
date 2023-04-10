#!/bin/env python3
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy
from compiler_gym.envs import LlvmEnv
from compiler_gym.wrappers import TimeLimit, CommandlineWithTerminalAction

import pickle

from ppo import PPO

checkpoint_path = "PPO_preTrained/llvm-ic-v0/PPO_llvm-ic-v0_0_20221206085337.pth"
gc_filepath = "gc_dict-16384.pkl"

with open(gc_filepath, "rb") as f:
  gc = pickle.load(f)


def policy(env: LlvmEnv) -> None:
  env = CommandlineWithTerminalAction(env)
  env = TimeLimit(env, 256)
  env.observation_space = "Programl"
  
  action_dim = env.action_space.n
  eps_clip = 0.1  # clip parameter for PPO
  gamma = 0.9  # discount factor
  lr_actor = 0.0001  # learning rate for actor network
  lr_critic = 0.0001  # learning rate for critic network
  K_epochs = 4  # update policy for K epochs in one PPO update

  ppo_agent = PPO(action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
  ppo_agent.load(checkpoint_path)
  

  done = False
  state = env.reset()
  while not done:
    state = gc.to_pyg(state)
    action = ppo_agent.select_action(state)
    ppo_agent.buffer.clear()

    state, _, done, _ = env.step(action)

if __name__ == "__main__":
    eval_llvm_instcount_policy(policy)
