import csv
import logging
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
from mingpt.utils import sample
from collections import deque
import random
import torch
import pickle
import blosc
import argparse
from fixed_replay_buffer import FixedReplayBuffer

def create_dataset(num_steps):
    # -- load data from memory (make more efficient)
    obss = []
    actions = []
    done_idxs = []
    rtgs = []

    while len(actions) < num_steps:
        ran = random.randint(1, 6900)
        obs=np.load('../traj/obss_'+str(ran)+'_.npy',allow_pickle= True)
        action=np.load('../traj/actions_'+str(ran)+'_.npy',allow_pickle= True)
        done_idx=np.load('../traj/done_idxs_'+str(ran)+'_.npy',allow_pickle= True)
        rtg=np.load('../traj/rtgs_'+str(ran)+'_.npy',allow_pickle= True)
        rtg = rtg*10
        if len(actions) == 0:
            obss = obs
            actions = action
            done_idxs = done_idx
            rtgs = rtg
        else:
            obss = np.append(obss,obs,axis=0)
            actions = np.append(actions,action)
            done_idxs = np.append(done_idxs,done_idx)
            rtgs = np.append(rtgs,rtg)


    # -- create timestep dataset
    start_index = 0
    timesteps = np.zeros(len(actions)+1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1
    print('max timestep is %d' % max(timesteps))

    return obss, actions, done_idxs, rtgs, timesteps
