import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from reader import *
from HMMGenerator import *
from HiddenMarkovModel import *
from forward_bakward import *


data_path =  "simple-examples/data/"

raw_data = ptb_raw_data(data_path)
train_data, valid_data, test_data, _ = raw_data


train_data = train_data[0:100000]

obs_seq, states = generate_HMM_observation(50, True_pi, True_T, True_E)

# hyper-params
state_size = 40
emission_size = 10000

# init
init_pi = np.random.rand(state_size)
init_pi = init_pi/init_pi.sum()

init_T = np.random.rand(state_size, state_size)
init_T = init_T/init_T.sum(axis=1)

init_E = np.random.rand(emission_size, state_size)
init_E = init_E/init_E.sum(axis=0)

model =  HiddenMarkovModel(init_T, init_E, init_pi, epsilon=0.0003, maxStep=12)

trans0, transition, emission, c = model.run_Baum_Welch_EM(obs_seq, summary=True, monitor_state_1=True)
