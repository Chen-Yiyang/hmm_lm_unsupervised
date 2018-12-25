import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from reader import *
from HMMGenerator import *
from HiddenMarkovModel import *
from forward_bakward import *


##data_path =  "simple-examples/data/"
##
##raw_data = reader.ptb_raw_data(data_path)
##train_data, valid_data, test_data, _ = raw_data
##
####for i in range(1,100):
####    print(train_data[i], end=' ')
##
##hidden_size = 200
##vocab_size = 10000
##
##data=train_data[1:10]
##
##embedding = tf.get_variable(
##    "embedding", [vocab_size, hidden_size], dtype=tf.float32)
##inputs = tf.nn.embedding_lookup(embedding, data)

# True values
True_pi = np.array([0.5, 0.5])
True_T = np.array([[0.85, 0.15],
                  [0.12, 0.88]])
True_E = np.array([[0.8, 0.0],
                   [0.1, 0.0],
                   [0.1, 1.0]])

# try Baum Welch
obs_seq, states = generate_HMM_observation(50, True_pi, True_T, True_E)
init_pi = np.array([0.5, 0.5])
init_T = np.array([[0.5, 0.5],
                  [0.5, 0.5]])
init_E = np.array([[0.3, 0.2],
                   [0.3, 0.5],
                   [0.4, 0.3]])

model =  HiddenMarkovModel(init_T, init_E, init_pi, epsilon=0.0001, maxStep=12)

trans0, transition, emission, c = model.run_Baum_Welch_EM(obs_seq, summary=True, monitor_state_1=True)

print("Transition Matrix: ")
print(transition)
print()
print("Emission Matrix: ")
print(emission)
print()
print("Reached Convergence: ")
print(c)

##Transition Matrix: 
##[[0.43532019 0.56467981]
## [0.21640773 0.78359227]]
##
##Emission Matrix: 
##[[0.71503812 0.04553823]
## [0.05107277 0.1299975 ]
## [0.2338891  0.82446426]]
