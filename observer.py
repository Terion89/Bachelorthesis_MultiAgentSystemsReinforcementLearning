import copy
from logging import getLogger

import chainer
from chainer import cuda
import chainer.functions as F

from chainerrl import agent
from chainerrl.misc.batch_states import batch_states
from chainerrl.misc.copy_param import synchronize_parameters
from chainerrl.replay_buffer import batch_experiences
from chainerrl.replay_buffer import batch_recurrent_experiences
from chainerrl.replay_buffer import ReplayUpdater

""" 
Observer Agent 

standing above the arena on an ice block, looking down, just here for the human user to observe the episode top down
actions: -

"""


class OBSERVER:

    def __init__(self):
        pass
