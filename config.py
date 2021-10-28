"""!
@file config.py
Sets the parameters for configuration
"""
import socket
from typing import Any, Optional

import numpy as np
import tensorflow as tf

##### Shapes and Capacities #####
if socket.gethostname() == "ckm4cad":
    ONSERVER = True
    op_parallelism_threads = 6
    batch_size_train = 16
    batch_capacity_train = 4000
    train_reader_instances = 2

else:
    ONSERVER = False
    op_parallelism_threads = 3
    batch_size_train = 100
    batch_capacity_train = 10000
    train_reader_instances = 1

batch_size_valid = batch_size_train
vald_reader_instances = 1
file_name_capacity = 140

data_train_split = 0.8
