import torch 
import numpy as np 
import torch.nn.functional as functional
import argparse
import random 

file_name = './../run_frame_reconstruction.sh'
with open(file_name, 'w') as f:
    base_str = 'python frame_reconstruction.py'
    for i in range(100):
        for j in range(100):
            ld0 = random.uniform(0.0, 5.0)
            ld2 = random.uniform(0.0, 5.0)
            to_write = '{} -ld0 {} -ld2 {}\n'.format(base_str, ld0, ld2)
            f.write(to_write)
            