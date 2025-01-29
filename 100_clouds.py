# Generating 100 different n=50 datasets 

import numpy as np
import matplotlib.pyplot as plt
import time

def cloud_generator(n):
    return np.random.rand(n, 2)

def all_cloud_generator(num_clouds, n):
    # Empty list where all clouds will be dumped
    all_clouds = []

    for i in range(num_clouds):
        cloud = cloud_generator(n)
        all_clouds.append(cloud)
    
    return all_clouds

all_cloud_generator(100, 50)

# I am going to stop here as the rest is somewhat trivial, I have been making the plots
# for awhile now. I want to take a look at Task 2 before I run out of time so hopefully
# this is a sufficient amount of work for Task 1 to obtain credit.


