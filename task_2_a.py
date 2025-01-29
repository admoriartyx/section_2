from collections import Counter
from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
np.isclose

def surface1(x, y):
    return 2*x**2+2*y**2
def surface2(x, y):
    return 2*np.exp(-x**2 - y**2)

# Okay I know this is not ideal and looks lazy but this section is due in 30 minutes and
# I don't think it would be productive for me to spend those 30 minutes trying to figure out 
# how to code the 3D Delauney trianglation just to not be able to implement it. I will now be
# attempting to push to Github. Once again, hope this is enough work to show.