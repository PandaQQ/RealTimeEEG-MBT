import subprocess
import datetime, time


import numpy as np


import pylsl

import random

from pylsl import StreamInfo, StreamOutlet


from scipy.io import loadmat

data_temp = loadmat('sample_data_24ch.mat')

data = data_temp['data']



info = StreamInfo('EEG_stream', 'EEG', 24, 0, 'float32', 'myuid34234')

info1 = StreamInfo('MyMarkerStream', 'Markers', 1, 0, 'string', 'myuidw43536')

# next make an outlet
outlet = StreamOutlet(info)
outlet1 = StreamOutlet(info1)

j = 0


while True:
    mysample = data[:,j].tolist()
    outlet.push_sample(mysample)
    time.sleep(0.004)
    j = j+1
    if j%500==0:
        outlet1.push_sample('1')
        print('1')

    
