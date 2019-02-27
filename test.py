from DataProcess import DataProcess
import os
import numpy as np

data =  DataProcess()

try:
    data.cicids_process_data_binary()
except Exception as e:
    print(e)


try:
    data.cicids_process_data_multiclass()
except Exception as e:
    print(e)


import winsound
winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
