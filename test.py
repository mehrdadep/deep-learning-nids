from DataProcess import DataProcess
import os

data =  DataProcess()

try:
    data.cicids_process_data_binary()
except:
    print('bin')
try:
    data.cicids_process_data_multiclass()
except:
    print('mult')

os.system("shutdown /s /t 1")

