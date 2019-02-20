from DataProcess import DataProcess
import os

data =  DataProcess()

try:
    data.cicids_process_data_multiclass()
except Exception as e:
    print(e)


