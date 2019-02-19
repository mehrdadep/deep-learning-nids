from DataProcess import DataProcess
import os
import numpy as np

# data =  DataProcess()

# try:
#     data.cicids_process_data_binary()
# except:
#     print('bin')
# try:
#     data.cicids_process_data_multiclass()
# except:
#     print('mult')

# os.system("shutdown /s /t 1")

a = [[0.58852], [0.05455] ,[1.1555], [1.6565] ,[2.655], [2.25] ,[3.3] ,[3.9] ,[4.6]]
a = np.array(a)
b = [np.round(x) for x in a]
print(a)

print(np.array(b))
