import os
import io
from numpy import float64, array,amax,amin

def get_current_working_directory():
    """
    get current directory
    """
    return os.getcwd()

def add_to_file(content, filename):
    """
    add content to filename in current directory
    """
    filepath = os.path.join(get_current_working_directory(), filename)

    try:
        if not os.path.exists(filepath):
            with open(filepath, 'w'):
                pass
        with open(filepath, "a") as f:
            f.write(content+'\n')
        return True
    except Exception:
        return False

def read_file(filename):
    """
        read a whole file as one with file name, not full path
    """
    filepath = os.path.join(get_current_working_directory(), 'data',filename)
    with open(filepath, 'r', encoding='utf-8') as content:
        return content.read()

def read_file_lines(filename):
    """
       read all lines of file with file name, not full path
    """
    filepath = os.path.join(get_current_working_directory(), 'data',filename)
    with open(filepath, 'r', encoding='utf-8') as content:
        return content.readlines()

def extract_features(a_line):
    """
    extract features based on comma (,), return an array
    """
    return [x.strip() for x in a_line.split(',')]

def numericalize_feature(feature,protocol_type,service,flag):
    protocol_type_count = len(protocol_type)
    service_count = len(service)
    flag_count = len(flag)
    second_index = int(protocol_type_count+1)
    third_index = int(protocol_type_count+service_count+1)
    forth_index = int(protocol_type_count+service_count+flag_count+1)

    # index 1 is protocol_type
    feature[1:1] = protocol_type[feature[1]]
    feature.pop(second_index)

    # index 2 + protocol_type_count is service
    feature[second_index:second_index] = service[feature[second_index]]
    feature.pop(third_index)
    # # index 3 + protocol_type_count + service_count is flag
    feature[third_index:third_index] = flag[feature[third_index]]
    feature.pop(forth_index)

    # make all values float64
    feature = [float64(x) for x in feature]

    return array(feature)

train_data = read_file_lines('KDDTrain+.txt')
test_data = read_file_lines('KDDTest+.txt')

# create arrays of arrays from lines
normalized_train_data_features = [extract_features(x) for x in train_data]
normalized_test_data_features = [extract_features(x) for x in test_data]

# train data: put index 0 to 40 in data, 41 and 42 into result (we don't need 41,42 for now)
normalized_train_data_results = [x[41:43] for x in normalized_train_data_features]
normalized_train_data_features = [x[0:41] for x in normalized_train_data_features]

# test data: put index 0 to 40 in data, 41 and 42 into result (we don't need 41,42 for now)
normalized_test_data_results = [x[41:43] for x in normalized_test_data_features]
normalized_test_data_features = [x[0:41] for x in normalized_test_data_features]

# stage 1 : numericalization --> index 1, 2 and 3 of dataset
# 1.1 extract all protocol_types, services and flags
protocol_type = dict()
service = dict()
flag = dict()
for entry in normalized_train_data_features:
    protocol_type[entry[1]] = ""
    service[entry[2]] = ""
    flag[entry[3]] = ""

keys= list(protocol_type.keys())
for i in range(0,len(keys)):
    protocol_type[keys[i]] = [int(d) for d in str(bin(i)[2:].zfill(len(protocol_type)))]

keys= list(service.keys())
for i in range(0,len(keys)):
    service[keys[i]] = [int(d) for d in str(bin(i)[2:].zfill(len(service)))]

keys= list(flag.keys())
for i in range(0,len(keys)):
    flag[keys[i]] = [int(d) for d in str(bin(i)[2:].zfill(len(flag)))]

# train data
normalized_train_data_features = [numericalize_feature(x,protocol_type,service,flag) for x in normalized_train_data_features]
normalized_train_data_features = array(normalized_train_data_features)

# test data
normalized_test_data_features = [numericalize_feature(x,protocol_type,service,flag) for x in normalized_test_data_features]
normalized_test_data_features = array(normalized_test_data_features)

# stage 2: normalization --> x = (x - MIN) / (MAX - MIN) --> based on columns

# train data
ymin_train = amin(normalized_train_data_features,axis=0)
ymax_train = amax(normalized_train_data_features,axis=0)

# test data
ymin_test = amin(normalized_test_data_features,axis=0)
ymax_test = amax(normalized_test_data_features,axis=0)



print('ymin[0],ymax[0]')