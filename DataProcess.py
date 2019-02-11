import os
import io
from numpy import float64, array, amax, amin, savetxt, genfromtxt


class DataProcess:
    _DATA_FOLDER = "data"
    @classmethod
    def get_current_working_directory(self):
        """
        get current directory
        """
        return os.getcwd()

    @classmethod
    def read_file_lines(self, dataset, filename):
        """
        read all lines of file with file name, not full path
        """
        filepath = os.path.join(
            self.get_current_working_directory(), 'data',dataset, filename)
        with open(filepath, 'r', encoding='utf-8') as content:
            return content.readlines()

    @classmethod
    def extract_features(self, a_line):
        """
        extract features based on comma (,), return an array
        """
        return [x.strip() for x in a_line.split(',')]

    @classmethod
    def numericalize_feature(self, feature, protocol_type, service, flag):
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

    @classmethod
    def numericalize_result(self, reslut, attack, attack_dict):
        second_index = int(1)
        # index 0 is attack
        reslut[0:0] = attack[attack_dict[reslut[0]]]
        reslut.pop(second_index)
        # make all values float64
        reslut = [float64(x) for x in reslut]

        return array(reslut)

    @classmethod
    def normalize_value(self, value, min, max):
        value = float64(value)
        min = float64(min)
        max = float64(max)

        if min == float64(0) and max == float64(0):
            return float64(0)
        result = float64((value - min) / (max - min))
        return result

    @classmethod
    def nsl_process_data_multiclass(self):
        """
        read from data folder and return a list
        [train_data, train_results, test_data, test_results]
        """
        train_data = self.read_file_lines('nsl', 'KDDTrain+.txt')
        test_data = self.read_file_lines('nsl', 'KDDTest+.txt')
        test_21_data = self.read_file_lines('nsl', 'KDDTest-21.txt')

        # create arrays of arrays from lines
        raw_train_data_features = [
            self.extract_features(x) for x in train_data]
        raw_test_data_features = [self.extract_features(x) for x in test_data]
        raw_test_21_data_features = [
            self.extract_features(x) for x in test_21_data]

        # train data: put index 0 to 40 in data, 41 and 42 into result (we don't need 41,42 for now)
        raw_train_data_results = [x[41:42] for x in raw_train_data_features]
        raw_train_data_features = [x[0:41] for x in raw_train_data_features]

        # test data: put index 0 to 40 in data, 41 and 42 into result (we don't need 41,42 for now)
        raw_test_data_results = [x[41:42] for x in raw_test_data_features]
        raw_test_data_features = [x[0:41] for x in raw_test_data_features]

        raw_test_21_data_results = [x[41:42]
                                    for x in raw_test_21_data_features]
        raw_test_21_data_features = [x[0:41]
                                     for x in raw_test_21_data_features]

        # stage 1 : numericalization --> index 1, 2 and 3 of dataset
        # 1.1 extract all protocol_types, services and flags
        protocol_type = dict()
        service = dict()
        flag = dict()
        attack = dict()
        attack_dict = {
            'normal': 'normal',
            'back': 'DoS',
            'land': 'DoS',
            'neptune': 'DoS',
            'pod': 'DoS',
            'smurf': 'DoS',
            'teardrop': 'DoS',
            'mailbomb': 'DoS',
            'apache2': 'DoS',
            'processtable': 'DoS',
            'udpstorm': 'DoS',
            'ipsweep': 'Probe',
            'nmap': 'Probe',
            'portsweep': 'Probe',
            'satan': 'Probe',
            'mscan': 'Probe',
            'saint': 'Probe',
            'ftp_write': 'R2L',
            'guess_passwd': 'R2L',
            'imap': 'R2L',
            'multihop': 'R2L',
            'phf': 'R2L',
            'spy': 'R2L',
            'warezclient': 'R2L',
            'warezmaster': 'R2L',
            'sendmail': 'R2L',
            'named': 'R2L',
            'snmpgetattack': 'R2L',
            'snmpguess': 'R2L',
            'xlock': 'R2L',
            'xsnoop': 'R2L',
            'worm': 'R2L',
            'buffer_overflow': 'U2R',
            'loadmodule': 'U2R',
            'perl': 'U2R',
            'rootkit': 'U2R',
            'httptunnel': 'U2R',
            'ps': 'U2R',
            'sqlattack': 'U2R',
            'xterm': 'U2R'
        }
        for entry in raw_train_data_features:
            protocol_type[entry[1]] = ""
            service[entry[2]] = ""
            flag[entry[3]] = ""

        for entry in raw_test_data_features:
            protocol_type[entry[1]] = ""
            service[entry[2]] = ""
            flag[entry[3]] = ""

        for entry in raw_train_data_results:
            attack[attack_dict[entry[0]]] = ""

        for entry in raw_test_data_results:
            attack[attack_dict[entry[0]]] = ""

        keys = list(protocol_type.keys())
        for i in range(0, len(keys)):
            protocol_type[keys[i]] = [int(d) for d in str(
                bin(i)[2:].zfill(len(protocol_type)))]

        keys = list(service.keys())
        for i in range(0, len(keys)):
            service[keys[i]] = [int(d)
                                for d in str(bin(i)[2:].zfill(len(service)))]

        keys = list(flag.keys())
        for i in range(0, len(keys)):
            flag[keys[i]] = [int(d) for d in str(bin(i)[2:].zfill(len(flag)))]

        keys = list(attack.keys())
        for i in range(0, len(keys)):
            attack[keys[i]] = [int(i)]

        # train data
        numericalized_train_data_features = [self.numericalize_feature(
            x, protocol_type, service, flag) for x in raw_train_data_features]
        normalized_train_data_features = array(
            numericalized_train_data_features)

        numericalized_train_data_results = [self.numericalize_result(
            x, attack, attack_dict) for x in raw_train_data_results]
        normalized_train_data_results = array(numericalized_train_data_results)

        # test data
        numericalized_test_data_features = [self.numericalize_feature(
            x, protocol_type, service, flag) for x in raw_test_data_features]
        normalized_test_data_features = array(numericalized_test_data_features)

        numericalized_test_data_results = [self.numericalize_result(
            x, attack, attack_dict) for x in raw_test_data_results]
        normalized_test_data_results = array(numericalized_test_data_results)

        # test 21 data
        numericalized_test_21_data_features = [self.numericalize_feature(
            x, protocol_type, service, flag) for x in raw_test_21_data_features]
        normalized_test_21_data_features = array(
            numericalized_test_21_data_features)

        numericalized_test_21_data_results = [self.numericalize_result(
            x, attack, attack_dict) for x in raw_test_21_data_results]
        normalized_test_21_data_results = array(
            numericalized_test_21_data_results)

        # stage 2: normalization --> x = (x - MIN) / (MAX - MIN) --> based on columns

        # train data
        ymin_train = amin(numericalized_train_data_features, axis=0)
        ymax_train = amax(numericalized_train_data_features, axis=0)

        # test data
        ymin_test = amin(numericalized_test_data_features, axis=0)
        ymax_test = amax(numericalized_test_data_features, axis=0)

        # test 21 data
        ymin_test_21 = amin(numericalized_test_21_data_features, axis=0)
        ymax_test_21 = amax(numericalized_test_21_data_features, axis=0)

        # normalize train
        for x in range(0, normalized_train_data_features.shape[0]):
            for y in range(0, normalized_train_data_features.shape[1]):
                normalized_train_data_features[x][y] = self.normalize_value(
                    normalized_train_data_features[x][y], ymin_train[y], ymax_train[y])

        # normalize test
        for x in range(0, normalized_test_data_features.shape[0]):
            for y in range(0, normalized_test_data_features.shape[1]):
                normalized_test_data_features[x][y] = self.normalize_value(
                    normalized_test_data_features[x][y], ymin_test[y], ymax_test[y])

        # normalize test 21
        for x in range(0, normalized_test_21_data_features.shape[0]):
            for y in range(0, normalized_test_21_data_features.shape[1]):
                normalized_test_21_data_features[x][y] = self.normalize_value(
                    normalized_test_21_data_features[x][y], ymin_test_21[y], ymax_test_21[y])

        mul_nsl = os.path.join(
            self.get_current_working_directory(), 'data', 'mul-nsl')
        if not os.path.exists(mul_nsl):
           os.makedirs(mul_nsl)
        filepath = os.path.join(
            self.get_current_working_directory(), 'data', 'mul-nsl', "normalized_train_data_features.csv")
        savetxt(filepath, normalized_train_data_features, delimiter=",")
        filepath = os.path.join(
            self.get_current_working_directory(), 'data', 'mul-nsl', "normalized_train_data_results.csv")
        savetxt(filepath, normalized_train_data_results, delimiter=",")
        filepath = os.path.join(
            self.get_current_working_directory(), 'data', 'mul-nsl', "normalized_test_data_features.csv")
        savetxt(filepath, normalized_test_data_features, delimiter=",")
        filepath = os.path.join(
            self.get_current_working_directory(), 'data', 'mul-nsl', "normalized_test_data_results.csv")
        savetxt(filepath, normalized_test_data_results, delimiter=",")
        filepath = os.path.join(
            self.get_current_working_directory(), 'data', 'mul-nsl', "normalized_test_21_data_features.csv")
        savetxt(filepath, normalized_test_21_data_features, delimiter=",")
        filepath = os.path.join(
            self.get_current_working_directory(), 'data', 'mul-nsl', "normalized_test_21_data_results.csv")
        savetxt(filepath, normalized_test_21_data_results, delimiter=",")

        return True

    @classmethod
    def nsl_process_data_binary(self):
        """
        read from data folder and return a list
        [train_data, train_results, test_data, test_results]
        """
        train_data = self.read_file_lines('nsl', 'KDDTrain+.txt')
        test_data = self.read_file_lines('nsl', 'KDDTest+.txt')
        test_21_data = self.read_file_lines('nsl', 'KDDTest-21.txt')

        # create arrays of arrays from lines
        raw_train_data_features = [
            self.extract_features(x) for x in train_data]
        raw_test_data_features = [self.extract_features(x) for x in test_data]
        raw_test_21_data_features = [
            self.extract_features(x) for x in test_21_data]

        # train data: put index 0 to 40 in data, 41 and 42 into result (we don't need 41,42 for now)
        raw_train_data_results = [x[41:42] for x in raw_train_data_features]
        raw_train_data_features = [x[0:41] for x in raw_train_data_features]

        # test data: put index 0 to 40 in data, 41 and 42 into result (we don't need 41,42 for now)
        raw_test_data_results = [x[41:42] for x in raw_test_data_features]
        raw_test_data_features = [x[0:41] for x in raw_test_data_features]

        raw_test_21_data_results = [x[41:42]
                                    for x in raw_test_21_data_features]
        raw_test_21_data_features = [x[0:41]
                                     for x in raw_test_21_data_features]

        # stage 1 : numericalization --> index 1, 2 and 3 of dataset
        # 1.1 extract all protocol_types, services and flags
        protocol_type = dict()
        service = dict()
        flag = dict()
        attack = dict()
        attack_dict = {
            'normal': 'normal',
            'back': 'abnormal',
            'land': 'abnormal',
            'neptune': 'abnormal',
            'pod': 'abnormal',
            'smurf': 'abnormal',
            'teardrop': 'abnormal',
            'mailbomb': 'abnormal',
            'apache2': 'abnormal',
            'processtable': 'abnormal',
            'udpstorm': 'abnormal',
            'ipsweep': 'abnormal',
            'nmap': 'abnormal',
            'portsweep': 'abnormal',
            'satan': 'abnormal',
            'mscan': 'abnormal',
            'saint': 'abnormal',
            'ftp_write': 'abnormal',
            'guess_passwd': 'abnormal',
            'imap': 'abnormal',
            'multihop': 'abnormal',
            'phf': 'abnormal',
            'spy': 'abnormal',
            'warezclient': 'abnormal',
            'warezmaster': 'abnormal',
            'sendmail': 'abnormal',
            'named': 'abnormal',
            'snmpgetattack': 'abnormal',
            'snmpguess': 'abnormal',
            'xlock': 'abnormal',
            'xsnoop': 'abnormal',
            'worm': 'abnormal',
            'buffer_overflow': 'abnormal',
            'loadmodule': 'abnormal',
            'perl': 'abnormal',
            'rootkit': 'abnormal',
            'httptunnel': 'abnormal',
            'ps': 'abnormal',
            'sqlattack': 'abnormal',
            'xterm': 'abnormal'
        }
        for entry in raw_train_data_features:
            protocol_type[entry[1]] = ""
            service[entry[2]] = ""
            flag[entry[3]] = ""

        for entry in raw_test_data_features:
            protocol_type[entry[1]] = ""
            service[entry[2]] = ""
            flag[entry[3]] = ""

        for entry in raw_train_data_results:
            attack[attack_dict[entry[0]]] = ""

        for entry in raw_test_data_results:
            attack[attack_dict[entry[0]]] = ""

        keys = list(protocol_type.keys())
        for i in range(0, len(keys)):
            protocol_type[keys[i]] = [int(d) for d in str(
                bin(i)[2:].zfill(len(protocol_type)))]

        keys = list(service.keys())
        for i in range(0, len(keys)):
            service[keys[i]] = [int(d)
                                for d in str(bin(i)[2:].zfill(len(service)))]

        keys = list(flag.keys())
        for i in range(0, len(keys)):
            flag[keys[i]] = [int(d) for d in str(bin(i)[2:].zfill(len(flag)))]

        keys = list(attack.keys())
        for i in range(0, len(keys)):
            attack[keys[i]] = [int(i)]

        # train data
        numericalized_train_data_features = [self.numericalize_feature(
            x, protocol_type, service, flag) for x in raw_train_data_features]
        normalized_train_data_features = array(
            numericalized_train_data_features)

        numericalized_train_data_results = [self.numericalize_result(
            x, attack, attack_dict) for x in raw_train_data_results]
        normalized_train_data_results = array(numericalized_train_data_results)

        # test data
        numericalized_test_data_features = [self.numericalize_feature(
            x, protocol_type, service, flag) for x in raw_test_data_features]
        normalized_test_data_features = array(numericalized_test_data_features)

        numericalized_test_data_results = [self.numericalize_result(
            x, attack, attack_dict) for x in raw_test_data_results]
        normalized_test_data_results = array(numericalized_test_data_results)

        # test 21 data
        numericalized_test_21_data_features = [self.numericalize_feature(
            x, protocol_type, service, flag) for x in raw_test_21_data_features]
        normalized_test_21_data_features = array(
            numericalized_test_21_data_features)

        numericalized_test_21_data_results = [self.numericalize_result(
            x, attack, attack_dict) for x in raw_test_21_data_results]
        normalized_test_21_data_results = array(
            numericalized_test_21_data_results)

        # stage 2: normalization --> x = (x - MIN) / (MAX - MIN) --> based on columns

        # train data
        ymin_train = amin(numericalized_train_data_features, axis=0)
        ymax_train = amax(numericalized_train_data_features, axis=0)

        # test data
        ymin_test = amin(numericalized_test_data_features, axis=0)
        ymax_test = amax(numericalized_test_data_features, axis=0)

        # test 21 data
        ymin_test_21 = amin(numericalized_test_21_data_features, axis=0)
        ymax_test_21 = amax(numericalized_test_21_data_features, axis=0)

        # normalize train
        for x in range(0, normalized_train_data_features.shape[0]):
            for y in range(0, normalized_train_data_features.shape[1]):
                normalized_train_data_features[x][y] = self.normalize_value(
                    normalized_train_data_features[x][y], ymin_train[y], ymax_train[y])

        # normalize test
        for x in range(0, normalized_test_data_features.shape[0]):
            for y in range(0, normalized_test_data_features.shape[1]):
                normalized_test_data_features[x][y] = self.normalize_value(
                    normalized_test_data_features[x][y], ymin_test[y], ymax_test[y])

        # normalize test 21
        for x in range(0, normalized_test_21_data_features.shape[0]):
            for y in range(0, normalized_test_21_data_features.shape[1]):
                normalized_test_21_data_features[x][y] = self.normalize_value(
                    normalized_test_21_data_features[x][y], ymin_test_21[y], ymax_test_21[y])
        bin_nsl = os.path.join(
            self.get_current_working_directory(), 'data', 'bin-nsl')
        if not os.path.exists(bin_nsl):
           os.makedirs(bin_nsl)
        filepath = os.path.join(
            self.get_current_working_directory(), 'data', 'bin-nsl', "normalized_train_data_features.csv")
        savetxt(filepath, normalized_train_data_features, delimiter=",")
        filepath = os.path.join(
            self.get_current_working_directory(), 'data', 'bin-nsl', "normalized_train_data_results.csv")
        savetxt(filepath, normalized_train_data_results, delimiter=",")
        filepath = os.path.join(
            self.get_current_working_directory(), 'data', 'bin-nsl', "normalized_test_data_features.csv")
        savetxt(filepath, normalized_test_data_features, delimiter=",")
        filepath = os.path.join(
            self.get_current_working_directory(), 'data', 'bin-nsl', "normalized_test_data_results.csv")
        savetxt(filepath, normalized_test_data_results, delimiter=",")
        filepath = os.path.join(
            self.get_current_working_directory(), 'data', 'bin-nsl', "normalized_test_21_data_features.csv")
        savetxt(filepath, normalized_test_21_data_features, delimiter=",")
        filepath = os.path.join(
            self.get_current_working_directory(), 'data', 'bin-nsl', "normalized_test_21_data_results.csv")
        savetxt(filepath, normalized_test_21_data_results, delimiter=",")

        return True

    @classmethod
    def return_processed_data_multiclass(self):
        filepath = os.path.join(
        self.get_current_working_directory(), 'data', 'mul-nsl', "normalized_train_data_features.csv")
        if(not os.path.isfile(filepath)):
            self.nsl_process_data_multiclass()
        filepath = os.path.join(
                self.get_current_working_directory(), 'data', 'mul-nsl', "normalized_train_data_features.csv")
        normalized_train_data_features = genfromtxt(filepath, delimiter=",")
        filepath = os.path.join(
                self.get_current_working_directory(), 'data', 'mul-nsl', "normalized_train_data_results.csv")
        normalized_train_data_results = genfromtxt(filepath, delimiter=",")
        filepath = os.path.join(
                self.get_current_working_directory(), 'data', 'mul-nsl', "normalized_test_data_features.csv")
        normalized_test_data_features = genfromtxt(filepath, delimiter=",")
        filepath = os.path.join(
                self.get_current_working_directory(), 'data', 'mul-nsl', "normalized_test_data_results.csv")
        normalized_test_data_results = genfromtxt(filepath, delimiter=",")
        filepath = os.path.join(
                self.get_current_working_directory(), 'data', 'mul-nsl', "normalized_test_21_data_features.csv")
        normalized_test_21_data_features = genfromtxt(filepath, delimiter=",")
        filepath = os.path.join(
                self.get_current_working_directory(), 'data', 'mul-nsl', "normalized_test_21_data_results.csv")
        normalized_test_21_data_results = genfromtxt(filepath, delimiter=",")

        return [normalized_train_data_features, normalized_train_data_results, normalized_test_data_features, normalized_test_data_results, normalized_test_21_data_features, normalized_test_21_data_results]

    @classmethod
    def return_processed_data_binary(self):
        filepath = os.path.join(
        self.get_current_working_directory(), 'data', 'bin-nsl', "normalized_train_data_features.csv")
        if(not os.path.isfile(filepath)):
            self.nsl_process_data_binary()
        filepath = os.path.join(
                self.get_current_working_directory(), 'data', 'bin-nsl', "normalized_train_data_features.csv")
        normalized_train_data_features = genfromtxt(filepath, delimiter=",")
        filepath = os.path.join(
                self.get_current_working_directory(), 'data', 'bin-nsl', "normalized_train_data_results.csv")
        normalized_train_data_results = genfromtxt(filepath, delimiter=",")
        filepath = os.path.join(
                self.get_current_working_directory(), 'data', 'bin-nsl', "normalized_test_data_features.csv")
        normalized_test_data_features = genfromtxt(filepath, delimiter=",")
        filepath = os.path.join(
                self.get_current_working_directory(), 'data', 'bin-nsl', "normalized_test_data_results.csv")
        normalized_test_data_results = genfromtxt(filepath, delimiter=",")
        filepath = os.path.join(
                self.get_current_working_directory(), 'data', 'bin-nsl', "normalized_test_21_data_features.csv")
        normalized_test_21_data_features = genfromtxt(filepath, delimiter=",")
        filepath = os.path.join(
                self.get_current_working_directory(), 'data', 'bin-nsl', "normalized_test_21_data_results.csv")
        normalized_test_21_data_results = genfromtxt(filepath, delimiter=",")

        return [normalized_train_data_features, normalized_train_data_results, normalized_test_data_features, normalized_test_data_results, normalized_test_21_data_features, normalized_test_21_data_results]
