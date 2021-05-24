from models.classic import Classic
from models.runner import Runner

while True:
    try:
        dataset = int(input("Select a dataset:\r\n0. NSL\r\n1. CICIDS\r\n"))
        if not (0 <= dataset <= 1):
            raise Exception
    except Exception:
        print("\r\nInvalid choice!\r\n")
        continue
    break
while True:
    try:
        run_type = int(
            input("Select a type:\r\n0. Binary\r\n1. Multiclass\r\n"))
        if not (0 <= run_type <= 1):
            raise Exception
    except Exception:
        print("\r\nInvalid choice!\r\n")
        continue
    break
while True:
    try:
        model_type = int(
            input("Select a model:\r\n0. Classic\r\n1. Conv1D\r\n2. "
                  "DNN\r\n3. GRU\r\n4. LSTM\r\n5. RNN\r\n")
        )
        if not (0 <= model_type <= 6):
            raise Exception
    except Exception:
        print("\r\nInvalid choice!\r\n")
        continue
    if model_type != 0:
        try:
            epochs = int(
                input(
                    "Select number of epochs:\r\n0. 1\r\n1. 25\r\n2. "
                    "50\r\n3. 100\r\n4. 250\r\n4. 500\r\n"))
            if not (0 <= epochs <= 4):
                raise Exception
            epochs_num = 0
            if epochs == 0:
                epochs_num = 1
            elif epochs == 1:
                epochs_num = 25
            elif epochs == 2:
                epochs_num = 50
            elif epochs == 3:
                epochs_num = 100
            elif epochs == 4:
                epochs_num = 250
            elif epochs == 5:
                epochs_num = 500
        except Exception:
            print("\r\nInvalid choice!\r\n")
            continue
    break

if model_type == 0:
    Classic.run(run_type, dataset)
else:
    Runner.run(run_type, dataset, model_type, epochs_num)
