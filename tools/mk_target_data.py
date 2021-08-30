import os
from glob import glob


def find_files(path, pattern):
    return glob(os.path.join(path, pattern), recursive=True)



target_path = '/home/wuyx/data/kws/target'

label_list = os.listdir(target_path)

train_list = []
val_list = []

for label in label_list:
    file_path = os.path.join(target_path, label)
    file_list = find_files(file_path, '*.wav')
    cut_num = int(len(file_list) * 0.9)
    train_list.extend(file_list[:cut_num])
    val_list.extend(file_list[cut_num:])

fw_train = open('./target_train.txt', 'w')
fw_val = open('./target_val.txt', 'w')

for train in train_list:
    fw_train.write('{}\n'.format(train))

for val in val_list:
    fw_val.write('{}\n'.format(val))



