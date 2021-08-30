import os
from glob import glob
from tqdm import tqdm

def find_files(path, pattern):
    return glob(os.path.join(path, pattern), recursive=True)


wav_path = '/home/wuyx/data/gsc'
val_path = '/home/wuyx/data/gsc/validation_list.txt'
test_path = '/home/wuyx/data/gsc/testing_list.txt'

line_list = find_files(wav_path, pattern='**/*.wav')

fw_train_path = open('./train_list.txt', 'w')
fw_val_path = open('./val_list.txt', 'w')
fw_test_path = open('./test_list.txt', 'w')
 
with open(val_path, 'r') as f:
    val_list = f.readlines()

with open(test_path, 'r') as f:
    test_list = f.readlines()

keyword_list = []
for line in tqdm(line_list):
    word_path = line.split('/', 5)[-1]
    keyword = word_path.split('/')[0]
    if keyword == '_background_noise_':
        continue

    if keyword not in keyword_list:
        keyword_list.append(keyword)

    if word_path + '\n' in val_list:
        fw_val_path.write(line + '\n')
    elif word_path + '\n' in test_list:
        fw_test_path.write(line + '\n')
    else:
        fw_train_path.write(line + '\n')

with open('./commands.txt', 'w') as f:
    for i in keyword_list:
        f.write('{}\n'.format(i))

