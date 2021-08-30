import os
from glob import glob
from tqdm import tqdm

def find_files(path, pattern):
    return glob(os.path.join(path, pattern), recursive=True)

source_path = '/home/wuyx/data/kws/语音指令'
save_path = '/home/wuyx/data/kws/target'

file_list = find_files(source_path, '**/*.*')
i = 0
for file in file_list:
    # 提取文件名： 开始录音 或 停止录音
    file_sp = file.split('/')
    file_name = file_sp[-1].split('.')[0]
    label_name = file_name[:4]
    recorder_name = file_sp[-2]

    if label_name == '开始录音':
        label = 'start_recording'
    elif label_name == '停止录音':
        label = 'end_recording'
    else:
        print(file)
        continue

    save_file = os.path.join(save_path, label, '{}_{}.wav'.format(recorder_name, file_name))

    if os.path.exists(save_file):
        save_file = os.path.join(save_path, label, '{}_{}_{}.wav'.format(recorder_name, file_name, i))
        i += 1

    os.system('ffmpeg -i {} -ar 16000 -ac 1 {}'.format(file, save_file))



