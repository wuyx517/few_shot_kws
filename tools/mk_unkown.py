from pydub import AudioSegment
import random
import os
from glob import glob
from tqdm import tqdm

def find_files(path, pattern):
    return glob(os.path.join(path, pattern), recursive=True)


def get_second_part_wav(main_wav_path, part_wav_path):
    sound = AudioSegment.from_mp3(main_wav_path)
    start_time = random.choice(range(len(sound)))
    end_time = start_time + 3000
    word = sound[start_time:end_time]
    word.export(part_wav_path, format="wav")

file_list = find_files('/home/wuyx/data/cv-corpus-7.0-2021-07-21/zh-CN/clips', '*.mp3')
random.shuffle(file_list)

for ifile in tqdm(file_list[:500]):
    file_name = os.path.split(ifile)[-1]
    save_path = os.path.join('/home/wuyx/data/kws/target/unkown', file_name[:-4] + '.wav')
    get_second_part_wav(ifile, save_path)
