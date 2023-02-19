import librosa
import os
import numpy as np
from tqdm import tqdm

def extract_audio(folder_path):
    file_path_list = []
    for file in os.listdir(folder_path):
        if file.endswith("wav"):
            file_path = os.path.join(folder_path, file)
            file_path_list.append(file_path)
    return file_path_list


negative_folder = r"F:\Database\Audios\整合\negative"
positive_folder = r"F:\Database\Audios\整合\positive"

# n_audio = extract_audio(negative_folder)
# p_audio = extract_audio(positive_folder)

# 获取两个文件夹下的所有音频文件
n_audio = [os.path.join(negative_folder, f) for f in os.listdir(negative_folder) if f.endswith('.wav')]
p_audio = [os.path.join(positive_folder, f) for f in os.listdir(positive_folder) if f.endswith('.wav')]


i = 0
for n_file in tqdm(n_audio, desc='Comparing audio files', unit='file'):
    for p_file in p_audio:
        n_data, n_sr = librosa.load(n_file, sr=None)
        p_data, p_sr = librosa.load(p_file, sr=None)
        if np.array_equal(n_data, p_data):
            i = i + 1
            print("重复次数{}".format(i))
            print("negative_file", n_file)
            print("positive_file", p_file)

print("finish!")





