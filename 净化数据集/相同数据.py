import librosa
import os
import numpy as np


def extract_audio(folder_path):
    file_path_list = []
    for file in os.listdir(folder_path):
        if file.endswith("wav"):
            file_path = os.path.join(folder_path, file)
            print()
            file_path_list.append(file_path)
    return file_path_list


negative_folder = r"E:\DataBase\ComParE2021-CCS-CSS-Data\ComParE2021_CCS\dist\negative"
positive_folder = r"E:\DataBase\ComParE2021-CCS-CSS-Data\ComParE2021_CCS\dist\positive"

n_audio = extract_audio(negative_folder)
p_audio = extract_audio(positive_folder)

i = 0
for n_file in n_audio:
    print(n_file)
    n_data, n_sr = librosa.load(n_file, sr=None)
    for p_file in p_audio:
        p_data, p_sr = librosa.load(p_file, sr=None)
        if np.array_equal(n_data, p_data):
            i = i + 1
            print("重复次数{}".format(i))
            print("negative_file", n_file)
            print("positive_file", p_file)

print("finish!")





