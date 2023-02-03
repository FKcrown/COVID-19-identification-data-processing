# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import shutil

csv_path = r"E:\DataBase\ComParE2021-CCS-CSS-Data\positive\vad\new\cough_list.csv"
csv_reader = pd.read_csv(csv_path, encoding='gb18030')
file_list = csv_reader["path"].tolist()
file_name_list = []
for file in file_list:
    file_name = file[file.rfind("\\") + 1: file.rfind(".")]
    file_name_list.append(file_name)
print(file_list)
print(file_name_list)


# def fold_extract(fold_path, copy_path, file_name_list):
#     for file in os.listdir(fold_path):
#         if file.endswith("jpg"):
#             file_name = file[file.rfind("\\") + 1: file.rfind(".")]
#             for defile_name in file_name_list:
#                 if file_name == defile_name + 'mel':
#                     print(file_name)
#                     file_path = os.path.join(fold_path, file_name + ".jpg")
#                     new_path = os.path.join(copy_path, file_name + ".jpg")
#                     shutil.copyfile(file_path, new_path)


# source_fold = r"F:\Database\Track1+CoughVid(增强)(mel)\positive"
# move_fold = r"F:\Database\Track1+CoughVid(增强)(mel)(cough_decated)\positive"
#
# fold_extract(source_fold, move_fold, file_name_list)

fold_path = r"E:\DataBase\ComParE2021-CCS-CSS-Data\positive\vad\new\mel"
copy_path = r"E:\DataBase\ComParE2021-CCS-CSS-Data\CCS_(cough_decated)\positive"

for file in os.listdir(fold_path):
    if file.endswith("jpg"):
        file_name = file[file.rfind("\\") + 1: file.rfind(".")]
        for defile_name in file_name_list:
            if file_name == defile_name + 'mel':
                print(file_name)
                file_path = os.path.join(fold_path, file_name + ".jpg")
                new_path = os.path.join(copy_path, file_name + ".jpg")
                shutil.copyfile(file_path, new_path)

print("finish!")
