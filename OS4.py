import pandas as pd
import os
import shutil


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


data_set_directory = "D:\\Coswara-Data-master"
curDir = os.listdir(data_set_directory)
dirs = curDir
for i in range(len(curDir)):
    dirs[i] = data_set_directory + "\\" + curDir[i]
print(dirs)

"""
移动整合negative和positive
"""

negative_folder = "E:\\Dataset\\negative"
positive_folder = "E:\\Dataset\\positive"
create_dir_not_exist(negative_folder)
create_dir_not_exist(positive_folder)

file_path = []
for i in range(len(dirs)):
    dir = dirs[i]
    dir_negative = dir + "\\negative"
    dir_positive = dir + "\\positive"
    dir_list = os.listdir(dir)
    for folder in dir_list:
        if folder == "negative":
            file = os.listdir(dir_negative)
            for j in range(len(file)):
                file_path.append(dir_negative + "\\" + file[j])
                old_file = file_path[j]
                new_file = negative_folder + "\\" + file[j]
                shutil.copyfile(old_file, new_file)
        if folder == "positive":
            file = os.listdir(dir_positive)
            for j in range(len(file)):
                file_path.append(dir_positive + "\\" + file[j])
                old_file = file_path[j]
                new_file = positive_folder + "\\" + file[j]
                shutil.copyfile(old_file, new_file)

print("---------------------")
print("data classify success")