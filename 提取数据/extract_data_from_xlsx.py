import pandas as pd
import os
import shutil


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_variable_name(var, locals=locals()):
    for k, v in locals.items():
        if v is var:
            return k


def extracting_files(list, path):
    for i in range(len(list)):
        files = os.listdir(dir_files)
        for file in files:
            if file == list[i]:
                old_name = os.path.join(dir_files, file)
                new_name = os.path.join(path, file)
                shutil.copyfile(old_name, new_name)
    print("=========={}提取完成==========".format(get_variable_name(list)))


file_dir = os.path.abspath(r"E:\DataBase\ComParE2021-CCS-CSS-Data\ComParE2021_CCS\dist")
dir_files = os.path.join(file_dir, "wav")
dir_negative = os.path.join(file_dir, "negative")
dir_positive = os.path.join(file_dir, "positive")
csv_path = r"E:\DataBase\ComParE2021-CCS-CSS-Data\metaData_CCS.csv"