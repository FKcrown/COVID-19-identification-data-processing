import os
import shutil


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def fold_extract(fold_path):
    file_list = []
    fold_name = fold_path[fold_path.rfind('\\') + 1: fold_path.rfind('.')]

    with open(fold_path, 'r') as f:
        for line in f.readlines():
            file_list.append(line[:-1])

    for file in os.listdir(database_path):
        if file.endswith(".jpg"):
            for file_name in file_list:
                if file[:file.rfind("-4")] == file_name:
                    old_file = os.path.join(database_path, str(file))
                    new_file = os.path.join(database_path, fold_name, str(file))
                    create_dir_not_exist(os.path.join(database_path, fold_name))
                    shutil.copyfile(old_file, new_file)
    print("{}提取成功！".format(fold_name))


"""
database_path: 总数据集目录
fold_list: 5折交叉验证数据集txt目录
"""

database_path = r"E:\DataBase\DICOVA_Track1\Track1-classified\positive\vad\new\mel"
fold_list = r"C:\Users\Lollipop\Desktop\新冠检测\covid-DiCOVA\DiCOVA_Train_Val_Data_Release\LISTS"

for fold in os.listdir(fold_list):
    if fold.endswith(".txt"):
        fold_path = os.path.join(fold_list, fold)
        fold_extract(fold_path)
