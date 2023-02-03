import pandas as pd
import os
import shutil


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_variable_name(var, locals = locals()):
    for k, v in locals.items():
        if v is var:
            return k


def extracting_files(list, path):
    for i in range(len(list)):
        files = os.listdir(dir_files)
        for file in files:
            if file[:file.rfind(".")] == list[i]:
                old_name = os.path.join(dir_files, file)
                new_name = os.path.join(path, file)
                shutil.copyfile(old_name, new_name)
    print("=========={}提取完成==========".format(get_variable_name(list)))


dir = os.path.abspath("E:\\DICOVA_Track1")
dir_files = os.path.join(dir, "Audio")
csv_path = os.path.join(dir, 'metadata.csv')
dir_negative = os.path.join(dir, "negative")
dir_positive = os.path.join(dir, "positive")

with open(csv_path, 'r', encoding='UTF-8') as f:
    csv = pd.read_csv(f)

r_max = csv.shape[0]
c_max = csv.shape[1]

file_name = csv['File_name'].tolist()  # 提取File_name

positive = []
negative = []
state = csv['Covid_status'].tolist()  # 提取志愿者的新冠状况Covid_status

# 按negative和positive分类提取id
for r in range(r_max):
    pd = state[r]
    if pd[:1] == 'p':
        positive.append(file_name[r])
    if pd[:1] == 'n':
        negative.append(file_name[r])

"""
提取
"""

create_dir_not_exist(dir_negative)
create_dir_not_exist(dir_positive)

extracting_files(negative, dir_negative)
extracting_files(positive, dir_positive)
