import pandas as pd
import os
import shutil


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def extracting_files(list, path):
    for i in range(len(list)):
        alllist = os.listdir(dir_files + "\\" + list[i])
        for file in alllist:
            if file[file.rfind("."):] == ".wav":
                if 'cough-heavy.wav' == file:
                    old_name = dir_files + '\\' + list[i] + '\\' + file
                    new_name = path + '\\' + list[i] + '-heavy.wav'
                    shutil.copyfile(old_name, new_name)
                if 'cough-shallow.wav' == file:
                    old_name = dir_files + '\\' + list[i] + '\\' + file
                    new_name = path + '\\' + list[i] + '-shallow.wav'
                    shutil.copyfile(old_name, new_name)


data_set_directory = "D:\\Coswara-Data-master"
curDir = os.listdir(data_set_directory)
dirs = curDir
for i in range(len(curDir)):
    dirs[i] = data_set_directory + "\\" + curDir[i]
print(dirs)



dir = dirs[42]
dir_name = os.path.split(dir)[1]
# dir = "D:\\Dataset1\\20210618"
dir_files = dir + "\\" + dir_name
dir_negative = dir + "\\negative"
dir_positive = dir + "\\positive"
csv_path = dir + "\\" + dir_name + ".csv"

if not os.path.exists(dir_files):
    os.mkdir(dir_files)
    os.system('cd {} '
              '& copy /b {}.tar.gz.a* {}.tar.gz '
              '& tar -zxvf {}.tar.gz -C {} '
              .format(dir,dir_name,dir_name,dir_name,dir))
# csv = pd.read_csv("D:/DataSet/20210714/20210714.csv")

with open(csv_path, 'r', encoding='UTF-8') as f:
    csv = pd.read_csv(f)

# csv的列数和行数
r_max = csv.shape[0]
c_max = csv.shape[1]

id_0 = csv['id'].tolist()  # 提取id列表

positive = []
state_0 = csv['covid_status'].tolist()  # 提取志愿者的新冠状况covid_status

# 按negative和positive分类提取id
for r in range(r_max):
    pd = state_0[r]
    if pd[:1] == 'p':
        positive.append(id_0[r])

id_1 = id_0
for i in range(len(positive)):
    id_1.remove(positive[i])
negative = id_1

"""
提取
"""

create_dir_not_exist(dir_negative)
create_dir_not_exist(dir_positive)

extracting_files(negative, dir_negative)
extracting_files(positive, dir_positive)

print("--------------------")
print("extract data success")
