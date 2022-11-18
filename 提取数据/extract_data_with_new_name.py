"""
old_path_list: 要提取的文件路径
new_name_list: 提取后的文件名
dir_path: 提取后放到的文件夹路径
"""

import pandas as pd
import os
import shutil
import tqdm


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_variable_name(var, locals=locals()):
    for k, v in locals.items():
        if v is var:
            return k


def extracting_files_with_new_name(old_path_list, new_name_list, dir_path):
    for i in tqdm.trange(len(old_path_list)):
        old_file = old_path_list[i]
        new_file = os.path.join(dir_path, new_name_list[i])
        shutil.copyfile(old_file, new_file)
    print("=========={}提取完成==========".format(get_variable_name(old_path_list)))


negative_csv = r'.\file_path_in_xlsx_negative.csv'
positive_csv = r'.\file_path_in_xlsx_positive.csv'
positive_name = r'.\file_path_positive_newname.csv'
negative_name = r'.\file_path_negative_newname.csv'

negative = pd.read_csv(negative_csv)
positive = pd.read_csv(positive_csv)
negative2 = pd.read_csv(negative_name)
positive2 = pd.read_csv(positive_name)

negative_path = negative["file_path"].tolist()
positive_path = positive["file_path"].tolist()

negative_new_name = negative2["file_path"].tolist()
positive_new_name = positive2["file_path"].tolist()

negative_dir = r'F:\Database\NeurIPs2021\negative'
positive_dir = r'F:\Database\NeurIPs2021\positive'

"""
提取
"""

create_dir_not_exist(negative_dir)
create_dir_not_exist(positive_dir)

extracting_files_with_new_name(negative_path, negative_new_name, negative_dir)
extracting_files_with_new_name(positive_path, positive_new_name, positive_dir)

print('finish!!')