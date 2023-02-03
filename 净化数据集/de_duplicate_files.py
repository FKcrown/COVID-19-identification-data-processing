import pandas as pd
import numpy as np
import os
import shutil


csv_path = r"C:\Users\Lollipop\Desktop\语音数据测试\去相同数据\重复文件.csv"
with open(csv_path, 'r', encoding='UTF-8') as f:
    csv = pd.read_csv(f)
data = pd.read_csv(csv_path)
defile_list = data["path"].tolist()

def fold_extract(fold_path, move_path):
    for file in os.listdir(fold_path):
        if file.endswith("wav"):
            for defile in defile_list:
                defile_name = defile[defile.rfind("\\") + 1 :]
                if file == defile_name:
                    print(file)
                    file_path = os.path.join(fold_path, file)
                    new_path = os.path.join(move_path, file)
                    shutil.move(file_path, new_path)


n_fold = r"E:\DataBase\Coswara\negative"
p_fold = r"E:\DataBase\Coswara\positive"
move_fold = r"E:\DataBase\Coswara\duplicate_files"
fold_extract(n_fold, move_fold)
fold_extract(p_fold, move_fold)
print("finish!")