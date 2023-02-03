import os
import numpy as np
import pandas as pd
import shutil

csv_path = r"C:\Users\Lollipop\Desktop\positive_cough_list.csv"
with open(csv_path, 'r', encoding='UTF-8') as f:
    csv = pd.read_csv(f)
data = pd.read_csv(csv_path)
file_list = data["path"].tolist()
print(file_list)


def fold_extract(fold_path, move_path):
    for file in os.listdir(fold_path):
        if not file.endswith("json"):
            file_name = file[: file.rfind(".")]
            for defile_name in file_list:
                if file_name == defile_name:
                    print(file_name)
                    file_path = os.path.join(fold_path, file)
                    new_path = os.path.join(move_path, file)
                    shutil.move(file_path, new_path)


p_fold = r"E:\DataBase\coughvid_20211012"
move_fold = r"E:\DataBase\coughvid_positive_over_0.6"

fold_extract(p_fold, move_fold)
print("finish!")
