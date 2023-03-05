"""
遍历指定文件夹下的同名文件夹，并将文件复制到根目录下的同名文件夹中
主要通过遍历根目录下的所有文件夹，找到同名文件夹，然后将同名文件夹中的所有文件复制到根目录下的同名文件夹中
"""

import os
import shutil

# 遍历的根目录
ROOT_FOLDER = r'F:\Database\Audios\Track1+CoughVid'

# 要遍历的同名文件夹名称列表
TARGET_FOLDER_NAMES = ['male', 'female']

# 遍历根目录及其子文件夹
for current_folder, subfolders, files in os.walk(ROOT_FOLDER):
    # 判断当前文件夹是否为指定的同名文件夹
    if os.path.basename(current_folder) in TARGET_FOLDER_NAMES:
        # 获取同名文件夹在根目录下的路径
        target_folder_path = os.path.join(ROOT_FOLDER, os.path.basename(current_folder))

        # 判断同名文件夹是否存在
        if not os.path.exists(target_folder_path):
            os.mkdir(target_folder_path)

        # 遍历同名文件夹
        for file_name in files:
            file_path = os.path.join(current_folder, file_name)
            target_file_path = os.path.join(target_folder_path, file_name)
            shutil.copy(file_path, target_file_path)