"""
指定文件夹名称，并在根目录下搜索，讲指定文件夹复制到根目录下
"""

import os
import shutil

# 指定根文件夹路径
ROOT_FOLDER = r'F:\Database\Audios\Track1+CoughVid\negative'

# 指定要搜索的文件夹名称列表
SEARCH_FOLDER_NAMES = ['female', 'male']

# 指定目标文件夹路径
DEST_FOLDER = r'F:\Database\Audios\Track1+CoughVid'

# 遍历根文件夹
for current_folder, subfolders, files in os.walk(ROOT_FOLDER):
    # 检查当前文件夹是否为指定的搜索文件夹名称之一
    if os.path.basename(current_folder) in SEARCH_FOLDER_NAMES:
        # 构建目标文件夹路径，保留原文件夹名称
        dest_folder_path = os.path.join(DEST_FOLDER, os.path.relpath(current_folder, ROOT_FOLDER))

        # 创建目标文件夹
        os.makedirs(dest_folder_path, exist_ok=True)

        # 遍历搜索文件夹中的所有文件，复制到目标文件夹
        for file_name in files:
            file_path = os.path.join(current_folder, file_name)
            dest_file_path = os.path.join(dest_folder_path, file_name)
            shutil.copy(file_path, dest_file_path)
