"""
这个程序的功能是根据源文件夹和目标文件夹下的文件名是否包含相同的字符串来复制文件到指定文件夹
"""

# 导入os和shutil模块
import os
import shutil

from tqdm import tqdm

# 定义源文件夹、目标文件夹和指定文件夹的路径
source_folder = r"F:\Database\Audios\Track1+CoughVid\训练集&测试集\原始数据集\原始训练集\positive"
target_folder = r"F:\Database\分数据集绘制谱图\Track1+CoughVid\logMel(2s)\positive"
specified_folder = r"F:\Database\Audios\Track1+CoughVid\训练集&测试集\原始数据集\原始训练集\positive\logMel(2s)"

# 获取源文件夹下的所有文件名，并去掉后缀，组成一个列表
source_files = os.listdir(source_folder)
source_names = [os.path.splitext(file)[0] for file in source_files]

# 检测并创建specified_folder
if not os.path.exists(specified_folder):
    os.makedirs(specified_folder)

# 遍历目标文件夹下的所有文件
for file in tqdm(os.listdir(target_folder)):
    # 获取文件名（不含后缀）
    name = os.path.splitext(file)[0]
    # 判断是否有源文件夹列表中的任何一个字符串在目标文件名中
    for source_name in source_names:
        if source_name in name:
            # 拼接完整的源文件路径和目标文件路径
            target_path = os.path.join(target_folder, file)
            specified_path = os.path.join(specified_folder, file)
            # 复制文件到指定文件夹
            shutil.copy(target_path, specified_path)

print("finish!")
