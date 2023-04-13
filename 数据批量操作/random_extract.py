"""
该文件用于在文件夹中随机提取指定数量的文件
"""


import os
import random
import shutil
import sys

from tqdm import tqdm

random.seed(0)


def moveFile(fileDir, trainDir, number):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    for file_name in pathDir:
        if not file_name.endswith(".wav"):
            pathDir.remove(file_name)
    filenumber = len(pathDir)
    # rate1 = 0.8  # 自定义抽取csv文件的比例，比方说100张抽80个，那就是0.8
    rate1 = number/filenumber
    picknumber1 = int(filenumber * rate1)  # 按照rate比例从文件夹中取一定数量的文件
    sample1 = random.sample(pathDir, picknumber1)  # 随机选取picknumber数量的样本
    if not os.path.exists(os.path.dirname(trainDir)):   # 如果tranDir上一级文件夹不存在则创建
        os.makedirs(os.path.dirname(trainDir))
    if not os.path.exists(trainDir):
        os.mkdir(trainDir)
    for name in tqdm(sample1):
        file_path = os.path.join(fileDir, name)
        new_file_path = os.path.join(trainDir, name)
        shutil.copyfile(file_path, new_file_path)


def random_extract_files(src_folder, dst_folder1, ratio, dst_folder2=None):
    """
    将源文件夹下的文件按比例随机提取到目标文件夹1中，剩余文件提取到目标文件夹2中
    :param src_folder: 源文件夹路径
    :param dst_folder1: 目标文件夹1的路径
    :param ratio: 提取到目标文件夹1中的文件比例，范围为0到1之间的小数
    :param dst_folder2: 目标文件夹2的路径，可选，默认为None，即不对剩余部分文件进行操作
    """
    # 检查并创建目标文件夹1
    os.makedirs(dst_folder1, exist_ok=True)
    os.makedirs(dst_folder2, exist_ok=True)

    # 遍历源文件夹下的所有文件
    for src_root, src_dirs, src_files in os.walk(src_folder):
        # 如果当前文件夹为空，则跳过该文件夹
        if not src_files:
            continue

        # 计算要提取到目标文件夹1中的文件数目
        num_files = len(src_files)
        num_extract1 = int(num_files * ratio)

        # 遍历源文件夹下的所有文件
        src_files = sorted(os.listdir(src_folder))
        src_list = list(range(len(src_files)))
        random.seed(0)
        random.shuffle(src_list)
        extract_list = src_list[:num_extract1]

        # 将一部分文件提取到目标文件夹1中，剩余文件提取到目标文件夹2中（如果有传入dst_folder2的话）
        for i, file in tqdm(enumerate(src_files), total=len(src_files), file=sys.stdout, ncols=80, colour='yellow'):
        # for i, file in enumerate(src_files):
            src_file = os.path.join(src_root, file)
            # tqdm.write(src_file)
            if i in extract_list:
                dst_file = os.path.join(dst_folder1, file)
                tqdm.write('{}:{}'.format(i, os.path.relpath(src_file, src_folder)))
            elif dst_folder2 is not None:
                dst_file = os.path.join(dst_folder2, file)
            else:
                continue
            shutil.copy(src_file, dst_file)

if __name__ == '__main__':
    src_folder = r'F:\Database\分数据集绘制谱图\整合\2s频谱图'
    dst_folder = r'F:\Database\分数据集绘制谱图\整合\测试集&训练集\测试集'
    dst_folder2 = r'F:\Database\分数据集绘制谱图\整合\测试集&训练集\训练集'
    # moveFile(src_folder, dst_folder, 5000)
    #
    # """
    # 5折数据集提取
    # """
    # for i in range(5):
    #     i += 1
    #     dst_folder = r'F:\Database\Audios\Track1+CoughVid\5折数据集\v{}\negative'.format(i)
    #     moveFile(src_folder, dst_folder, 500)
    # print("finish!!")

    """
    将一个文件夹下的所有子文件夹和文件移动到另一个文件夹下,并保留之前目录级别关系
    """
    for root, dirs, files in os.walk(src_folder):
        if not dirs and files:  # 如果当前文件夹不存在子文件夹，但存在文件
            origin_path = root
            new_path = os.path.join(dst_folder, os.path.relpath(root, src_folder))
            new_path2 = os.path.join(dst_folder2, os.path.relpath(root, src_folder))
            random_extract_files(origin_path, new_path, 0.1, new_path2)
            print("origin_path: {}".format(origin_path))
