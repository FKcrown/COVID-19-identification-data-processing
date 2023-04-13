import os
import shutil

from tqdm import tqdm

spec_type = ['chirplet', 'MFCC', 'logMel', 'TFDF']
classify_type = ['negative', 'positive']


def move_spec(folder_path, spec_folder):
    """
    将folder_path下的谱图文件按谱图种类移动到spec_folder下
    :param folder_path: 要移动的文件夹总目录
    :param spec_folder: 存放谱图的文件夹
    """
    os.makedirs(spec_folder, exist_ok=True)
    for spec in tqdm(spec_type, desc="移动谱图", ncols=80, colour='yellow'):
        spec_path = os.path.join(spec_folder, spec)
        os.makedirs(spec_path, exist_ok=True)
        for classify in classify_type:
            classify_path = os.path.join(spec_path, classify)
            origin_folder = os.path.join(folder_path, classify, 'vad', 'new', spec)
            new_folder = classify_path
            try:
                # 尝试使用shutil.copytree函数复制文件夹
                shutil.copytree(origin_folder, new_folder)
            except FileExistsError:
                # 如果new_folder文件已存在，则删除
                shutil.rmtree(new_folder)
                # 再次尝试使用shutil.copytree函数复制文件夹
                shutil.copytree(origin_folder, new_folder)


folder_path = r'F:\Database\Audios\整合'
spec_folder = r"F:\Database\分数据集绘制谱图\整合\2s频谱图合集"
move_spec(folder_path, spec_folder)
