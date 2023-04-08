"""
随机提取指定数量的文件，并从另一个指定的文件夹里提取包含该文件名的文件
"""
import os
import random
import shutil

import librosa
from tqdm import tqdm

random.seed(0)
def get_audio_files(folder, duration):
    """
    从给定的文件夹中获取所有长度大于duration秒的音频文件
    """
    audio_files = []
    for filename in tqdm(os.listdir(folder), desc='读取音频并筛选'):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            # 通过librosa库读取音频长度（秒）
            y, sr = librosa.load(filepath, sr=None)
            length = librosa.get_duration(y=y, sr=sr)
            if length >= duration:
                audio_files.append(filepath)
    return audio_files


def get_related_files(folder, audio_path_list):
    """
    从给定的文件夹中获取包含给定文件名（不含后缀）的所有音频文件
    """
    related_files = []
    for file_name in os.listdir(folder):
        for audio_name in [os.path.basename(audio_path) for audio_path in audio_path_list]:
            if os.path.splitext(audio_name)[0] in os.path.splitext(file_name)[0]:
                file_path = os.path.join(folder, file_name)
                if os.path.isfile(file_path):
                    related_files.append(file_path)
    return related_files


if __name__ == '__main__':
    # 定义文件夹路径
    original_folder = r'F:\Database\Audios\Track1+CoughVid\n(over 4s)\vad'
    related_folder = os.path.join(original_folder, 'new')
    original_output_folder = r'F:\Database\Audios\Track1+CoughVid\n&p(200 over 4s)\negative(200)'
    related_output_folder = os.path.join(original_output_folder, 'new')

    # 获取所有长度大于duration秒的音频文件，并从中随机选择200个文件
    audio_path = get_audio_files(original_folder, 3)
    audio_select_path = random.sample(audio_path, 200)

    # 获取包含这200个文件名的所有音频文件
    related_files = get_related_files(related_folder, audio_select_path)

    # 如果输出文件夹不存在，则创建
    if not os.path.exists(related_output_folder):
        os.makedirs(related_output_folder)

    if not os.path.exists(original_output_folder):
        os.makedirs(original_output_folder)

    # 将所有相关文件复制到输出文件夹中
    for filepath in related_files:
        shutil.copy(filepath, related_output_folder)

    # 将随机选取的200个文件复制到第二个输出文件夹中
    for filepath in audio_select_path:
        shutil.copy(filepath, original_output_folder)

    print('finish!')
