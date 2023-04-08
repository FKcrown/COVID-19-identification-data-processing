"""
此程序使用ffmpeg将文件夹中的视频和音频文件转换为wav格式
"""
import mimetypes
import os
import subprocess

mimetypes.add_type('audio/amr', '.amr')  # 添加amr文件类型


def guess_file_type(filename):
    """
    判断文件类型，支持判断的文件类型有：音频文件、视频文件、图片文件
    :param filename: 需要判断的文件路径
    :return: 文件类型名称
    """
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type is None:
        return 'Unknown'
    elif mime_type.startswith('audio'):
        return 'Audio'
    elif mime_type.startswith('video'):
        return 'Video'
    elif mime_type.startswith('image'):
        return 'Image'
    else:
        return 'Unknown'


# 遍历folder_path文件夹下的所有文件，并判断是否为音频文件或者视频文件，如果是则转换为wav格式
def file_to_wav(origin_folder_path, convert_wav_folder):
    """
    遍历folder_path文件夹下的所有文件，并判断是否为音频文件或者视频文件，如果是则转换为wav格式
    :param origin_folder_path: 需要转换的文件夹路径
    :param convert_wav_folder: 转换后的wav文件存放的文件夹路径
    """
    if not os.path.exists(convert_wav_folder):
        os.makedirs(convert_wav_folder)
    for file in os.listdir(origin_folder_path):
        input_path = os.path.join(origin_folder_path, file)
        # 判断文件是否为音频文件或者视频文件
        if guess_file_type(input_path) == "Audio" or guess_file_type(input_path) == "Video":
            output_path = os.path.join(convert_wav_folder, os.path.splitext(file)[0] + ".wav")
            print(output_path)
            input_list.append(input_path)
            # 将音频文件转换为wav格式
            command = f"ffmpeg -loglevel quiet -y -i {input_path} -ab 160k -ac 1 -ar 16000 -vn {output_path}"
            """
            -loglevel quiet: 不输出日志信息。
            -y: 覆盖输出文件。
            -i：指定输入文件。
            -ab：设置音频比特率。
            -ac：设置音频通道数。
            -ar：设置音频采样率。
            -vn：禁用视频。
            """
            subprocess.call(command, shell=True)
    print("转换完成")


input_list = []
folder_path = r"F:\Database\Audios\自建数据集\202065794_附件_3"
wav_folder = r"F:\Database\Audios\自建数据集\202065794_附件_3\wav"
file_to_wav(folder_path, wav_folder)

suffix = {}
# 统计input_list中的文件名，以后缀名称为key，相同后缀的文件名为value，存入suffix字典中
for file in input_list:
    if os.path.splitext(file)[1] not in suffix.keys():
        suffix[os.path.splitext(file)[1]] = [file]
    else:
        suffix[os.path.splitext(file)[1]].append(file)
print(suffix)
