import os
import random
import shutil


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
    if not os.path.exists(trainDir):
        os.mkdir(trainDir)
    for name in sample1:
        file_path = os.path.join(fileDir, name)
        new_file_path = os.path.join(trainDir, name)
        shutil.copyfile(file_path, new_file_path)


if __name__ == '__main__':
    fileDir = r'E:\DataBase\Coswara\negative'
    trainDir = r'E:\DataBase\Coswara\negative\trans'

    moveFile(fileDir, trainDir, 500)
    print("finish!!")
