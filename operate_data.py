"""
此文件用于先将图片重命名，再划分
此文件用于划分train和val数据集
比例暂定4:1
"""
"""
此文件现在存在的缺陷是不能随机打乱图片并划分，需要增添此功能
尝试解决的思路是：在创建列表的时候，就将列表的顺序打乱，这样的话就相当于是输入了乱序的数据集了
使用random中的shufle()函数打乱原始列表的顺序, 这种方法并不改变列表的地址
此问题已经解决
"""
"""
现在需要做的是文件夹已经存在时，不在创建文件夹
同时做到，不给原来的图片改名，而是给新复制的图片改名
"""
import os
import random
import shutil
import datetime


def copy(src_path, target_path):
    # 获取文件夹里面内容
    filelist = os.listdir(src_path)
    # 遍历列表
    for file in filelist:
        # 拼接路径
        path = os.path.join(src_path, file)
        tar_path = target_path
        # 判断是文件夹还是文件
        if os.path.isdir(path):
            # tar_path = os.path.join(tar_path, file)
            tar_path = tar_path + "/" + file
            os.mkdir(tar_path)
            # 递归调用copy
            copy(path, tar_path)
        else:
            # 不是文件夹则直接进行复制
            with open(path, 'rb') as rstream:
                container = rstream.read()
                path1 = os.path.join(target_path, file)
                with open(path1, 'wb') as wstream:
                    wstream.write(container)
    else:
        print('复制完成!')


original_dataset_dir = "D:/学习/大创/data/训练数据集/data/melspec(1000_50)(vad)(fold)"
base_dir = "D:/学习/大创/data/训练数据集/data/melspec(1000_50)(vad)(fold)(new)"
path_1 = 'positive'
path_2 = 'negative'
# 加时间戳
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
base_dir = base_dir + "_" + str(nowTime)
os.mkdir(base_dir)

# 文件复制
copy_path = base_dir + "_copy"
os.mkdir(copy_path)
# 调用copy函数
copy(original_dataset_dir, copy_path)

# 分别对应划分后的训练、验证的目录
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'val')
os.mkdir(validation_dir)

# 猫的训练图集目录
train_cats_dir = os.path.join(train_dir, path_1)
os.mkdir(train_cats_dir)
# 狗的训练图集目录
train_dogs_dir = os.path.join(train_dir, path_2)
os.mkdir(train_dogs_dir)
# 猫的验证图像目录
validation_cats_dir = os.path.join(validation_dir, path_1)
os.mkdir(validation_cats_dir)
# 狗的验证图线目录
validation_dogs_dir = os.path.join(validation_dir, path_2)
os.mkdir(validation_dogs_dir)

# 导出阴性和阳性的列表
positive_path = copy_path + "/positive"
negative_path = copy_path + "/negative"
positive_list = os.listdir(positive_path)
negative_list = os.listdir(negative_path)

# 随机打乱两个列表中元素的顺序
random.shuffle(positive_list)
random.shuffle(negative_list)

"""
重命名文件
"""
num = 0
for i in positive_list:
    old_name = positive_path + os.sep + positive_list[num]
    new_name = positive_path + os.sep + "positive." + str(num) + ".jpg"
    os.rename(old_name, new_name)
    print(old_name, "=====>", new_name)
    num += 1

num = 0
for i in negative_list:
    old_name = negative_path + os.sep + negative_list[num]
    new_name = negative_path + os.sep + "negative." + str(num) + ".jpg"
    os.rename(old_name, new_name)
    print(old_name, "=====>", new_name)
    num += 1

"""
复制图片数据
"""
path_1_num_0 = len(os.listdir(positive_path))
path_2_num_0 = len(os.listdir(negative_path))
path_1_num_1 = int(path_1_num_0 * 0.8)
path_1_num_2 = path_1_num_0 - path_1_num_1
path_2_num_1 = int(path_2_num_0 * 0.8)
path_2_num_2 = path_2_num_0 - path_2_num_1

# 将前80%猫的图像复制到train_cats_dir
fnames = ['positive.{}.jpg'.format(i) for i in range(path_1_num_1)]
for fname in fnames:
    src = os.path.join(copy_path, 'positive', fname)
    dst = os.path.join(train_cats_dir, fname)
    # shutil.copyfile(src, dst):将名为src的文件的内容复制到名为dst的文件中
    shutil.copyfile(src, dst)
# 将接下来20%猫的图像复制到validation_cats_dir
fnames = ['positive.{}.jpg'.format(i) for i in range(path_1_num_2)]
for fname in fnames:
    src = os.path.join(copy_path, 'positive', fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
# 将前80%狗的图像复制到train_dogs_dir
fnames = ['negative.{}.jpg'.format(i) for i in range(path_2_num_1)]
for fname in fnames:
    src = os.path.join(copy_path, 'negative', fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
# 将接下来20%狗的图像复制到validation_dogs_dir
fnames = ['negative.{}.jpg'.format(i) for i in range(path_2_num_2)]
for fname in fnames:
    src = os.path.join(copy_path, 'negative', fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

shutil.rmtree(copy_path)
print("完成")
