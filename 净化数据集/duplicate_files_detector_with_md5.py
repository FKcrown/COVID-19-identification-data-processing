import os
import hashlib
import shutil
import pandas as pd


def find_duplicate_files_with_md5(folder1_path, folder2_path, new_folder_path, duplicates_csv_path):
    """
    通过MD5检测重复文件，并将重复文件移动到新文件夹中，并将重复文件的文件名记录到CSV文件中。

    参数：
    folder1_path：要检查的第一个文件夹的路径
    folder2_path：要检查的第二个文件夹的路径
    new_folder_path：移动重复文件的新文件夹的路径
    duplicates_csv_path：保存重复文件名的CSV文件的路径
    """
    # 检查new_folder_path是否存在，如果不存在则创建一个名为new的文件夹
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    # 定义一个空列表来存储重复的文件名和文件夹来源
    duplicates = []

    # 遍历folder1中的所有文件
    for filename in os.listdir(folder1_path):
        # 构建文件的完整路径
        filepath = os.path.join(folder1_path, filename)

        # 判断文件是否为普通文件
        if os.path.isfile(filepath):
            # 计算文件的MD5值
            with open(filepath, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            # 遍历folder2中的所有文件，查找相同MD5值的文件
            for filename2 in os.listdir(folder2_path):
                filepath2 = os.path.join(folder2_path, filename2)
                if os.path.isfile(filepath2):
                    with open(filepath2, 'rb') as f2:
                        file_hash2 = hashlib.md5(f2.read()).hexdigest()
                    if file_hash == file_hash2:
                        # 如果找到了相同MD5值的文件，将其记录到重复文件列表中
                        folder1_name = os.path.basename(folder1_path)
                        folder2_name = os.path.basename(folder2_path)
                        folder_name = folder1_name if filename in os.listdir(folder1_path) else folder2_name
                        folder_path = os.path.join(new_folder_path, folder_name)
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
                        duplicates.append((filename, folder_name))
                        # 将重复文件移动到新文件夹中
                        shutil.move(filepath, folder_path)
                        shutil.move(filepath2, folder_path)

    # 将重复文件列表转换为Pandas DataFrame，并将其保存到CSV文件中
    df = pd.DataFrame(duplicates, columns=['file', 'folder'])
    df.to_csv(duplicates_csv_path, index=False)


negative_folder = r'F:\Database\Audios\test测试\negative'
positive_folder = r'F:\Database\Audios\test测试\positive'
duplicate_folder = r'F:\Database\Audios\test测试\duplicate'
duplicate_csv = r'F:\Database\Audios\test测试\duplicate.csv'

find_duplicate_files_with_md5(negative_folder, positive_folder, duplicate_folder, duplicate_csv)
print("Finish!")


