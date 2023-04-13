import os

os.makedirs('F:\Database\分数据集绘制谱图\整合\父目录', exist_ok=True)
jpg_files = []
for root, dirs, files in os.walk('F:\Database\分数据集绘制谱图\整合\父目录'):
    print(root)
    print(dirs)
    print(files)

