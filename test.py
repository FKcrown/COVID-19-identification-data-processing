import numpy as np
import librosa
#
# path = r"C:\Users\Lollipop\Desktop\语音数据测试\去相同数据\0eQAE4NgLUWYqAaMuWoO4XgIgvF2-shallow.wav"
# data, sr = librosa.load(path, sr=None)
#
# path2 = r"C:\Users\Lollipop\Desktop\语音数据测试\去相同数据\0kFnp420ZNR1jLp0JWCjMGDzvXo2-shallow.wav"
# data2, sr2 = librosa.load(path2, sr=None)
#
# print(data)
# print(data2)
#
# print((data == data2).all())


#%%
path = r"F:\Database\Track1+CoughVid（增强）\positive\resample\c8eb7e60-39c2-4d2e-ad21-855a6b3eee94_AddGaussianSNRNew_000-16.0K.wav"
data, sr = librosa.load(path, sr=None)

print(abs(max(data.max(), data.min())))

# #%%
# import csv
#
# header = ['空音频']
# data = ['Afghanistan', 652090, 'AF', 'AFG']
#
#
# with open('countries.csv', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)
#
#     # 写入头
#     writer.writerow(header)
#
#     # 写入数据
#     writer.writerow(file_path)
