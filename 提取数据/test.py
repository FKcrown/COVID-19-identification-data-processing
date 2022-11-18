import numpy as np
import pandas as pd
import os
import csv

xlsx_path = r"C:\Users\Lollipop\Desktop\NeurlPS2021_negative.xlsx"

data = pd.read_excel(xlsx_path)

r_max = data.shape[0]
c_max = data.shape[1]
print(data.shape)
# print(data)
uid = data["Uid"].tolist()
# covid_test = data["Covid_Test"].tolist()
folder_name = data['Folder Name'].tolist()
cough_filename = data['Cough filename'].tolist()
dataset_path = r'F:\NeurIPs2021-data\covid19_data_0426\covid19_data_0426'
file_path_xlsx = []
n = 0
for i in range(len(uid)):
    file_path = os.path.join(dataset_path,uid[i],folder_name[i],cough_filename[i])
    if os.path.isfile(file_path):
        file_path_xlsx.append(file_path)
    else:
        # print('{} is not a file'.format(file_path))
        n += 1

csv_path = r'.\file_path_in_xlsx_negative.csv'
with open (csv_path, 'w', encoding='utf8') as f:
    writer = csv.writer(f, lineterminator='\n')
    for line in file_path_xlsx:
        writer.writerow([line])


print('finish')
print(n)
