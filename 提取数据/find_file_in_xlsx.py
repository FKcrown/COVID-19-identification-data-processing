import os
import pandas as pd
import csv

folder_path = r"F:\NeurIPs2021-data\covid19_data_0426\covid19_data_0426"
# folder_path = r"F:\Game\Celeste"

path_collection = [os.path.join(root, fn) for root, dirs, files in os.walk(folder_path) for fn in files]

print("path_collection_created!")

csv_path = r".\file_path.csv"
# csv_reader = pd.read_csv(csv_path, encoding='utf8')

with open(csv_path, 'w', encoding='utf8') as f:
    writer = csv.writer(f, lineterminator='\n')
    for line in path_collection:
        writer.writerow([line])

print("finished!")
