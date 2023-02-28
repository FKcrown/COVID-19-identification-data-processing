import os
import time
from tqdm import tqdm

file_list = ['file1', 'file2', 'file3']

for file in tqdm(file_list, desc='Processing files'):
    os.system('cls')
    tqdm.write(file)
    time.sleep(1)
