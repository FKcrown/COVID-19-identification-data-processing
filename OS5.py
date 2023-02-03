import os

folder = r"E:\DataBase\DICOVA_Track1\Track1-classified\positive-1\output"
for file in os.listdir(folder):
    print(file)
    file_path = os.path.join(folder, file)
    # file_path = r"E:\DataBase\DICOVA_Track1\Track1-classified\positive-1\output
    # \AHGDPacO_cough_AddGaussianSNRNew_000.wav"
    new_file_path = file_path.split("AddGaussianSNRNew_0")[0]+file_path.split("AddGaussianSNRNew_0")[1]
    os.rename(file_path, new_file_path)
    print(new_file_path)


