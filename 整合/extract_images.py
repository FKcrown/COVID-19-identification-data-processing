import os
import shutil


def extract_images(dir_path, image_type):
    folders = os.listdir(dir_path)
    print(folders)
    if image_type in folders:
        folders.remove(image_type)
    for folder in folders:
        split_audio_dir = os.path.join(dir_path, folder, "vad", "new")
        image_dir = os.path.join(split_audio_dir, image_type)

        file_new_dir = os.path.join(dir_path, image_type)
        image_new_dir = os.path.join(dir_path, image_type, folder)
        if not os.path.exists(file_new_dir):
            os.mkdir(file_new_dir)
        if not os.path.exists(image_new_dir):
            os.mkdir(image_new_dir)

        for file in os.listdir(image_dir):
            file_path = os.path.join(image_dir, file)
            new_file_path = os.path.join(image_new_dir, file)
            shutil.copyfile(file_path, new_file_path)


dir_path = r"E:\DataBase\DICOVA_Track1\Track1-classified(fold)\positive"
image_type = "mel"
extract_images(dir_path, image_type)
print("==finish!!==")




