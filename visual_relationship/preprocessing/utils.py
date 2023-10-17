import pandas as pd
import os
import shutil
import random
import subprocess
from PIL import Image

# generate list containing the images id to download using downloader.py
def generate_image_list(type_csv: str, train_pct=1.0):
    if type_csv == "train":
        across_images_objects_train = pd.read_csv("../2.5VRD_data/across_images_objects_train.csv")["image_id"].tolist()
        within_image_objects_train = pd.read_csv("../2.5VRD_data/within_image_objects_train.csv")["image_id"].tolist()

        train_images = across_images_objects_train + within_image_objects_train
        train_images = list(set(train_images))
        train_images = random.sample(train_images, int(len(train_images) * train_pct))
        train_images = [s for s in train_images if '+' not in s]

        f = open("download_ids.txt", "w")
        for i in range(len(train_images)):
            f.write('train/' + train_images[i] + '\n')
        f.close()
        subprocess.run("python downloader.py download_ids.txt --download_folder=../images/images_train", check=True)
    elif type_csv == "validation":
        across_images_objects_val = pd.read_csv("../2.5VRD_data/across_images_objects_validation.csv")["image_id"].tolist()
        within_image_objects_val = pd.read_csv("../2.5VRD_data/within_image_objects_validation.csv")["image_id"].tolist()

        val_images = across_images_objects_val + within_image_objects_val
        val_images = list(set(val_images))
        val_images = [s for s in val_images if '+' not in s]

        f = open("download_val_ids.txt", "w")
        for i in range(len(val_images)):
            f.write('validation/' + val_images[i] + '\n')
        f.close()
        subprocess.run("python downloader.py download_val_ids.txt --download_folder=../images/images_validation", check=True)
    elif type_csv == "test":
        across_images_objects_test = pd.read_csv("../2.5VRD_data/across_images_objects_test.csv")["image_id"].tolist()
        within_image_objects_test = pd.read_csv("../2.5VRD_data/within_image_objects_test.csv")["image_id"].tolist()

        test_images = across_images_objects_test + within_image_objects_test
        test_images = list(set(test_images))
        test_images = [s for s in test_images if '+' not in s]

        f = open("download_test_ids.txt", "w")
        for i in range(len(test_images)):
            f.write('test/' + test_images[i] + '\n')
        f.close()
        subprocess.run("python downloader.py download_test_ids.txt --download_folder=../images/images_test", check=True)
    else:
        raise ValueError("Type_csv should be either train, validation or test")


# Modify across_images and within images train files according to the train images available
def create_data_subset():
    across_images_objects_train = pd.read_csv("../2.5VRD_data/across_images_objects_train.csv")
    within_image_objects_train = pd.read_csv("../2.5VRD_data/within_image_objects_train.csv")
    across_images_vrd_train = pd.read_csv("../2.5VRD_data/across_images_vrd_train.csv")
    within_image_vrd_train = pd.read_csv("../2.5VRD_data/within_image_vrd_train.csv")

    keep_list = os.listdir("../images/images_train")
    keep_list = [f.split('.')[0] for f in keep_list]

    within_image_objects_train = within_image_objects_train[within_image_objects_train['image_id'].isin(keep_list)]
    across_images_objects_train = across_images_objects_train[across_images_objects_train['image_id'].isin(keep_list)]
    within_image_vrd_train = within_image_vrd_train[within_image_vrd_train['image_id_1'].isin(keep_list)]
    within_image_vrd_train = within_image_vrd_train[within_image_vrd_train['image_id_2'].isin(keep_list)]
    across_images_vrd_train = across_images_vrd_train[across_images_vrd_train['image_id_1'].isin(keep_list)]
    across_images_vrd_train = across_images_vrd_train[across_images_vrd_train['image_id_2'].isin(keep_list)]

    folder_path = "../subset_data"
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)

    across_images_objects_train.to_csv(folder_path + '/across_images_objects_train.csv', index=False)
    within_image_objects_train.to_csv(folder_path + '/within_image_objects_train.csv', index=False)
    across_images_vrd_train.to_csv(folder_path + '/across_images_vrd_train.csv', index=False)
    within_image_vrd_train.to_csv(folder_path + '/within_image_vrd_train.csv', index=False)

    for csv_file in os.listdir("../2.5VRD_data"):
        if "train" not in csv_file:
            shutil.copy("../2.5VRD_data/" + csv_file, folder_path)


def resize_images(folder_path, target_size):
    # Ottieni la lista di file nella cartella
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)

        # Carica l'immagine utilizzando Pillow
        image = Image.open(file_path)

        if image.mode == 'L':
            # Trasforma l'immagine in RGB
            image = image.convert('RGB')

        # Ridimensiona l'immagine alla dimensione target
        resized_image = image.resize(target_size)

        # Salva l'immagine ridimensionata sovrascrivendo il file originale

        resized_image.save(os.path.join("../images/images_validation_resized", file))

def compare_folders(folder1, folder2, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the list of files in each folder
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    # Find the files in the second folder that are not in the first folder
    unique_files = files2 - files1

    # Copy the unique files to the output folder
    for file in unique_files:
        src = os.path.join(folder2, file)
        dst = os.path.join(output_folder, file)
        shutil.copy2(src, dst)

generate_image_list("validation")
generate_image_list("test")
create_data_subset()

