import os
import shutil
import numpy as np
# Used datasets
# https://www.kaggle.com/datasets/mikhailma/test-dataset
# https://www.kaggle.com/datasets/cry2003/google-recaptcha-v2-images

# replace with the path to the folder containing the images from the dataset
# .-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._
image_path = 'kaggle/'
new_image_path = 'processed_dataset/images'
# .-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._

# image_directories = glob.glob(os.path.join(image_path, "*"))
image_files: list[str] = [os.path.join(dir_path, file_name) for dir_path, _, file_names in os.walk(image_path) for file_name in file_names]

# select 80% of the images for training, 10% for validation, and 10% for testing
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

train_images = []
val_images = []
test_images = []

np.random.shuffle(image_files)
train_images: list[str] = image_files[:int(len(image_files) * train_ratio)]
val_images: list[str] = image_files[int(len(image_files) * train_ratio):int(len(image_files) * (train_ratio + val_ratio))]
test_images: list[str] = image_files[int(len(image_files) * (train_ratio + val_ratio)):]


# Copy the images to the new directories
os.makedirs(new_image_path, exist_ok=True)
os.makedirs(os.path.join(new_image_path, 'train'), exist_ok=True)
os.makedirs(os.path.join(new_image_path, 'val'), exist_ok=True)
os.makedirs(os.path.join(new_image_path, 'test'), exist_ok=True)

for image_file in train_images:
    dirname = os.path.normpath(image_file).split("/")[-2]
    os.makedirs(os.path.join(new_image_path, 'train', dirname), exist_ok=True)
    shutil.copy(image_file, os.path.join(new_image_path, 'train', dirname, os.path.basename(image_file)))

for image_file in val_images:
    dirname = os.path.normpath(image_file).split("/")[-2]
    os.makedirs(os.path.join(new_image_path, 'val', dirname), exist_ok=True)
    shutil.copy(image_file, os.path.join(new_image_path, 'val', dirname, os.path.basename(image_file)))

for image_file in test_images:
    dirname = os.path.normpath(image_file).split("/")[-2]
    os.makedirs(os.path.join(new_image_path, 'test', dirname), exist_ok=True)
    shutil.copy(image_file, os.path.join(new_image_path, 'test', dirname, os.path.basename(image_file)))
