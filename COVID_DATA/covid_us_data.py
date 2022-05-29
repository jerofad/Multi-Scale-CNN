""" From the images in the clean  folder.
    We split them into classes ['covid', 'normal', 'pneumonia', 'other']

"""
import os
from imutils import paths
import shutil
# get names of files from a directory 

source_path = 'data/image/clean'
target_path = 'data/image/covid_data/'


image_paths = list(paths.list_images(source_path))
# Create directoru for each class:

class_names = ['covid', 'normal', 'pneumonia', 'other']
for c in class_names:
    if not os.path.exists(os.path.join(target_path, c)):
        #create
        os.makedirs(os.path.join(target_path, c))

    for img_pth in image_paths:
        if img_pth.split('_')[2] == c:
            shutil.copy(img_pth, target_path+c)

# Rename the normal class folder to 'regular'

os.rename('data/image/covid_data/normal', 'data/image/covid_data/regular')