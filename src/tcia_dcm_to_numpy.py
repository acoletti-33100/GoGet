import os
import matplotlib.image as mpimg
from PIL import Image
import pandas as pd
import pydicom as dcm
import numpy as np
import h5py
from pathlib import Path
from utils import count_polyps_size

"""
# Notes
    This files contains all the functions related to the conversion of .dcm files to .h5 files.
    - new_dir_path: string representing the path where to save all png images
    - dcm_folder: directory where are stored all .dcm files downloaded from the TCIA dataset.
    - path_xls_no_polyp: string representing the path of the xls file for the no polyps found.
    - path_xls_10_polyp: string representing the path of the xls file for the 10 mm polyps found.
    - path_xls_6_9_polyp: string representing the path of the xls file for the 6 to 9 mm polyps found.
    - npy_dir: path where to save the 3 polyps numpy arrays as .h5 files.
# How it works
    0. reads the .xls files, with filename given by the user, to create 3 arrays:
       no_polyp, 10_mm_polyp, 6_9_mm_polyp.
    1. Converts all .dcm files to .png images maintaining the same directory
       structure as folder_dcm.
    2. Convert all png images to numpy arrays keeping as separate arrays
       no polyp/6-9polyp/10polyp.
    3. Saves all numpy arrays (as .h5), in a directory named 'pngs',
       to file for later use.
"""


def convert_dcm_to_png(folder_dcm, folder_png):
    """
    # Notes
        converts all .dcm files to .png images maintaining the same directory
        structure as folder_dcm.
        It behaves recursively, so it also checks subdirectories.
        Also, 'folder_dcm' must contain only .dcm files and directories.
        All files under 'folder_dcm' will be converted.
    # Arguments
        - folder_dcm: string representing the path of the directory input dcm images
        - folder_png: string representing the path of the directory output png images, directory is empty
    # See
        how to deal with .dcm images:
        https://medium.com/@vivek8981/dicom-to-jpg-and-extract-all-patients-information-using-python-5e6dd1f1a07d
    """
    images_path = os.listdir(folder_dcm)
    for tmp in images_path:
        path = Path(folder_dcm) / tmp
        if path.is_dir():
            new_dir = Path(folder_png) / tmp
            new_dir.mkdir(parents=True, exist_ok=True)
            convert_dcm_to_png(folder_dcm + os.sep + tmp, folder_png + os.sep + tmp)
        else:
            image = tmp
            ds = dcm.dcmread(os.path.join(folder_dcm, image))
            pixels = ds.pixel_array
            image = image.replace('.dcm', '.png')
            mpimg.imsave(os.path.join(folder_png, image), pixels)


def aux_convert_image_to_numpy(hf,
                               folder,
                               dataset_name,
                               counter,
                               index,
                               remainder_threshold,
                               remainder,
                               prefix_name,
                               img_size
                               ):
    """
    # Notes
        Auxiliary function for convert_image_to_numpy(*, *).
        This function converts the png images in "folder" to numpy
        array which is concatenated in "polyp_arr".
        It behaves recursively, so it also checks subdirectories.
        Also, 'folder' must contain only images files and directories.
        All files under 'folder' will be converted.
        This function takes care of normalizing data before returning it, divides by 255 the
        numpy array representing the images.
        Also, creates a new dataset in the .h5 files for every 100 images. So,
        for example the file no-polyp.h5 will have lots of datasets each
        of size (100, img_size, img_size, 4) with names: "no_polyps" (first 100 images),
        "no_polyp_100" (second batch of 100 images), "no_polyp_200" (third batch of images
        with 'indexes' [200, 299]), and so on.
    # Arguments
        - folder: string representing the path of the directory where to find the images to convert to numpy arrays.
        - polyp_arr: this array should be a png image in RGBA (Red, Green, Blue,
        Alpha channels, alpha is transparency) so polyp_arr.shape
        is (img_size, img_size, 4).
    # Returns
        numpy array (normalized) for the specific case (no polyp, 10 mm polyp, 6 to 9 mm polyp)
        with shape (100, img_size, img_size, 4).
    # See
        how to chunk: https://www.oreilly.com/library/view/python-and-hdf5/9781491944981/ch04.html
    """
    children = os.listdir(folder)  # get all children of folder as a list
    for child in children:
        new_potential_folder = folder + os.sep + child
        path = Path(new_potential_folder)
        if path.exists() and path.is_dir():
            counter, dataset_name = aux_convert_image_to_numpy(
                hf,
                new_potential_folder,
                dataset_name,
                counter,
                index,
                remainder_threshold,
                remainder,
                prefix_name,
                img_size
            )
        elif path.exists() and path.is_file():
            img = Image.open(path)
            img = img.resize((img_size, img_size))
            arr = np.array(img)
            # arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)  # RGBA -> RGB
            # arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)  # RGB -> GRAYSCALE
            arr = arr / 255.0
            # arr = arr.reshape(1, img_size, img_size, 1)  # GRAYSCALE
            arr = arr.reshape(1, img_size, img_size, 4)  # RGBA
            if index < hf[dataset_name][:].shape[0]:
                hf[dataset_name][index, :, :, :] = arr
                counter += 1
                index += 1
        if counter != 0 and remainder_threshold != counter and (counter % 100) == 0:
            dataset_name = prefix_name + '_' + str(counter)
            index = 0
        if remainder_threshold == counter:
            dataset_name = prefix_name + '_' + str(remainder)
            index = 0
    return counter, dataset_name


def convert_image_to_numpy(xls_no_polyp, xls_10_mm_polyp, xls_6_9_mm_polyp, folder, npy_dir):
    """
    # Notes
        Converts images to numpy arrays and saves them for later use.
        It behaves recursively, so it also checks subdirectories.
        Also, 'folder' must contain only images files and directories.
        All files under 'folder' will be converted.
    # Arguments
        - xls_no_polyp: xls file from the TCIA webiste indicating all polyps free cases.
        - xls_10_mm_polyp: xls file from the TCIA webiste indicating all polyps more than 10 mm cases.
        - xls_6_9_mm_polyp: xls file from the TCIA webiste indicating all polyps 6-9 mm cases.
        - folder: string representing the directory where to find the input images.
        - npy_dir: string representing the directory where to save numpy arrays as .h5 files.
    # How it works
        1. read all xls files in pandas dataframes.
        2. create numpy arrays where to save the converted png images.
        3. create temporary list (initially empty) to save the ids for each
            polyp category.
        4. create lists where to save only the ids from the dataframe of each
            polyp category.
        5. loop for all directories in "folder" to compare the list created in 4
            and find the path to the directories whcih belong to each category.
            So, the output will be 3 lists with the ids (which the dataset uses
            as directory name) to identify the 3 polyp category.
        6. loop for all images in "folder" to look for png images to convert
            to numpy array based on polyp category using the list created in
            point 5.
        7. Normalize (divide by 255.) the 3 array.
        8. saves the 3 numpy arrays (X, img_size, img_size, 4) in a specified directory
            as .h5 files.
    """
    df_xls_no_polyp = pd.read_excel(xls_no_polyp)
    df_xls_10_mm_polyp = pd.read_excel(xls_10_mm_polyp)
    df_xls_6_9_mm_polyp = pd.read_excel(xls_6_9_mm_polyp)
    images_path = os.listdir(folder)
    tmp_np = []
    tmp_10_p = []
    tmp_6_9_p = []
    list_np = df_xls_no_polyp.iloc[:, 0].values.tolist()
    list_10_p = df_xls_10_mm_polyp.iloc[:, 0].values.tolist()
    list_6_9_p = df_xls_6_9_mm_polyp.iloc[:, 0].values.tolist()
    for l in images_path:
        if l in list_np:
            tmp_np.append(l)
        elif l in list_10_p:
            tmp_10_p.append(l)
        elif l in list_6_9_p:
            tmp_6_9_p.append(l)
    size_no_polyp, size_10_polyp, size_6_9_polyp = count_polyps_size(tmp_np, tmp_10_p, tmp_6_9_p, folder)
    print('size_10_polyp: ', size_10_polyp)
    print('size_no_polyp: ', size_no_polyp)
    print('size_6_9_polyp: ', size_6_9_polyp)
    on_convert_image_to_numpy(tmp_np, npy_dir, size_no_polyp, folder, 'no_polyp')
    print('finished converting no polyp')
    on_convert_image_to_numpy(tmp_10_p, npy_dir, size_10_polyp, folder, 'polyp_10')
    print('finished converting 10 mm polyp')
    on_convert_image_to_numpy(tmp_6_9_p, npy_dir, size_6_9_polyp, folder, 'polyp_6_9')


def on_convert_image_to_numpy(polyp_id_list, npy_dir, size, folder, prefix_name_h5):
    """
    # Notes
    # Arguments
        - polyp_id_list: list of strings with the ids for the polyps case;
        these strings are the directories where to find the images to convert to arrays.
        - npy_dir: string representing the path where to save the .h5 files.
        - size: number of images of the given polyp category, same as from polyp_id_list.
        - folder: string representing the directory where to find the input images.
        - prefix_name_h5: string representing the name to give the dataset in thee .h5 file.
        It will be further appended with an integer (eg: "prefix_name_h5_100", "prefix_name_h5_200", ...).
    """
    with h5py.File(npy_dir + os.sep + prefix_name_h5 + '.h5', 'w') as hf:
        counter = 0  # counts the number of images currently converted
        index = 0
        remainder = (size % 100)
        remainder_threshold = size - remainder
        img_size = 256
        # grayscale version switch to 1, for rgba switch to 4
        # TODO: switch to user input to decide whether rgba or grayscale
        number_dataset = size // 100
        hundreds_counter = 0
        for i in range(number_dataset):
            hf.create_dataset(
                prefix_name_h5 + '_' + str(hundreds_counter),
                (100, img_size, img_size, 4),
                dtype='float32',
                chunks=(5, img_size, img_size, 4))
            hundreds_counter += 100
        hf.create_dataset(prefix_name_h5 + '_' + str(remainder), (remainder, img_size, img_size, 4), dtype='float32',
                          chunks=(5, img_size, img_size, 4))
        dataset_name = prefix_name_h5 + '_0'
        for j in polyp_id_list:
            counter, dataset_name = aux_convert_image_to_numpy(
                hf,
                folder + os.sep + j,
                dataset_name,
                counter,
                index,
                remainder_threshold,
                remainder,
                prefix_name_h5,
                img_size
            )
