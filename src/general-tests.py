import os
import numpy as np
import pandas as pd
import pydicom as dcm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras import datasets


def main():
    """
    """
    various_test()


def various_test():
    # convert_dcm_to_png('data/CT_COLONOGRAPHY/1.3.6.1.4.1.9328.50.4.0001/01-01-2000-1-Abdomen24ACRINColoIRB2415'
    #                   '-04_Adult-0.4.1/', 'data/png')
    # to_gray_scale('data/prova-png/prova.png', 'fig.png')
    path = 'data/CT_COLONOGRAPHY/1.3.6.1.4.1.9328.50.4.0001/01-01-2000-1-Abdomen24ACRINColoIRB2415-04_Adult-0.4.1/2' \
           '.000000-Topo_prone_0.6_T20s-.1224/1-1.dcm'
    path1 = 'data/CT_COLONOGRAPHY/1.3.6.1.4.1.9328.50.4.0001/01-01-2000-1-Abdomen24ACRINColoIRB2415-04_Adult-0.4.1/1' \
            '.000000-Topo_supine_0.6_T20s-.1222/1-1.dcm'
    path2 = 'data/CT_COLONOGRAPHY/1.3.6.1.4.1.9328.50.4.0001/01-01-2000-1-Abdomen24ACRINColoIRB2415-04_Adult-0.4.1/4' \
            '.000000-Coloprone_1.0_B30f-0.4.2/1-001.dcm'
    path3 = 'data/CT_COLONOGRAPHY/1.3.6.1.4.1.9328.50.4.0001/01-01-2000-1-Abdomen24ACRINColoIRB2415-04_Adult-0.4.1/3' \
            '.000000-Colosupine_1.0_B30f-4.563/1-001.dcm'
    show_4_dcm_image(path, path1, path2, path3)
    # prova()
    # show_example_dcm_image(path)
    # show_example_png_image('pngs/A/B/b1.png')
    # show_example_png_image('data/prova-png/prova.png')
    # convert_image_to_numpy('data/prova-png/prova.png')
    # show_example_dcm_image('data/prova/prova.dcm')


def prova():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    print('train_images.shape: ', train_images.shape)
    print('train_labels.shape: ', train_labels.shape)
    print('type(train_images): ', type(train_images))
    print('type(train_labels): ', type(train_labels))


def test_aux_0(ds, keyword_list):
    for k in keyword_list:
        print(k)
        print('type: ', type(ds.data_element(k)))
        print('values: ', ds.data_element(k))
        print('######')


def test_aux_1(ds):
    ds.convert_pixel_data('jpeg_ls')
    ds.decompress('jpeg_ls')
    arr1 = ds.pixel_array
    print('type(arr1): ', type(arr1))
    print('arr1.shape: ', arr1.shape)
    return arr1


def test_plot(arr, arr1):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(arr)
    ax[1].imshow(arr1)
    ax[0].set_title('ds.pixel_array')
    ax[1].set_title('numpy')
    plt.tight_layout()
    plt.show()
    plt.imshow(arr1)
    plt.show()


def test_convert_dcm_to_pixel(img):
    ds = dcm.dcmread(img)
    # np.set_printoptions(threshold=np.inf)
    keyword_list = ds.dir()
    print('type(ds): ', type(ds))
    print('type(keyword_list): ', type(keyword_list))
    print('list number of elements: ', len(keyword_list))
    print('type(ds.pixel_array): ', type(ds.pixel_array))
    print('ds.pixel_array.shape: ', ds.pixel_array.shape)
    ds.decompress('numpy')
    print(ds.get_private_item(2, 1, 'PixelData'))
    # test_aux_0(keyword_list)
    # arr1 = test_aux_1(ds)
    # test_plot(ds.pixel_array, arr1)


def show_4_dcm_image(img0, img1, img2, img3):
    """
    Plots an example image in dcm format from the dataset
    """
    ds0 = dcm.dcmread(img0)
    ds1 = dcm.dcmread(img1)
    ds2 = dcm.dcmread(img2)
    ds3 = dcm.dcmread(img3)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].imshow(ds0.pixel_array)
    ax[0, 1].imshow(ds1.pixel_array)
    ax[1, 0].imshow(ds2.pixel_array)
    ax[1, 1].imshow(ds3.pixel_array)
    ax[0, 0].set_title('Topo prone (1-1.dcm)')
    ax[0, 1].set_title('Topo supine (1-1.dcm)')
    ax[1, 0].set_title('Colo prone (1-001.dcm)')
    ax[1, 1].set_title('Colo supine (1-001.dcm)')
    txt = '1.3.6.1.4.1.9328.50.4.0001/01-01-2000-1-Abdomen24ACRINColoIRB2415-04_Adult-0.4.1'
    fig.text(0.5, .006, txt, ha='center')
    plt.tight_layout(True)
    plt.show()


def show_example_dcm_image(image):
    """
    Plots an example image in dcm format from the dataset
    """
    ds = dcm.dcmread(image)
    img = ds.pixel_array
    print('img.shape: ', img.shape)
    print('type(image): ', type(image))
    plt.imshow(img)
    plt.show()


def show_example_png_image(image):
    img = Image.open(image)
    arr = np.array(img)
    print(img.format)
    print(img.mode)
    print(img.size)
    print('type(arr): ', type(arr))
    print('arr.shape: ', arr.shape)
    img.show(arr)
    plt.show()


def convert_dcm_to_png(folder_dcm, folder_png):
    """
    # Arguments
        folder_dcm: path of the dcm images
        folder_png: path of the png images
    # See
        how to deal with .dcm images:
        https://medium.com/@vivek8981/dicom-to-jpg-and-extract-all-patients-information-using-python-5e6dd1f1a07d
    """
    images_path = os.listdir(folder_dcm)
    print('images_path: ', images_path)
    for n, image in enumerate(images_path):
        print('image: ', image)
        exit()
        ds = dcm.dcmread(os.path.join(folder_dcm, image))
        pixels = ds.pixel_array
        image = image.replace('.dcm', '.png')
        mpimg.imsave(os.path.join(folder_png, image), pixels)


def convert_image_to_numpy(image):
    img = np.array(Image.open(image))
    print('type(arr): ', type(img))
    print('arr.shape: ', img.shape)
    plt.imshow(img)
    plt.show()


def to_gray_scale(img, gray_img):
    """
    # TODO
        change to deal an entire folder and give as input the gray directory for the
        gray output images.
    # Notes
        Read an image & convert it to gray-scale.
    # Arguments
        img: path to the input colored image.
        gray_img: path to the gray output image to save.
    # See
        https://www.tutorialspoint.com/working-with-images-in-python
    """
    image = Image.open(img).convert('L')
    image.save(gray_img)


main()
