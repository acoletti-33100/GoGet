import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical
from classification_extraction import create_cnn, create_voting_classifier
from utils import binary_reverse_to_categorical, plot_cm
from utils import concat_list
from utils import plot_vc_info
from classification_extraction import assemble_features_found


def test_binary_reverse_to_categorical():
    """
    # Notes
        Test function for binary_reverse_to_categorical().
    """
    arr = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1])
    print('Original arr: ', arr)
    arr = to_categorical(arr, num_classes=2)
    print('To categorical arr: ', arr)
    arr = binary_reverse_to_categorical(arr)
    print('Reverse arr: ', arr)


def test_saved_model():
    print('###')
    print('test_saved_model')
    print('###')
    model = create_cnn(256, 256, 4, 100)
    model.load_weights('data/color/pos/train-weights/weights')
    weights_out_layer = model.get_layer('out_layer').weights
    arr = np.array(weights_out_layer[0][:])
    print('arr.shape: ', arr.shape)
    plt.imshow(arr)
    plt.show()


def test_load_ftrs():
    print('###')
    print('test_load_ftrs')
    print('###')
    model = create_cnn(256, 256, 4, 1)
    weights_out_layer = model.get_layer('out_layer').weights
    arr = np.array(weights_out_layer[0][:])
    print('arr.shape: ', arr.shape)
    print(arr)


def test_create_net():
    print('###')
    print('test_create_net')
    print('###')
    model = create_cnn(256, 256, 4, 100)
    model.load_weights('data/color/neg/train-weights/weights')
    weights_out_layer = model.get_layer('out_layer').weights
    print('type(weights_out_layer): ', type(weights_out_layer))
    print('len(weights_out_layer): ', len(weights_out_layer))
    print('weights_out_layer: ', weights_out_layer)
    arr = np.array(weights_out_layer[0][:])
    # arr = arr / 255
    plt.imshow(arr)
    plt.show()


def load_exp1_test():
    model = create_cnn(256, 256, 4)
    # model = create_cnn(256, 256, 1)
    # model.load_weights('data/training-ckpts-exp1/weights/weights')
    model.load_weights('data/color/training-ckpts/cp-0030.ckpt')
    # model1 = tf.keras.models.load_model('data/gray/models/gg_cnn.h5')
    weights_out_layer = model.get_layer('out_layer').weights
    print('type(weights_out_layer): ', type(weights_out_layer))
    print('len(weights_out_layer): ', len(weights_out_layer))
    print('type(weights_out_layer[0]): ', type(weights_out_layer[0]))
    print('type(weights_out_layer[1]): ', type(weights_out_layer[1]))
    # print('weights_out_layer: ', weights_out_layer)
    print('###')
    print('###')
    arr = np.array(weights_out_layer[0][:])
    print('arr.shape: ', arr.shape)
    plt.imshow(arr, cmap='gray', vmin=0)
    plt.show()


def test_assemble_features_found():
    print('###')
    print('test_assemble_features_found')
    print('###')
    path_pos = 'data/color/pos/features'
    path_neg = 'data/color/neg/features'
    features = assemble_features_found(path_pos, path_neg, 256, 256, 4, 2)
    print('features.shape: ', features.shape)
    print('features: ', features)
    for f in features:
        if np.all(f == 0):
            print('all zeros')


def test_vc():
    print('###')
    print('test_vc')
    print('###')
    neg_ftrs = np.zeros((32, 1))
    pos_ftrs = np.ones((32, 1))
    weights = np.random.randint(4, size=(32, 1))
    features = weights.flatten()
    print('features.shape: ', features.shape)
    for i in range(50):
        weights_tmp = np.random.randint(4, size=(32, 1))
        features = np.vstack((features, weights_tmp.flatten()))
    # labels = np.zeros(neg_ftrs.shape[0])
    # labels = np.concatenate((labels, np.ones(pos_ftrs.shape[0])))
    labels = np.random.randint(2, size=features.shape[0])
    print('features.shape: ', features.shape)
    print('labels.shape: ', labels.shape)
    exit()
    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        labels,
        shuffle=True,
        stratify=labels,
        test_size=0.1,
        random_state=42
    )
    ftrs_sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    x_transformed = ftrs_sel.fit_transform(x_train)
    vc = create_voting_classifier()
    vc.fit(x_transformed, y_train)
    y_pred = vc.predict(x_valid)
    y_pred_train = vc.predict(x_train)
    acc = accuracy_score(y_valid, y_pred)
    print('accuracy: ', acc)
    plot_cm(y_valid, y_pred, 'images/test', 0.5)


def test_concat_list():
    a = ['asd', 'assa', '23']
    b = ['re', '55', 'tre']
    c = ['1', 'a']
    print(a)
    print(b)
    concat_list(a, b)
    concat_list(a, c)
    print('Result: ', a)


def test_show_image():
    print('###')
    print('test_show_image')
    print('###')
    new_dir_path = '../../../../../../../../opt/python-project-data/avp/allh5s'
    path_no_p = new_dir_path + os.sep + 'no_polyp.h5'
    path_10_p = new_dir_path + os.sep + 'polyp_10.h5'
    with h5py.File(path_10_p, 'r') as hf:
        keys = list(hf.keys())
        arr = hf[keys[-1]][0, :]
        plt.imshow(arr)
        plt.savefig('10.png')
        plt.show()
    with h5py.File(path_no_p, 'r') as hf1:
        keys1 = list(hf1.keys())
        arr1 = hf1[keys1[-1]][0, :]
        plt.imshow(arr1)
        plt.savefig('no.png')
        plt.show()


def test_vc_plots():
    tmp = np.linspace(-1.0, 2.0, 64)
    x = np.zeros(1, 64)
    x[0] = tmp
    y = np.zeros(32)
    y = np.concatenate((y, np.ones(32)))
    for i in range(32):
        x = np.concatenate((x, np.linspace(-1.0, 0.0, 64)))
    for i in range(32):
        x = np.concatenate((x, np.linspace(0.2, 1.5, 64)))
    print('x.shape: ', x.shape)
    print('y.shape: ', y.shape)
    plot_vc_info(x, y)


def test_load_h5(folder):
    """
    Test if loads correctly.
    """
    print('###')
    print('test_load_h5')
    print('###')
    with h5py.File(folder, 'r') as hf:
        keys = hf.keys()
        print(keys)
        print(len(keys))
        print('type(keys): ', type(keys))
        exit()
        c = 0
        l = []
        img_size = 256
        zeros = np.zeros((img_size, img_size, 4))
        # zeros = np.zeros((img_size, img_size, 1))
        for k in keys:
            print(k + ': ', hf[k].shape)
            size = hf[k].shape[0] - 2
            for i in range(size):
                # and np.array_equal(hf[k][i, :, :, :], zeros):
                if zeros.shape == hf[k][i, :, :, :].shape:
                    if np.all(hf[k][i, :, :, :] == 0):
                        c += 1
                        l.append(i)
                    else:
                        # to show grayscale images
                        plt.imshow(hf[k][i, :, :, :], cmap='gray', vmin=0)
                        plt.show()
                    if not np.all(hf[k][i, :, :, :] == 0) and \
                            np.array_equal(hf[k][i, :, :, :], hf[k][i + 1, :, :, :]):
                        print('issue')
            print(k + ': ', c)
            # print('l: ', l)
            print('###')
            l = []
            c = 0
            # print('###')
            # print("next key")
            # print('###')
            # img = plt.imshow(arr[i, :, :, :])
            # plt.show()

