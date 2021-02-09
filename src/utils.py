import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, average_precision_score
from sklearn.metrics import plot_precision_recall_curve
from pathlib import Path


def plot_classification_info(nn, x_valid, x_train, y_valid, y_train, history, path_exp, batch_size):
    """
    # Notes
        Plots images of the metrics for the specified neural network given as argument (clf)
    # Arguments
        - nn: classifier to plot images for.
        - x_valid: validation X data for clf.
        - x_train: training X data for clf.
        - y_valid: validation Y data for clf.
        - y_train: training Y data for clf.
        - history: history created from calling clf.fit(X, Y) of a classifier.
        - path_exp: string representing the path where to save the output images.
        - batch_size: batch size for the .fit() call.
    """
    new_dir = Path(path_exp + os.sep + 'images')
    new_dir.mkdir(parents=True, exist_ok=True)
    plot_acc_and_loss(history, path_exp)
    plot_metrics(history, path_exp)
    plot_loss(history, 'Loss', 0, path_exp)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    y_hat_valid = nn.predict(x_valid, batch_size=batch_size)
    y_hat_train = nn.predict(x_train, batch_size=batch_size)
    tmp_label_valid = binary_reverse_to_categorical(y_valid)
    tmp_label_hat_valid = binary_reverse_to_categorical(y_hat_valid)
    tmp_label_train = binary_reverse_to_categorical(y_train)
    tmp_label_hat_train = binary_reverse_to_categorical(y_hat_train)
    plot_cm(tmp_label_valid, tmp_label_hat_valid, path_exp, 0.5)
    plot_roc(tmp_label_train, tmp_label_hat_train, tmp_label_valid, tmp_label_hat_valid, colors, path_exp)


def count_polyps(xls_no_polyp, xls_10_mm_polyp, xls_6_9_mm_polyp, folder):
    """
    :param xls_no_polyp:
    :param xls_10_mm_polyp:
    :param xls_6_9_mm_polyp:
    :param folder:
    :return:
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
    return count_polyps_size(tmp_np, tmp_10_p, tmp_6_9_p, folder)


def count_polyps_size(tmp_np, tmp_10_p, tmp_6_9_p, path):
    """
    # Notes
    # Arguments
        - tmp_np: list of strings, contains the name of the folder (as a string) where all no polyps
        found cases images are located.
        - tmp_10_p: list of strings, contains the name of the folder (as a string) where all 10 mm polyps
        found cases images are located.
        - tmp_6_9_p: list of strings, contains the name of the folder (as a string) where all 6 to 9 mm
        polyps found cases images are located.
        - path: path as a string, it is the parent folder where to look each list element.
    # Returns
        3 integers representing the number of files contained in all directories
        listed in "tmp_np", "tmp_10_p", "tmp_6_9_p"
    """
    index_np = 0
    index_10_mm = 0
    index_6_9_mm = 0
    for i in tmp_np:
        index_np += count_number_files(path + os.sep + i)
    for i in tmp_10_p:
        index_10_mm += count_number_files(path + os.sep + i)
    for i in tmp_6_9_p:
        index_6_9_mm += count_number_files(path + os.sep + i)
    return index_np, index_10_mm, index_6_9_mm


def count_number_files(path):
    """
    # Notes
        Counts recursively the number of files in "path".
    # Arguments
        - path: path of the folder where to count the files.
    # Returns
        integer representing the number of files in "path"
    # See
        https://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python
    """
    return sum([len(files) for _, _, files in os.walk(path)])


def plot_roc(tmp_label_train, tmp_label_hat_train, tmp_label_valid, tmp_label_hat_valid, colors, path_exp):
    on_plot_roc('Train', tmp_label_train, tmp_label_hat_train, color=colors[0])
    on_plot_roc('Valid', tmp_label_valid, tmp_label_hat_valid, color=colors[0], linestyle='--')
    plt.savefig(path_exp + os.sep + 'images' + os.sep + 'roc.png')


def plot_acc_and_loss(history, path_exp):
    """
    # Notes
        Plots images of the metrics of the history argument of a classifier.
    # Arguments
        - history: history created from calling clf.fit(X, Y) of a classifier.
        - path_exp: string representing the path where to save the output images.
        - end_name: string representing
    # See
        https://appliedmachinelearning.blog/2019/07/29/transfer-learning-using-feature-extraction-from-trained-models-food-images-classification/
    """
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.xticks(np.arange(0, 30, step=5))
    plt.legend(['train', 'valid'], loc='upper left')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.xticks(np.arange(0, 30, step=5))
    plt.legend(['train', 'valid'], loc='upper right')
    plt.savefig(path_exp + os.sep + 'images' + os.sep + 'acc_loss.png')


def concat_list(l1, l2):
    """
    # Notes
        Appends each element of l2 to the end of l1, keeping the order of l2.
    # Arguments
    - l1:
    - l2:
    # Returns
        list l1 with length equals to len(l1) + len(l2).
    """
    for i in l2:
        l1.append(i)
    return l1


def plot_metrics(history, path):
    """
    # Notes
        Plots the metrics of the history of a classifier.
    # Arguments
        - history: history created from calling clf.fit(X, Y) of a classifier.
        - path: string representing the path where to save this figure.
    # Source
        https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metrics = ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.xticks(np.arange(0, 30, step=5))
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])
        plt.legend()
        plt.savefig(path + os.sep + 'images' + os.sep + metric + '.png')


def on_plot_roc(name, labels, predictions, **kwargs):
    """
    # Notes
        Plots the ROC curve.
    # Arguments
        - name:
        - labels:
        - predictions:
        - kwargs:
    # Usage
        Below are two examples of how to call this function:
            - plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
            - plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
    # Source
        https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    """
    fp, tp, _ = roc_curve(labels, predictions)
    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.xticks(np.arange(0, 30, step=5))
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5, 20])
    plt.ylim([80, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.legend(loc='lower right')


def plot_cm(labels, predictions, path, prefix_filename, p=0.5):
    """
    # Notes
        Plots a confusion matrix.
    # Arguments
        - labels:
        - predictions:
        - p:
        - path: string representing the path where to save this figure.
    # Usage
    # Source
        https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    """
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    print('confusion matrix save here: ', path)
    plt.savefig(path + os.sep + prefix_filename + '-cm.png')
    plt.show()


def from_str_to_bool(string):
    """
    # Notes
        converts string to boolean value.
    # Arguments
        - string: string, either one of these values: "true", "false"
    """
    res = string.lower()
    if res == 'true':
        return True
    else:
        return False


def binary_reverse_to_categorical(labels):
    """
    # Notes
        Decodes a binary array, made of zeros and ones, created with
        .to_categorical() to an array of only zeros and ones.
        Example: converts [[0, 1], [1, 0]] to [0, 1] (depends on how to_categorical() encodes
        the values).
    # Arguments
        - labels: array with shape (X, Y) converted with tf.keras.to_categorical(). It contains only 2 classes
        {0, 1}.
    # Returns
        numpy array of zeros and ones.
    """
    res = np.zeros(labels.shape[0])
    print('res.shape: ', res.shape)
    tmp = np.array([0, 1])
    tmp = to_categorical(tmp, num_classes=2)
    for index_label in range(labels.shape[0]):
        if np.array_equal(tmp[0, :], labels[index_label, :]):  # zero case
            res[index_label] = 0
        else:  # one case
            res[index_label] = 1
    return res


def plot_loss(history, label, n, path):
    """
    # Notes
        Plots the loss for the history argument.
        Uses a log scale to show the wide range of values.
    # Arguments
        - history:
        - label:
        - n:
        - path: string representing the path where to save this figure.
    # Source
        https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.semilogy(history.epoch, history.history['loss'],
                 color=colors[n], label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'],
                 color=colors[n], label='Val ' + label,
                 linestyle="--")
    plt.xlabel('Epoch')
    plt.xticks(np.arange(0, 30, step=5))
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path + os.sep + 'images' + os.sep + 'loss.png')


