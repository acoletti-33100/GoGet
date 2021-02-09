import os
import numpy as np
from sklearn.metrics import accuracy_score
import h5py
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from utils import plot_cm
from utils import plot_classification_info
from utils import concat_list
from utils import count_polyps

"""
# See
    features selection tree based: https://sklearn.org/auto_examples/ensemble/plot_forest_importances_faces.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-faces-py
    https://sklearn.org/auto_examples/ensemble/plot_forest_importances.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-py
    plot sklearn: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
    https://scikit-learn.org/stable/auto_examples/classification/plot_classification_probability.html#sphx-glr-auto-examples-classification-plot-classification-probability-py
"""


def classification(path_pos_class, path_neg_class, path_images):
    """
    # Notes
        Classifies features using a Voting classifier with a 'soft' voting scheme.
    # Arguments
        - path_pos_class: string, path where the keras checkpoints with the weights are stored
        for the positive class images.
        - path_neg_class: string, path where the keras checkpoints with the weights are stored
        for the negative class images.
        - path_images: string, path where to save images (eg: data/images).
    """
    features, labels = assemble_features_found(path_pos_class, path_neg_class, 256, 256, 4, 2)
    print('Number of features: ', features.shape)
    print('Number of labels: ', labels.shape)
    clf = ExtraTreesClassifier(n_estimators=50)
    params_ftrs_selector = grid_search_for_extra_tree(clf, features, labels)
    clf = ExtraTreesClassifier(
        n_estimators=50,
        criterion=params_ftrs_selector['criterion'],
        max_depth=params_ftrs_selector['max_depth'],
        min_samples_split=params_ftrs_selector['min_samples_split'],
        min_samples_leaf=params_ftrs_selector['min_samples_leaf'],
        max_features=params_ftrs_selector['max_features']
    )
    clf = clf.fit(features, labels)
    model = SelectFromModel(clf, prefit=True)
    x_transformed = model.transform(features)
    print('Number of features after selection: ', x_transformed.shape)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_transformed,
        labels,
        shuffle=True,
        stratify=labels,
        test_size=0.2,
        random_state=42
    )
    name_dt = 'dt'
    name_svm = 'svm'
    name_knn = 'knn'
    vc = create_voting_classifier(name_dt, name_knn, name_svm)
    params_grid = grid_search_for_vc(vc, x_train, y_train, name_dt, name_knn, name_svm)
    vc = create_voting_classifier(name_dt, name_knn, name_svm, params_grid)
    vc = vc.fit(x_train, y_train)
    print('x_valid.shape: ', x_valid.shape)
    y_pred = vc.predict(x_valid)
    acc = accuracy_score(y_valid, y_pred)
    print('Voting classifier accuracy(y_valid, y_pred): ', acc)
    plot_cm(y_valid, y_pred, path_images, 'valid', 0.5)
    y_pred_train = vc.predict(x_train)
    plot_cm(y_train, y_pred_train, path_images, 'train', 0.5)
    print('Voting classifier accuracy(y_train, y_pred_train): ', accuracy_score(y_train, y_pred_train))


def grid_search_for_extra_tree(clf, x, y):
    """
    # Notes
        Grid search for Extra tree classifier (sklearn.ensemble.ExtraTreeClassifier) used for feature selection,
        searches: criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_features
        parameters.
    # Arguments
        - clf: instance of the extra tree classifier
        - x: features
        - y: labels
    # Return
        dict with the best parameters from Grid-search, grid.best_params_
    """
    params = {
        'criterion': ('gini', 'entropy'),
        'max_depth': [1, 10],
        'min_samples_split': [2, 6],
        'min_samples_leaf': [1, 5],
        'max_features': ('auto', 'sqrt', 'log2')
    }
    grid = GridSearchCV(estimator=clf, param_grid=params, scoring='accuracy', cv=2)
    grid.fit(x, y)
    best_params = grid.best_params_
    print('###')
    print('Results of grid-search for feature selection')
    print(best_params)
    print('###')
    return best_params


def grid_search_for_vc(clf, x, y, name_dt, name_knn, name_svm):
    """
    # Arguments
    - clf: Voting classifier instance
    - x: features
    - y: labels
    - name_dt: string, name of the decision tree component in vc
    - name_knn: string, name of the K-NN component in vc
    - name_svm: string, name of the svm component in vc
    #Return
        dict with the best parameters from Grid-search, grid.best_params_
    """
    params_grid = {
        name_dt + '__max_depth': [10, 4],
        name_dt + '__min_samples_split': [0.5, 0.1],
        name_dt + '__min_samples_leaf': [0.5, 0.1],
        name_knn + '__n_neighbors': [15, 7],
        name_svm + '__kernel': ('poly', 'rbf', 'linear'),
        name_svm + '__C': [1.0, 0.7],
        name_svm + '__gamma': [0.1, 0.001]
    }
    grid = GridSearchCV(estimator=clf, param_grid=params_grid, scoring='accuracy', cv=2)
    grid.fit(x, y)
    best_params = grid.best_params_
    print('###')
    print('Results of grid-search for Voting classifier')
    print(best_params)
    print('###')
    return best_params


def features_extraction(blacklist, path_weights, pngs_folder,
                        batch_size, path_pos_ftrs, path_neg_ftrs,
                        path_xls_no_polyp, path_xls_10_polyp, path_xls_6_9_polyp,
                        path_no_p, path_10_p, path_6_9_p):
    """
    # Notes
        Extracts features from groups of 100 images using a trained instance of a gg CNN.
    # Arguments
        - blacklist: list of strings representing the ids of the batches h5py that must not be
        - path_weights: string, path where the keras checkpoints for the weights is saved,
        called as model.load_weights(path_weights).
        - batch_size: batch size for the .fit() call.
        - path_pos_ftrs: string, path where to save the output images for the positive class.
        - path_neg_ftrs: string, path where to save the output images for the negative class.
        - path_no_p: string, path to the .h5 file for the no polyp found case.
        - path_10_p: string, path to the .h5 file for the 10 mm polyp found case.
        - path_6_9_p: string, path to the .h5 file for the 6 to 9 mm polyp found case.
    """
    print('Starting extracting features')
    # find positive features
    size_no_p, size_10_p, size_6_9_p = count_polyps(path_xls_no_polyp, path_xls_10_polyp,
                                                    path_xls_6_9_polyp, pngs_folder)
    print('size_10_polyp: ', size_10_p)
    print('size_no_polyp: ', size_no_p)
    print('size_6_9_polyp: ', size_6_9_p)
    remainder_6_9_p = size_6_9_p // 100
    remainder_10_p = size_10_p // 100
    remainder_no_p = size_no_p // 100
    # otherwise class imbalance problem
    number_ftrs = np.min(np.array([remainder_no_p, remainder_6_9_p, remainder_10_p]))
    for i in range(number_ftrs):
        ftr_filename = 'ftrs' + str(i)
        print('Iteration: ', i)
        list_ids_pos = features_finder(
            blacklist,
            batch_size,
            path_no_p,
            path_10_p,
            path_6_9_p,
            path_pos_ftrs,
            1,
            True,
            path_weights,
            ftr_filename)
        concat_list(blacklist, list_ids_pos)  # to avoid going through the same images
    blacklist = []  # empty blacklist, ids pos != ids neg class
    print('###')
    print('finished extracting positive features')
    print('###')
    print('Now starting to extract negative features')
    print('###')
    for i in range(number_ftrs):
        ftr_filename = 'ftrs' + str(i)
        # path where to save negative features related data
        list_ids_neg = features_finder(
            blacklist,
            batch_size,
            path_no_p,
            path_10_p,
            path_6_9_p,
            path_neg_ftrs,
            1,
            False,
            path_weights,
            ftr_filename)
        concat_list(blacklist, list_ids_neg)  # to avoid going through the same images


def train_valid_gg_cnn(
        blacklist, batch_size, path_no_p,
        path_10_p, path_6_9_p, checkpoint_path,
        path_exp, flag_resume_from_ckpts, flag_test, checkpoint_path_save):
    """
    # Notes
    # Arguments
        - blacklist: list of strings representing the ids of the batches h5py that must not be
        - batch_size: batch size for the .fit() call.
        - path_no_p: string representing the path to the .h5 file for the no polyp found case.
        - path_10_p: string representing the path to the .h5 file for the 10 mm polyp found case.
        - path_6_9_p: string representing the path to the .h5 file for the 6 to 9 mm polyp found case.
        - checkpoint_path: string, path of a keras checkpoint from which resume training, as model.load_weights(checkpoint_path).
        - path_exp: string representing the path where to save the output images, model and model's weights.
        - flag_resume_from_ckpts: boolean flag, used to indicate whether to resume training from checkpoint_path.
        - flag_test: boolean flag, used to indicate whether this run is for testing.
        - checkpoint_path_save: string, path where to save the checkpoints while training the CNN, used as parameter of a callback.
    """
    if flag_test:
        features, labels, list_ids_used = load_features_labels(path_no_p, path_10_p, path_6_9_p, blacklist, 1)
    else:
        features, labels, list_ids_used = load_features_labels(path_no_p, path_10_p, path_6_9_p, blacklist, 8)
    with open('blacklist.txt', 'a') as f:
        print(list_ids_used, file=f)
    print('features.shape: ', features.shape)
    print('labels.shape: ', labels.shape)
    print('##')
    print('list ids used for training and validation of GoGet_CNN')
    print(list_ids_used)
    print('##')
    num_classes = np.unique(labels).shape[0]
    labels = to_categorical(labels, num_classes=num_classes)
    epochs = 20
    save_freq = batch_size * 5
    print('save_freq: ', save_freq)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path_save,  # path_exp + os.sep + 'train-ckpts' + os.sep + checkpoint_path,
        verbose=1,
        # save_freq=save_freq
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        labels,
        shuffle=True,
        stratify=labels,
        test_size=0.1,
        random_state=42
    )
    gg_cnn = create_cnn(features.shape[1], features.shape[2], features.shape[3], num_classes, False)
    if flag_resume_from_ckpts:
        gg_cnn.load_weights(checkpoint_path)
    else:
        print('Warning: either checkpoint does not exists or flag_resume_from_ckpts is False.' +
              ' Set to True if you want to resume training from checkpoint ' +
              '(path: ' + checkpoint_path + ').')
    history = gg_cnn.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[cp_callback],
        validation_data=(x_valid, y_valid),
        shuffle=True
    )
    gg_cnn.save_weights(path_exp + os.sep + 'weights')
    gg_cnn.save(path_exp + os.sep + 'gg_cnn.h5')
    plot_classification_info(gg_cnn, x_valid, x_train, y_valid, y_train, history, path_exp, batch_size)


def features_finder(
        blacklist,
        batch_size,
        path_no_p,
        path_10_p,
        path_6_9_p,
        path_exp,
        max_batch_images,
        flag_positive_polyp,
        path_weights,
        ftr_filename):
    """
    # Notes
        Extract the positive or negative features from the inputs images. The images are loaded
        from .h5 files.
    # Arguments
        - blacklist: list of strings representing the ids of the batches h5py that must not be
        fetched.
        - batch_size: batch size for the .fit() call.
        - path_no_p: string representing the path to the .h5 file for the no polyp found case.
        - path_10_p: string representing the path to the .h5 file for the 10 mm polyp found case.
        - path_6_9_p: string representing the path to the .h5 file for the 6 to 9 mm polyp found case.
        - path_exp: string representing the path where to save the output images.
        - max_batches_images:
        - flag_positive_polyp: binary flag to signal whether the features and labels all belong
        to the positive class (polyps found, 6 to 9 mm and 10 mm) or not.
        - path_weights: string, path where the keras checkpoints for the weights is saved,
        called as model.load_weights(path_weights).
        - ftr_filename:
    # Returns
        features found from the CNN as numpy array.
    """
    if flag_positive_polyp:
        features, labels, list_ids = load_positive_features_labels(path_10_p, path_6_9_p, blacklist, max_batch_images)
    else:
        features, labels, list_ids = load_negative_features_labels(path_no_p, blacklist, max_batch_images)
    print('features.shape: ', features.shape)
    print('labels.shape: ', labels.shape)
    print('np.unique(labels): ', np.unique(labels))
    print('###')
    num_classes = 2
    on_features_finder(features, labels, path_exp, batch_size, num_classes, path_weights, ftr_filename)  # TODO
    return list_ids


def assemble_features_found(path_pos_class, path_neg_class, width, height, depth, num_classes):
    """
    # Notes
     Combines the positive and negative features (weights of the output layer) found in a single
     array and compute the associated binary labels array.
    # Arguments
        - path_pos_class: string, path where the keras checkpoints with the weights are stored
        for the positive class images.
        - path_neg_class: string, path where the keras checkpoints with the weights are stored
        for the negative class images.
        - width: images width to create CNN.
        - height: height width to create CNN.
        - depth: depth width to create CNN.
        - num_classes: number of output classes for the output layer of the CNN.
    # Returns
        numpy arrays with features from both negative and positive classes, also
        return the labels array made of zeros and ones associated with the features.
    """
    gg = create_cnn(width, height, depth, num_classes, True, False)
    pos_ftrs = os.listdir(path_pos_class)
    neg_ftrs = os.listdir(path_neg_class)
    features = on_assemble_features_found(path_pos_class, pos_ftrs, gg)
    labels = np.zeros(features.shape[0])
    features = np.concatenate((features, on_assemble_features_found(path_neg_class, neg_ftrs, gg)))
    labels = np.concatenate((labels, np.ones(features.shape[0] - labels.shape[0])))
    return features, labels


def on_assemble_features_found(path_class, ftrs_path, gg):
    """
    # Notes
        Auxiliary function to load and combine the discovered features.
    # Arguments
    - path_class: string, path of the keras checkpoint where to find the weights to load in gg.
    - ftrs_path: list of strings, contains all the filenames in the directory where the keras checkpoints
    of the weights are stored.
    - gg: model, CNN.
    # Returns
        Flattened array of weights from the output layer.
    """
    list_paths = []
    for ftr_path in ftrs_path:
        aux = ftr_path.find('index')
        if aux != -1:
            list_paths.append(ftr_path[:aux - 1])
    len_list_paths = len(list_paths)
    weights_tmp = gg.get_layer('out_layer').weights
    arr_tmp = weights_tmp[0][:]
    res = np.zeros((len_list_paths, arr_tmp.shape[0] * arr_tmp.shape[1]))
    count = 0
    for l in list_paths:
        gg.load_weights(path_class + os.sep + l)
        weights_out_layer = gg.get_layer('out_layer').weights
        arr = np.array(weights_out_layer[0][:])
        res[count] = arr.flatten()
        count += 1
    return res


def on_features_finder(features, labels, path_exp, batch_size, num_classes, path_weights, ftr_filename):
    """

    # Arguments
        - features: features numpy array.
        - labels: labels numpy array.
        - path_exp: string representing the path where to save the output images.
        - batch_size: batch size for the .fit() call.
        - num_classes: integer, number of the classes of the input images.
        - path_weights: string, path where the keras checkpoints for the weights is saved,
        called as model.load_weights(path_weights).
        - ftr_filename: string, name of the file where to save the checkpoint,
        as model.save_weights(path_exp + os.sep + 'features' + os.sep + ftr_filename).
    # Returns
        features found from the CNN as numpy array.
    """
    gg_cnn = create_cnn(features.shape[1], features.shape[2], features.shape[3], num_classes, True)
    gg_cnn.load_weights(path_weights)
    gg_cnn.fit(
        features,
        labels,
        epochs=20,
        batch_size=batch_size,
        validation_split=0
    )
    gg_cnn.save_weights(path_exp + os.sep + 'features' + os.sep + ftr_filename)


def create_cnn(width, height, depth, num_classes_output, flag_ftrs_extraction, flag_summary=True):
    """
    # Notes
        Creates the CNN to use to extract features.
        Metrics used for training the CNN with images from both classes:
            - TruePositives
            - FalsePositives
            - TrueNegatives
            - FalseNegatives
            - BinaryAccuracy
            - Precision
            - Recall
            - AUC
    # Arguments
    - width: integer, width of a input image, for tf.keras.Input(shape=(width, *, *))
    - height: integer, height of a input image, for tf.keras.Input(shape=(*, height, *))
    - depth: integer, number of channels of a input image, for tf.keras.Input(shape=(*, *, depth))
    - num_classes_output: integer, number of the classes of the input images.
    - flag_ftrs_extraction: Boolean flag to indicate whether the net is used to extract features (true, train only with
    one class of images) or not (false).
    - flag_summary: boolean, flag to indicate whether to print the summary of the network (default True).
    # Returns
        CNN.
    """
    print('width: ', width)
    print('height: ', height)
    print('depth: ', depth)
    net = Sequential(name='GoGetCnn')
    net.add(tf.keras.Input(shape=(width, height, depth)))
    net.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv_1'
        )
    )
    net.add(MaxPool2D(pool_size=(2, 2), name='max_pool_1'))
    net.add(BatchNormalization(name='batch_norm_1'))
    net.add(Dropout(0.45, name='dropout_1'))
    net.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv_2'
        )
    )
    net.add(MaxPool2D(pool_size=(4, 4), name='max_pool_2'))
    net.add(BatchNormalization(name='batch_norm_2'))
    net.add(Dropout(0.7, name='dropout_2'))
    net.add(Flatten(name='flatten'))
    net.add(
        Dense(
            32,
            activation='elu',
            kernel_initializer='he_normal',
            name='dense_1'
        )
    )
    net.add(BatchNormalization(name='batch_norm_3'))
    net.add(
        Dense(
            num_classes_output,
            activation='softmax',
            name='out_layer'
        )
    )
    metrics = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='acc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ]
    if not flag_ftrs_extraction:
        net.compile(
            optimizer=optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam'),
            loss='binary_crossentropy',
            metrics=metrics
        )
    else:  # For features extraction (input only images from one class)
        net.compile(
            optimizer=optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam'),
            loss='mse'
        )
    if flag_summary:
        net.summary()
    return net


def create_voting_classifier(name_dt, name_knn, name_svc, params=None):
    """
    # Notes
        Creates the voting classifier which performs the classification
        task for the system.
    # See
        gamma, C parameters tuning: https://vitalflux.com/svm-rbf-kernel-parameters-code-sample/
        DT params tuning: https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3
    """
    if params is None:
        dt = DecisionTreeClassifier()
        knn = KNeighborsClassifier()
        svc = SVC(probability=True)
    else:
        dt = DecisionTreeClassifier(
            max_depth=params[name_dt + '__max_depth'],
            min_samples_split=params[name_dt + '__min_samples_split'],
            min_samples_leaf=params[name_dt + '__min_samples_leaf'],
        )
        knn = KNeighborsClassifier(n_neighbors=params[name_knn + '__n_neighbors'])
        svc = SVC(
            kernel=params[name_svc + '__kernel'],
            C=params[name_svc + '__C'],
            gamma=params[name_svc + '__gamma'],
            probability=True
        )
    vc = VotingClassifier(
        estimators=[(name_dt, dt), (name_knn, knn), (name_svc, svc)],
        voting='soft', weights=[2, 1, 1],
        n_jobs=-1
    )
    return vc


def load_positive_features_labels(path_10_p, path_6_9_p, blacklist, max_batch_images):
    """
    # Notes
        Loads only all positive data in a numpy array array and create a labels arrays made of ones
        and zeros.
    # Arguments
        - path_10_p: string representing the path to the .h5 file for the 10 mm polyp found case.
        - path_6_9_p: string representing the path to the .h5 file for the 6 to 9 mm polyp found case.
        - blacklist: list of strings representing the ids of the batches h5py that must not be
        fetched.
        - max_batch_images: integer, maximum number of images to fetch.
    # Returns
        Features and labels for the desired dataset.
    """
    features, list_ids = load_polyp_features(path_10_p, blacklist, '10 mm ', max_batch_images)
    f_6_9, list_ids_tmp = load_polyp_features(path_6_9_p, blacklist, '6 to 9 mm ', max_batch_images)
    labels = np.ones(features.shape[0])
    labels = np.concatenate((labels, np.ones(f_6_9.shape[0])))
    features = np.concatenate((features, f_6_9))
    concat_list(list_ids, list_ids_tmp)
    return features, labels, list_ids


def load_negative_features_labels(path_no_p, blacklist, max_batch_images):
    """
    # Notes
        Loads only all negative data in a numpy array array and create a labels arrays made of ones
        and zeros.
    # Arguments
        - path_no_p: string representing the path to the .h5 file for the no polyp found case.
        - blacklist: list of strings representing the ids of the batches h5py that must not be
        fetched.
        - max_batch_images: integer, maximum number of images to fetch.
    # Returns
        Features and labels for the desired dataset.
    """
    features, list_ids = load_polyp_features(path_no_p, blacklist, 'no ', max_batch_images)
    labels = np.zeros(features.shape[0])
    return features, labels, list_ids


def load_features_labels(path_no_p, path_10_p, path_6_9_p, blacklist, max_batch_images):
    """
    # Notes
        Load all data in a numpy array array and create a labels arrays made of ones
        and zeros.
    # Arguments
        - path_no_p: string representing the path to the .h5 file for the no polyp found case.
        - path_10_p: string representing the path to the .h5 file for the 10 mm polyp found case.
        - path_6_9_p: string representing the path to the .h5 file for the 6 to 9 mm polyp found case.
        - blacklist: list of strings representing the ids of the batches h5py that must not be
        fetched.
        - max_batch_images: integer, maximum number of images to fetch.
    # Returns
        Features and labels for the desired dataset and a list of all the used ids of the polyps
        from the .h5 file.
    """
    aux = int(max_batch_images / 2)
    features, list_ids = load_polyp_features(path_no_p, blacklist, 'no ', max_batch_images)
    f_10, tmp_ids_10 = load_polyp_features(path_10_p, blacklist, '10 mm ', aux)
    f_6_9, tmp_ids_6_9 = load_polyp_features(path_6_9_p, blacklist, '6 to 9 mm ', aux)
    concat_list(list_ids, tmp_ids_10)
    concat_list(list_ids, tmp_ids_6_9)
    labels = np.zeros(features.shape[0])
    labels = np.concatenate((labels, np.ones(f_10.shape[0])))
    labels = np.concatenate((labels, np.ones(f_6_9.shape[0])))
    features = np.concatenate((features, f_10))
    features = np.concatenate((features, f_6_9))
    return features, labels, list_ids


def load_polyp_features(path_polyp, blacklist, info, max_batch_images):
    """
    # Notes
    # Arguments
        - path_polyp: string representing the path to the .h5 file for the specific polyp found case.
        - blacklist: list of strings representing the ids of the batches h5py that must not be
        - info: string, beginning of the message to print.
        - max_batch_images: integer, maximum number of images to fetch.
    # returns
        numpy array of features for training and validation and a list of all ids used for the polyp
        dataset name in the .h5 file 'path_polyp'.
    """
    with h5py.File(path_polyp, 'r') as hf:
        count = 0
        keys = list(hf.keys())
        list_ids = []
        features = np.zeros(hf[keys[-1]][:].shape)
        for k in keys:  # hf[k][:] is a dataset with 100 images (unless last one one)
            if not (k in blacklist) and (not np.all(hf[k][:] == 0)) and (count < max_batch_images):
                features = np.concatenate((features, hf[k][:]))
                count += 1
                list_ids.append(k)
        features = features[hf[keys[-1]][:].shape[0]:, :]
        print(info + 'polyps found features number: ', features.shape)
        return features, list_ids


def plot_gg_cnn():
    print('###')
    print('plot_gg_cnn')
    print('###')
    gg = create_cnn(256, 256, 4, 2, False)
    plot_model(gg,
               to_file='model.png',
               show_shapes=True,
               show_layer_names=True,
               rankdir="LR",
               expand_nested=False,
               dpi=400,
               )
