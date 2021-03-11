import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import keras
from keras import backend as K
import tensorflow as tf
import random
import pickle
import matplotlib.image as mpimg
import cv2
from opts import parse_opts_offline
from keras.models import Model
from keras.layers import Dense, Activation
import itertools

def resize_frame(frame):
    frame = cv2.resize(frame, (112, 112))
    return frame

def get_frame_names(path, nclips, clip_size, step_size, is_val = False):
    frame_names = os.listdir(path)
    frame_names.sort()
    num_frames = len(frame_names)

    # set number of necessary frames
    if nclips > -1:
        num_frames_necessary = clip_size * nclips * step_size
    else:
        num_frames_necessary = num_frames

    # pick frames
    offset = 0
    if num_frames_necessary > num_frames:
        # pad last frame if video is shorter than necessary
        frame_names += [frame_names[-1]] * (num_frames_necessary - num_frames)
    elif num_frames_necessary < num_frames:
        # If there are more frames, then sample starting offset
        diff = (num_frames - num_frames_necessary)
        # Temporal augmentation
        if not is_val:
            offset = np.random.randint(0, diff)
    frame_names = frame_names[offset:num_frames_necessary +
                                offset:step_size]

    return frame_names

def load_model(model, path_to_weights, n_finetune_layers, n_finetune_classes, lr):
    # load pretrain model
    if model == 'c3d':
        import c3d_model
        model = c3d_model.get_model(summary=False)

    layer = model.layers[-2]
    model_rm_softmax = Model(inputs=model.input, outputs=layer.output)
    x = model_rm_softmax.layers[-1].output
    x = Dense(n_finetune_classes, activation='softmax')(x)
    model = Model(model_rm_softmax.input, x)

    model.load_weights(path_to_weights)

    opt = keras.optimizers.Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def load_data(dataset, selected_ids, labels, data_path, n_samples, clip_size, dim, n_channels, nclips, step_size):
    n_samples = len(selected_ids)
    y = np.empty((n_samples), dtype=int)
    X = np.empty((n_samples, clip_size, *dim, n_channels))

    for i, ID in enumerate(selected_ids):
        
        frames_names = get_frame_names(dataset + '/' + ID + '/frames/', nclips, clip_size, step_size)
        for j, name in enumerate(frames_names):
            img = mpimg.imread(dataset + '/' + ID + '/frames/' + name)
            img = np.fliplr(img)
            X[i, j,] = resize_frame(img)
            
        y[i] = labels[ID]
        print('id:', ID, '---------', 'label:', y[i])

    return X, y

if __name__ == '__main__':
    # get input params
    opt = parse_opts_offline()

    # model params
    LR = opt.learning_rate
    N_CLASS = opt.n_finetune_classes
    N_FINETUNE_LAYER = opt.n_finetune_layers
    CLIP_SIZE = opt.frame_clip_size
    STEP_SIZE = opt.frame_step_size
    DIM = (opt.frame_dim_h, opt.frame_dim_w)
    MODEL = opt.model

    # paths
    MODEL_WEIGHT = opt.model_weight
    DATA_PATH = opt.data_path

    # load model
    model = load_model(MODEL, MODEL_WEIGHT, N_FINETUNE_LAYER, N_CLASS, LR)

    ###################### !!!!!!!!!!!!!!!!!!!!!! ###########################
    # modify the following 2 variables so that you can make predictions on:
    # 1. a provided dataset and 2. your own collected dataset.
    # data_ids is a list of folder names that each folder contains the frames of a data sample.
    # data_labels is a dictionary of true labels that each data sample (the folder name) should have a corresponding label.
    data_ids = ['thumb_down', 'swiping_up', 'swiping_down', 'thumb_up', 'no_gesture', 'swiping_left', 'shaking_hand', 'swiping_right', 'doing_other_things', 'stop_sign']
    data_labels = {'thumb_down':5, 
                    'swiping_up':3, 
                    'swiping_down':2, 
                    'thumb_up':4, 
                    'no_gesture':8, 
                    'swiping_left':0, 
                    'shaking_hand':6, 
                    'swiping_right':1, 
                    'doing_other_things':9, 
                    'stop_sign':7} 
    #########################################################################

    # load data
    dataset = "transformed_class_dataset_4"
    X_test, y_true = load_data(dataset, data_ids, data_labels, DATA_PATH, 1, CLIP_SIZE, DIM, 3, 1, STEP_SIZE)

    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    print("prediction and actual")
    print(y_pred, y_true)

    _, acc = model.evaluate(X_test, keras.utils.to_categorical(y_true, num_classes=N_CLASS), verbose=0)
    print('accuracy: ' + str(round(acc, 3)))
    print("-----------\n")

