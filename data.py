import os
import cv2
import random
import numpy as np
import tensorflow as tf


def shuffle(train_set):
    if train_set is None or len(train_set) == 0: return [[], []]
    tmp = list(zip(train_set[0], train_set[1]))
    random.shuffle(tmp)
    image, label = zip(*tmp)
    return [list(image), list(label)]


def load_dataset(root_dir, valid_ratio=0.2, is_train=True, normalize=True):
    img_dir = os.path.join(root_dir, 'image')
    lbl_dir = os.path.join(root_dir, 'label')
    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.png', '.jpeg', 'jpg', '.bmp'))]
    lbl_files = [os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir) if f.endswith(('.png', '.jpeg', 'jpg', '.bmp'))]
    img_num = len(img_files)
    dataset = [[], []]
    dataset[0].extend(img_files)
    dataset[1].extend(lbl_files)
    if is_train:
        dataset = shuffle(dataset)
    valid_idx = int(img_num * valid_ratio)
    img_list = [cv2.resize(cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32), (640, 640)) for f in dataset[0][valid_idx:]]
    lbl_list = [cv2.resize(cv2.imread(f, cv2.IMREAD_GRAYSCALE), (640, 640)) for f in dataset[1][valid_idx:]]
    img_list_val = [cv2.resize(cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32), (640, 640)) for f in dataset[0][:valid_idx]]
    lbl_list_val = [cv2.resize(cv2.imread(f, cv2.IMREAD_GRAYSCALE), (640, 640)) for f in dataset[1][:valid_idx]]
    images = np.array(img_list, dtype=np.float32)
    labels = np.array(lbl_list, dtype=np.uint8)
    images_val = np.array(img_list_val, dtype=np.float32)
    labels_val = np.array(lbl_list_val, dtype=np.uint8)
    if normalize:
        images = images / 255.
        images_val = images_val / 255.
    labels = np.where(labels > 0, 1, 0)
    labels_val = np.where(labels_val > 0, 1, 0)
    if is_train:
        images = images[:, :, :, np.newaxis]
        images_val = images_val[:, :, :, np.newaxis]
    return images, labels, images_val, labels_val


def make_one_hot(lbls, classes):
    labels = tf.one_hot(lbls, classes)
    return labels
