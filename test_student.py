import os
import cv2
import tensorflow as tf
from datetime import datetime
import numpy as np
from data import load_dataset
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

if __name__ == '__main__':
    print('====================student model test====================')
    start_time = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    classes = 2
    batch_size = 2
    input_layer = tf.keras.layers.Input([640, 640, 1])
    model_path = 'D:/Work/01_Knowledge_Distillation/model_student/model(2022_07_12-22_28_50)'

    ##load dataset
    print('-----------------------load dataset-----------------------')
    root_dir = 'D:/Work/01_Knowledge_Distillation/data/test'
    test_images, test_labels, _, _ = load_dataset(root_dir, is_train=False)
    test_images = np.expand_dims(test_images, axis=3)
    test_images = tf.constant(test_images, dtype=tf.float32)

    ##load model
    print('-----------------------load model------------------------')
    model = tf.keras.models.load_model(model_path)

    ##test
    print('--------------------------test---------------------------')
    test_result_dir = 'D:/Work/01_Knowledge_Distillation/test_result_student/' + start_time
    os.mkdir(test_result_dir)
    i = 0
    while True:
        batch_images = test_images[0:2, :, :, :]
        pred_res = model.predict(batch_images)
        pred_res = tf.where(pred_res > 0.5, 1, 0)
        pred_res = np.argmax(pred_res, -1)

        img1 = np.array(batch_images[0], dtype=np.float32) * 255
        res1 = np.array(pred_res[0], dtype=np.uint8) * 255
        img2 = np.array(batch_images[1], dtype=np.float32) * 255
        res2 = np.array(pred_res[1], dtype=np.uint8) * 255
        cv2.imwrite(test_result_dir + '/' +str(i) + '_img.png', img1)
        cv2.imwrite(test_result_dir + '/' + str(i) + '_pred.png', res1)
        cv2.imwrite(test_result_dir + '/' + str(i+1) + '_img.png', img2)
        cv2.imwrite(test_result_dir + '/' + str(i+1) + '_pred.png', res2)

        test_images = test_images[2:, :, :, :]
        i += 2
        if len(test_images) == 0:
            break