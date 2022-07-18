import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from model import unet_logit, DistillationModel
from data import load_dataset, make_one_hot
from loss import distillation_loss
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE


if __name__ == '__main__':
    print('==================student model training==================')
    start_time = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    classes = 2
    epochs = 200
    batch_size = 1
    learning_rate = 0.001

    ##load dataset
    print('-----------------------load dataset-----------------------')
    root_dir = 'D:/Work/01_Knowledge_Distillation/data/train'
    train_images, train_labels, valid_images, valid_labels = load_dataset(root_dir)
    img_num = train_images.shape[0]
    val_img_num = valid_images.shape[0]
    steps_per_epoch = int(img_num/batch_size) + bool(img_num%batch_size)
    validation_steps = int(val_img_num/batch_size) + bool(val_img_num%batch_size)

    ##augmentation
    # train_images = tf.image.random_brightness(train_images, 0.15)
    # train_images = tf.image.random_flip_left_right(train_images)
    # train_images = tf.image.random_flip_up_down(train_images)
    # valid_images = tf.image.random_brightness(valid_images, 0.15)
    # valid_images = tf.image.random_flip_left_right(valid_images)
    # valid_images = tf.image.random_flip_up_down(valid_images)

    train_images = tf.data.Dataset.from_tensor_slices(train_images)
    train_labels = tf.data.Dataset.from_tensor_slices(train_labels)
    train_dataset = tf.data.Dataset.zip((train_images, train_labels))
    train_dataset = train_dataset.batch(batch_size).prefetch(AUTOTUNE)

    valid_images = tf.data.Dataset.from_tensor_slices(valid_images)
    valid_labels = tf.data.Dataset.from_tensor_slices(valid_labels)
    valid_dataset = tf.data.Dataset.zip((valid_images, valid_labels))
    valid_dataset = valid_dataset.batch(batch_size).prefetch(AUTOTUNE)

    ##build model
    #teacher model
    model_save_dir = 'D:/Work/01_Knowledge_Distillation/model_student/model(' + start_time + ')'
    teacher_model_path = 'D:/Work/01_Knowledge_Distillation/model_teacher/model(2022_07_14-13_44_00)_teacher_F'
    teacher_model = tf.keras.models.load_model(teacher_model_path)
    input_t = teacher_model.input
    output_t = teacher_model.layers[31].output
    teacher_model = tf.keras.models.Model(input_t, output_t)
    #student model
    input_layer = tf.keras.layers.Input([640, 640, 1])
    student_model = unet_logit(input_layer, classes=classes, init_depth=16)
    #model
    model = DistillationModel(teacher_model, student_model)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],#name='sparse_categorical_accuracy'
                  loss=distillation_loss,
                  alpha=0.1,
                  temperature=10)


