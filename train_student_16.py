import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from model import unet_logit, DistillationModel
from data import load_dataset, make_one_hot
from loss import distillation_loss
from callbacks import CheckPoint
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE


if __name__ == '__main__':
    root_dir = 'D:/Work/01_Knowledge_Distillation/'
    print('==================student model training==================')
    start_time = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    classes = 2
    epochs = 200
    batch_size = 2
    learning_rate = 0.001
    model_save_dir = root_dir + 'model_student/model(' + start_time + ')'
    check_point_save_dir = root_dir + 'model_student/checkpoints/' + start_time
    os.mkdir(check_point_save_dir)

    ##load dataset
    print('-----------------------load dataset-----------------------')
    data_dir = root_dir + 'data/train'
    train_images, train_labels, valid_images, valid_labels = load_dataset(data_dir, valid_ratio=0.1)
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
    train_labels = train_labels.map(lambda x: make_one_hot(x, tf.constant(classes, dtype=tf.int32)),
                                    num_parallel_calls=AUTOTUNE)
    train_dataset = tf.data.Dataset.zip((train_images, train_labels))
    train_dataset = train_dataset.batch(batch_size).prefetch(AUTOTUNE)

    valid_images = tf.data.Dataset.from_tensor_slices(valid_images)
    valid_labels = tf.data.Dataset.from_tensor_slices(valid_labels)
    valid_labels = valid_labels.map(lambda x: make_one_hot(x, tf.constant(classes, dtype=tf.int32)),
                                    num_parallel_calls=AUTOTUNE)
    valid_dataset = tf.data.Dataset.zip((valid_images, valid_labels))
    valid_dataset = valid_dataset.batch(batch_size).prefetch(AUTOTUNE)

    ##build model
    #teacher model
    teacher_model_path = root_dir + 'model_teacher/model(2022_08_01)_e113'
    teacher_model = tf.keras.models.load_model(teacher_model_path)
    input_t = teacher_model.input
    output_t = teacher_model.layers[35].output
    teacher_model = tf.keras.models.Model(input_t, output_t)
    #student model
    input_layer = tf.keras.layers.Input([640, 640, 1])
    student_model = unet_logit(input_layer, classes=classes, init_depth=16)
    #model
    model = DistillationModel(teacher_model=teacher_model, student_model=student_model)
    metric = 'accuracy'
    monitor_metric_name = 'val_accuracy'
    #metric = tf.keras.metrics.Accuracy()
    #monitor_metric_name = 'val_accuracy'
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=[metric],
                  loss=distillation_loss,
                  alpha=0.25,
                  temperature=8)

    #for transfer learning
    #model.student_model.load_weights("D:/Public/qtkim/Knowledge_Distillation/model_student/checkpoints/2022_08_01-15_01_11/Unet_Student_e113-acc0.9626-val_acc0.9643-val_loss0.2462.hdf5")

    #callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor_metric_name,
                                                      patience=20, mode='auto')
    reduce_rl = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor_metric_name, factor=0.7, patience=5,
                                                  cooldown=10, min_lr=0.00001, mode='auto')
    checkpoint = CheckPoint(checkpoint_dir=check_point_save_dir)
    callbacks_list = [early_stopping, reduce_rl, checkpoint]

    ##train
    history = model.fit(train_dataset,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=valid_dataset,
                        validation_steps=validation_steps,
                        callbacks=callbacks_list,
                        class_weight=None,
                        initial_epoch=0)

    model.student_model.save(model_save_dir)
