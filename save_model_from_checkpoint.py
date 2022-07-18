import os
import tensorflow as tf
from datetime import datetime
from model import unet

if __name__ == '__main__':
    print('==================teacher model training==================')
    start_time = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    classes = 2
    learning_rate = 0.001
    input_layer = tf.keras.layers.Input([640, 640, 1])
    model_save_dir = 'D:/Work/01_Knowledge_Distillation/model_teacher/model(' + start_time + ')'
    checkpoint_path = 'D:/Work/01_Knowledge_Distillation/model_teacher/checkpoints/2022_07_13-16_53_52/Unet_Teacher_e021-acc0.98-val_acc0.98.hdf5'

    ##build model
    print('-----------------------build model------------------------')
    teacher_model = unet(input_layer, classes=classes, init_depth=32)
    teacher_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
    teacher_model.load_weights(checkpoint_path)

    ##save
    teacher_model.save(model_save_dir)