import os
import tensorflow as tf
from datetime import datetime
from model import unet_logit

if __name__ == '__main__':
    print('==================teacher model training==================')
    start_time = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    classes = 2
    learning_rate = 0.001
    input_layer = tf.keras.layers.Input([640, 640, 1])
    model_save_dir = 'D:/Work/01_Knowledge_Distillation/model_student/model(' + start_time + ')'
    checkpoint_path = 'D:/Work/01_Knowledge_Distillation/model_student/checkpoints/asdf.hdf5'

    ##build model
    print('-----------------------build model------------------------')
    teacher_model = unet_logit(input_layer, classes=classes, init_depth=16)
    teacher_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
    teacher_model.load_weights(checkpoint_path)

    ##save
    teacher_model.save(model_save_dir)