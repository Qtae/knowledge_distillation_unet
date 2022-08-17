import os
import tensorflow as tf
from datetime import datetime
from model import unet_logit, unet

if __name__ == '__main__':
    root_dir = 'D:/Public/qtkim/Knowledge_Distillation/'
    print('==================teacher model training==================')
    start_time = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    classes = 2
    learning_rate = 0.001
    input_layer = tf.keras.layers.Input([640, 640, 1])
    model_save_dir = root_dir + 'model_student/model(' + start_time + ')_e113'
    checkpoint_path = root_dir + 'model_student/checkpoints/Unet_Student_e073-acc0.9938-val_acc0.9826-val_loss1.3583.hdf5'

    ##build model
    print('-----------------------build model------------------------')
    teacher_model = unet_logit(input_layer, classes=classes, init_depth=16)
    teacher_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    teacher_model.load_weights(checkpoint_path)

    ##save
    teacher_model.save(model_save_dir)