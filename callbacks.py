import os
import tensorflow as tf


class CheckPoint(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, epoch, logs=None):
        checkpoint_filename='Unet_Student_e{0:03d}-acc{1:.4f}-val_acc{2:.4f}-val_loss{3:.4f}.hdf5'.\
            format(epoch, logs['categorical_accuracy'],
                   logs['val_categorical_accuracy'],
                   logs['val_student_loss'])
        checkpoint_filepath = os.path.join(self.checkpoint_dir, checkpoint_filename)
        self.model.student_model.save_weights(checkpoint_filepath)