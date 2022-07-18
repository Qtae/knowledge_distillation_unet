import numpy as np
import tensorflow as tf


class DistillationModel(tf.keras.Model):
    def __init__(self, teacher_model, student_model):
        super(DistillationModel, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model

    def compile(self, optimizer, metrics, loss, alpha=0.1, temparature=3):
        super(DistillationModel, self).__init__()
        self.loss = loss
        self.alpha = alpha
        self.temparature = temparature

    def train_step(self, data):
        x, y = data

        # Forward pass of teacher
        logit_teacher = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            logit_student = self.student(x, training=True)

            student_loss = self.student_loss_fn(y, student_predictions)

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            distillation_loss = (
                    self.distillation_loss_fn(
                        tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                        tf.nn.softmax(student_predictions / self.temperature, axis=1),
                    )
                    * self.temperature ** 2
            )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results


def unet(input_layer, classes, init_depth=16):
    conv1 = tf.keras.layers.Conv2D(init_depth, 3, padding='same', activation='relu', kernel_initializer='he_normal')(input_layer)
    conv1 = tf.keras.layers.Conv2D(init_depth, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv1)
    conv2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv2)
    conv2 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv2)
    conv3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(init_depth*4, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv3)
    conv3 = tf.keras.layers.Conv2D(init_depth*4, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv3)
    conv4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(init_depth*8, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv4)
    conv4 = tf.keras.layers.Conv2D(init_depth*8, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv4)
    conv5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv4)

    conv5 = tf.keras.layers.Conv2D(init_depth*16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv5)
    conv5 = tf.keras.layers.Conv2D(init_depth*16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv5)
    up1 = tf.keras.layers.Conv2DTranspose(init_depth*16, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv5)
    concat1 = tf.keras.layers.concatenate([conv4, up1], axis=3)

    conv6 = tf.keras.layers.Conv2D(init_depth*8, 3, padding='same', activation='relu', kernel_initializer='he_normal')(concat1)
    conv6 = tf.keras.layers.Conv2D(init_depth*8, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv6)
    up2 = tf.keras.layers.Conv2DTranspose(init_depth*8, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv6)
    concat2 = tf.keras.layers.concatenate([conv3, up2], axis=3)

    conv7 = tf.keras.layers.Conv2D(init_depth*4, 3, padding='same', activation='relu', kernel_initializer='he_normal')(concat2)
    conv7 = tf.keras.layers.Conv2D(init_depth*4, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv7)
    up3 = tf.keras.layers.Conv2DTranspose(init_depth*4, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv7)
    concat3 = tf.keras.layers.concatenate([conv2, up3], axis=3)

    conv8 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(concat3)
    conv8 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv8)
    up4 = tf.keras.layers.Conv2DTranspose(init_depth*2, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv8)
    concat4 = tf.keras.layers.concatenate([conv1, up4], axis=3)

    conv9 = tf.keras.layers.Conv2D(init_depth, 3, padding='same', activation='relu', kernel_initializer='he_normal')(concat4)
    conv9 = tf.keras.layers.Conv2D(init_depth, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv9)

    output_conv = tf.keras.layers.Conv2D(classes, 1, padding='same', kernel_initializer='he_normal')(conv9)

    output_sftmx = tf.keras.activations.softmax(output_conv)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_sftmx)
    return model


def unet_logit(input_layer, classes, init_depth=16):
    conv1 = tf.keras.layers.Conv2D(init_depth, 3, padding='same', activation='relu', kernel_initializer='he_normal')(input_layer)
    conv1 = tf.keras.layers.Conv2D(init_depth, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv1)
    conv2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv2)
    conv2 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv2)
    conv3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(init_depth*4, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv3)
    conv3 = tf.keras.layers.Conv2D(init_depth*4, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv3)
    conv4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(init_depth*8, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv4)
    conv4 = tf.keras.layers.Conv2D(init_depth*8, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv4)
    conv5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv4)

    conv5 = tf.keras.layers.Conv2D(init_depth*16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv5)
    conv5 = tf.keras.layers.Conv2D(init_depth*16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv5)
    up1 = tf.keras.layers.Conv2DTranspose(init_depth*16, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv5)
    concat1 = tf.keras.layers.concatenate([conv4, up1], axis=3)

    conv6 = tf.keras.layers.Conv2D(init_depth*8, 3, padding='same', activation='relu', kernel_initializer='he_normal')(concat1)
    conv6 = tf.keras.layers.Conv2D(init_depth*8, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv6)
    up2 = tf.keras.layers.Conv2DTranspose(init_depth*8, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv6)
    concat2 = tf.keras.layers.concatenate([conv3, up2], axis=3)

    conv7 = tf.keras.layers.Conv2D(init_depth*4, 3, padding='same', activation='relu', kernel_initializer='he_normal')(concat2)
    conv7 = tf.keras.layers.Conv2D(init_depth*4, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv7)
    up3 = tf.keras.layers.Conv2DTranspose(init_depth*4, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv7)
    concat3 = tf.keras.layers.concatenate([conv2, up3], axis=3)

    conv8 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(concat3)
    conv8 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv8)
    up4 = tf.keras.layers.Conv2DTranspose(init_depth*2, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv8)
    concat4 = tf.keras.layers.concatenate([conv1, up4], axis=3)

    conv9 = tf.keras.layers.Conv2D(init_depth, 3, padding='same', activation='relu', kernel_initializer='he_normal')(concat4)
    conv9 = tf.keras.layers.Conv2D(init_depth, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv9)

    output_conv = tf.keras.layers.Conv2D(classes, 1, padding='same', kernel_initializer='he_normal')(conv9)

    output_sftmx = tf.keras.activations.softmax(output_conv)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_sftmx)
    return model

if __name__=="__main__":
    print("unet model test")
    input_layer = tf.keras.layers.Input([640, 640, 1])
    model = unet(input_layer, classes=2)