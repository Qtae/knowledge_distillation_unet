import os
import tensorflow as tf


class DistillationModel(tf.keras.Model):
    def __init__(self, teacher_model, student_model):
        super(DistillationModel, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model

    def compile(self, optimizer, metrics, loss,
                alpha=0.1, temperature=3):
        super(DistillationModel, self).compile(optimizer=optimizer, metrics=metrics)
        #self.student_model.compile(optimizer=optimizer, metrics=metrics, loss='binary_crossentropy')
        self.compiled_metrics
        self.loss = loss
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data

        logits_teacher = self.teacher_model(x, training=False)

        with tf.GradientTape() as tape:
            logits_student = self.student_model(x, training=True)
            normal_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y, logits_student)
            distil_loss = tf.keras.losses.KLDivergence()(tf.nn.softmax(logits_teacher / self.temperature),
                                                         tf.nn.softmax(logits_student / self.temperature))
            total_loss = (1 - self.alpha) * normal_loss + self.alpha * (self.temperature**2) * distil_loss

        trainable_vars = self.student_model.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, tf.nn.softmax(logits_student))

        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": normal_loss,
                        "distil_loss": distil_loss,
                        "total_loss": total_loss})
        return results

    def test_step(self, data):
        x, y = data

        logits_student = self.student_model(x, training=False)

        student_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y, logits_student)

        self.compiled_metrics.update_state(y, tf.nn.softmax(logits_student))

        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
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
    #up1 = tf.keras.layers.Conv2DTranspose(init_depth*16, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv5)
    up1 = tf.keras.layers.Conv2D(init_depth*16, 3, padding='same', activation='relu', kernel_initializer='he_normal')((tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)))
    concat1 = tf.keras.layers.concatenate([conv4, up1], axis=3)

    conv6 = tf.keras.layers.Conv2D(init_depth*8, 3, padding='same', activation='relu', kernel_initializer='he_normal')(concat1)
    conv6 = tf.keras.layers.Conv2D(init_depth*8, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv6)
    #up2 = tf.keras.layers.Conv2DTranspose(init_depth*8, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv6)
    up2 = tf.keras.layers.Conv2D(init_depth*8, 3, padding='same', activation='relu', kernel_initializer='he_normal')((tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)))
    concat2 = tf.keras.layers.concatenate([conv3, up2], axis=3)

    conv7 = tf.keras.layers.Conv2D(init_depth*4, 3, padding='same', activation='relu', kernel_initializer='he_normal')(concat2)
    conv7 = tf.keras.layers.Conv2D(init_depth*4, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv7)
    #up3 = tf.keras.layers.Conv2DTranspose(init_depth*4, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv7)
    up3 = tf.keras.layers.Conv2D(init_depth*4, 3, padding='same', activation='relu', kernel_initializer='he_normal')((tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)))
    concat3 = tf.keras.layers.concatenate([conv2, up3], axis=3)

    conv8 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(concat3)
    conv8 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv8)
    #up4 = tf.keras.layers.Conv2DTranspose(init_depth*2, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv8)
    up4 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')((tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)))
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
    # up1 = tf.keras.layers.Conv2DTranspose(init_depth*16, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv5)
    up1 = tf.keras.layers.Conv2D(init_depth * 16, 3, padding='same', activation='relu', kernel_initializer='he_normal')((tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)))
    concat1 = tf.keras.layers.concatenate([conv4, up1], axis=3)

    conv6 = tf.keras.layers.Conv2D(init_depth * 8, 3, padding='same', activation='relu', kernel_initializer='he_normal')(concat1)
    conv6 = tf.keras.layers.Conv2D(init_depth * 8, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv6)
    # up2 = tf.keras.layers.Conv2DTranspose(init_depth*8, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv6)
    up2 = tf.keras.layers.Conv2D(init_depth * 8, 3, padding='same', activation='relu', kernel_initializer='he_normal')((tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)))
    concat2 = tf.keras.layers.concatenate([conv3, up2], axis=3)

    conv7 = tf.keras.layers.Conv2D(init_depth * 4, 3, padding='same', activation='relu', kernel_initializer='he_normal')(concat2)
    conv7 = tf.keras.layers.Conv2D(init_depth * 4, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv7)
    # up3 = tf.keras.layers.Conv2DTranspose(init_depth*4, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv7)
    up3 = tf.keras.layers.Conv2D(init_depth * 4, 3, padding='same', activation='relu', kernel_initializer='he_normal')((tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)))
    concat3 = tf.keras.layers.concatenate([conv2, up3], axis=3)

    conv8 = tf.keras.layers.Conv2D(init_depth * 2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(concat3)
    conv8 = tf.keras.layers.Conv2D(init_depth * 2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv8)
    # up4 = tf.keras.layers.Conv2DTranspose(init_depth*2, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv8)
    up4 = tf.keras.layers.Conv2D(init_depth * 2, 3, padding='same', activation='relu', kernel_initializer='he_normal')((tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)))
    concat4 = tf.keras.layers.concatenate([conv1, up4], axis=3)

    conv9 = tf.keras.layers.Conv2D(init_depth, 3, padding='same', activation='relu', kernel_initializer='he_normal')(concat4)
    conv9 = tf.keras.layers.Conv2D(init_depth, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv9)

    output_conv = tf.keras.layers.Conv2D(classes, 1, padding='same', kernel_initializer='he_normal')(conv9)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_conv)
    return model


def unet_reduced(input_layer, classes, init_depth=16):
    conv1 = tf.keras.layers.Conv2D(init_depth, 3, padding='same', activation='relu', kernel_initializer='he_normal')(input_layer)
    conv1 = tf.keras.layers.Conv2D(init_depth, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv1)
    conv2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv2)
    conv2 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv2)
    conv3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv3)
    conv3 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv3)
    conv4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv4)
    conv4 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv4)
    conv5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv4)

    conv5 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv5)
    conv5 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv5)
    #up1 = tf.keras.layers.Conv2DTranspose(init_depth*16, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv5)
    up1 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')((tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)))
    concat1 = tf.keras.layers.concatenate([conv4, up1], axis=3)

    conv6 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(concat1)
    conv6 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv6)
    #up2 = tf.keras.layers.Conv2DTranspose(init_depth*8, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv6)
    up2 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')((tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)))
    concat2 = tf.keras.layers.concatenate([conv3, up2], axis=3)

    conv7 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(concat2)
    conv7 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv7)
    #up3 = tf.keras.layers.Conv2DTranspose(init_depth*4, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv7)
    up3 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')((tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)))
    concat3 = tf.keras.layers.concatenate([conv2, up3], axis=3)

    conv8 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(concat3)
    conv8 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv8)
    #up4 = tf.keras.layers.Conv2DTranspose(init_depth*2, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv8)
    up4 = tf.keras.layers.Conv2D(init_depth*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')((tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)))
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