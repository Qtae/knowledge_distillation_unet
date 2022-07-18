import tensorflow as tf


def distillation_loss(logits_student, logits_teacher, labels, alpha, temperature):
    normal_loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    normal_loss = normal_loss_func(labels, logits_student)
    softmx_teacher = tf.nn.softmax(logits_teacher / temperature)
    softmx_student = tf.nn.softmax(logits_student / temperature)
    distil_loss_func = tf.keras.losses.KLDivergence()
    distil_loss = distil_loss_func(softmx_teacher, softmx_student)
    total_loss = (1 - alpha) * normal_loss + 2 * alpha * (temperature ** 2) * distil_loss

    return total_loss, normal_loss, distil_loss
