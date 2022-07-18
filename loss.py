import tensorflow as tf


def distillation_loss(logits_student, logits_teacher, labels, alpha, temparature):
    normal_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits_student)
    logits_pt = logits_student / temparature
    labels_pt = logits_student / temparature
    teacher_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_pt, logits=logits_pt)

    total_loss = (1 - alpha) * normal_loss + 2 * alpha * (temparature ** 2) * teacher_loss

    return total_loss
