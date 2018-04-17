import tensorflow as tf


def gram_matrix(x):
    """

    :param x: Input tensor
    :return: Gram matrix (self inner product of x_reshape)
    """

    batch_size, w, h, ch = x.shape.as_list()
    x = tf.reshape(x, [batch_size, w * h, ch])
    return tf.matmul(x, x, transpose_a=True)


def loss_fun(target_content_features,
             target_style_features,
             transferred_content_features,
             transferred_style_features,
             content_loss_weight,
             style_loss_weight):
    """

    :param target_content_features: List of target content features tensor
    :param target_style_features: List of target style features tensor
    :param transferred_content_features: List of transferred content features tensor
    :param transferred_style_features: List of transferred style features tensor
    :param content_loss_weight: Content loss
    :param style_loss_weight: Style loss
    :return: Total loss
    """

    assert len(target_content_features) == len(transferred_content_features)
    assert len(target_style_features) == len(transferred_style_features)

    content_loss = 0
    for i in range(len(transferred_content_features)):
        content_loss = content_loss \
            + 2 * tf.nn.l2_loss(target_content_features[i] - transferred_content_features[i])

    style_loss = 0
    for i in range(len(transferred_style_features)):
        _, w, h, ch = target_style_features[i].shape.as_list()
        gram_target = gram_matrix(target_style_features[i])
        gram_transferred = gram_matrix(transferred_style_features[i])
        style_loss = style_loss \
            + (2 * tf.nn.l2_loss(gram_target - gram_transferred)) / (4 * ch ** 2 * (w * h) ** 2)

    return content_loss_weight * content_loss + style_loss_weight * style_loss
