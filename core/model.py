import tensorflow as tf


def vgg16(inputs,
          weights):
    """VGG16 as loss model

    :param inputs: Input tensor (image)
    :param weights: Numpy array
    :return: Dict of features map tensor
    """

    mean = tf.constant(
        [123.68, 116.779, 103.939],
        dtype=tf.float32,
        shape=[1, 1, 1, 3],
        name='imagenet_mean')
    inputs = inputs - mean

    net = {}

    with tf.variable_scope("vgg16", reuse=tf.AUTO_REUSE):
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.get_variable(
                initializer=tf.constant(weights["conv1_1_W"]),
                trainable=False,
                name='conv1_1_W')
            biases = tf.get_variable(
                initializer=tf.constant(weights["conv1_1_b"]),
                trainable=False,
                name='conv1_1_b')
            conv1_1 = tf.nn.conv2d(
                inputs,
                filter=kernel,
                strides=[1, 1, 1, 1],
                padding='SAME')
            conv1_1 = tf.nn.bias_add(conv1_1, biases)
            net["conv1_1"] = conv1_1 = tf.nn.relu(conv1_1, name=scope)

        with tf.name_scope('conv1_2') as scope:
            kernel = tf.get_variable(
                initializer=tf.constant(weights["conv1_2_W"]),
                trainable=False,
                name='conv1_2_W')
            biases = tf.get_variable(
                initializer=tf.constant(weights["conv1_2_b"]),
                trainable=False,
                name='conv1_2_b')
            conv1_2 = tf.nn.conv2d(
                conv1_1,
                filter=kernel,
                strides=[1, 1, 1, 1],
                padding='SAME')
            conv1_2 = tf.nn.bias_add(conv1_2, biases)
            net["conv1_2"] = conv1_2 = tf.nn.relu(conv1_2, name=scope)

        pool1 = tf.nn.avg_pool(
            conv1_2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool1')

        with tf.name_scope('conv2_1') as scope:
            kernel = tf.get_variable(
                initializer=tf.constant(weights["conv2_1_W"]),
                trainable=False,
                name='conv2_1_W')
            biases = tf.get_variable(
                initializer=tf.constant(weights["conv2_1_b"]),
                trainable=False,
                name='conv2_1_b')
            conv2_1 = tf.nn.conv2d(
                pool1,
                filter=kernel,
                strides=[1, 1, 1, 1],
                padding='SAME')
            conv2_1 = tf.nn.bias_add(conv2_1, biases)
            net["conv2_1"] = conv2_1 = tf.nn.relu(conv2_1, name=scope)

        with tf.name_scope('conv2_2') as scope:
            kernel = tf.get_variable(
                initializer=tf.constant(weights["conv2_2_W"]),
                trainable=False,
                name='conv2_2_W')
            biases = tf.get_variable(
                initializer=tf.constant(weights["conv2_2_b"]),
                trainable=False,
                name='conv2_2_b')
            conv2_2 = tf.nn.conv2d(
                conv2_1,
                filter=kernel,
                strides=[1, 1, 1, 1],
                padding='SAME')
            conv2_2 = tf.nn.bias_add(conv2_2, biases)
            net["conv2_2"] = conv2_2 = tf.nn.relu(conv2_2, name=scope)

        pool2 = tf.nn.avg_pool(
            conv2_2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool2')

        with tf.name_scope('conv3_1') as scope:
            kernel = tf.get_variable(
                initializer=tf.constant(weights["conv3_1_W"]),
                trainable=False,
                name='conv3_1_W')
            biases = tf.get_variable(
                initializer=tf.constant(weights["conv3_1_b"]),
                trainable=False,
                name='conv3_1_b')
            conv3_1 = tf.nn.conv2d(
                pool2,
                kernel,
                [1, 1, 1, 1],
                padding='SAME')
            conv3_1 = tf.nn.bias_add(conv3_1, biases)
            net["conv3_1"] = conv3_1 = tf.nn.relu(conv3_1, name=scope)

        with tf.name_scope('conv3_2') as scope:
            kernel = tf.get_variable(
                initializer=tf.constant(weights["conv3_2_W"]),
                trainable=False,
                name='conv3_2_W')
            biases = tf.get_variable(
                initializer=tf.constant(weights["conv3_2_b"]),
                trainable=False,
                name='conv3_2_b')
            conv3_2 = tf.nn.conv2d(
                conv3_1,
                filter=kernel,
                strides=[1, 1, 1, 1],
                padding='SAME')
            conv3_2 = tf.nn.bias_add(conv3_2, biases)
            net["conv3_2"] = conv3_2 = tf.nn.relu(conv3_2, name=scope)

        with tf.name_scope('conv3_3') as scope:
            kernel = tf.get_variable(
                initializer=tf.constant(weights["conv3_3_W"]),
                trainable=False,
                name='conv3_3_W')
            biases = tf.get_variable(
                initializer=tf.constant(weights["conv3_3_b"]),
                trainable=False,
                name='conv3_3_b')
            conv3_3 = tf.nn.conv2d(
                conv3_2,
                filter=kernel,
                strides=[1, 1, 1, 1],
                padding='SAME')
            conv3_3 = tf.nn.bias_add(conv3_3, biases)
            net["conv3_3"] = conv3_3 = tf.nn.relu(conv3_3, name=scope)

        pool3 = tf.nn.avg_pool(
            conv3_3,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool3')

        with tf.name_scope('conv4_1') as scope:
            kernel = tf.get_variable(
                initializer=tf.constant(weights["conv4_1_W"]),
                trainable=False,
                name='conv4_1_W')
            biases = tf.get_variable(
                initializer=tf.constant(weights["conv4_1_b"]),
                trainable=False,
                name='conv4_1_b')
            conv4_1 = tf.nn.conv2d(
                pool3,
                filter=kernel,
                strides=[1, 1, 1, 1],
                padding='SAME')
            conv4_1 = tf.nn.bias_add(conv4_1, biases)
            net["conv4_1"] = conv4_1 = tf.nn.relu(conv4_1, name=scope)

        with tf.name_scope('conv4_2') as scope:
            kernel = tf.get_variable(
                initializer=tf.constant(weights["conv4_2_W"]),
                trainable=False,
                name='conv4_2_W')
            biases = tf.get_variable(
                initializer=tf.constant(weights["conv4_2_b"]),
                trainable=False,
                name='conv4_2_b')
            conv4_2 = tf.nn.conv2d(
                conv4_1,
                filter=kernel,
                strides=[1, 1, 1, 1],
                padding='SAME')
            conv4_2 = tf.nn.bias_add(conv4_2, biases)
            net["conv4_2"] = conv4_2 = tf.nn.relu(conv4_2, name=scope)

        with tf.name_scope('conv4_3') as scope:
            kernel = tf.get_variable(
                initializer=tf.constant(weights["conv4_3_W"]),
                trainable=False,
                name='conv4_3_W')
            biases = tf.get_variable(
                initializer=tf.constant(weights["conv4_3_b"]),
                trainable=False,
                name='conv4_3_b')
            conv4_3 = tf.nn.conv2d(
                conv4_2,
                filter=kernel,
                strides=[1, 1, 1, 1],
                padding='SAME')
            conv4_3 = tf.nn.bias_add(conv4_3, biases)
            net["conv4_3"] = conv4_3 = tf.nn.relu(conv4_3, name=scope)

        pool4 = tf.nn.avg_pool(
            conv4_3,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool4')

        with tf.name_scope('conv5_1') as scope:
            kernel = tf.get_variable(
                initializer=tf.constant(weights["conv5_1_W"]),
                trainable=False,
                name='conv5_1_W')
            biases = tf.get_variable(
                initializer=tf.constant(weights["conv5_1_b"]),
                trainable=False,
                name='conv5_1_b')
            conv5_1 = tf.nn.conv2d(
                pool4,
                filter=kernel,
                strides=[1, 1, 1, 1],
                padding='SAME')
            conv5_1 = tf.nn.bias_add(conv5_1, biases)
            net["conv5_1"] = conv5_1 = tf.nn.relu(conv5_1, name=scope)

        with tf.name_scope('conv5_2') as scope:
            kernel = tf.get_variable(
                initializer=tf.constant(weights["conv5_2_W"]),
                trainable=False,
                name='conv5_2_W')
            biases = tf.get_variable(
                initializer=tf.constant(weights["conv5_2_b"]),
                trainable=False,
                name='conv5_2_b')
            conv5_2 = tf.nn.conv2d(
                conv5_1,
                filter=kernel,
                strides=[1, 1, 1, 1],
                padding='SAME')
            conv5_2 = tf.nn.bias_add(conv5_2, biases)
            net["conv5_2"] = conv5_2 = tf.nn.relu(conv5_2, name=scope)

        with tf.name_scope('conv5_3') as scope:
            kernel = tf.get_variable(
                initializer=tf.constant(weights["conv5_3_W"]),
                trainable=False,
                name='conv5_3_W')
            biases = tf.get_variable(
                initializer=tf.constant(weights["conv5_3_b"]),
                trainable=False,
                name='conv5_3_b')
            conv5_3 = tf.nn.conv2d(
                conv5_2,
                filter=kernel,
                strides=[1, 1, 1, 1],
                padding='SAME')
            conv5_3 = tf.nn.bias_add(conv5_3, biases)
            net["conv5_3"] = tf.nn.relu(conv5_3, name=scope)

    return net
