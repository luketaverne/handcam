import tensorflow as tf
import tensorflow

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (
    Input,
    concatenate,
    ZeroPadding2D,
    Lambda,
    Dense,
    Dropout,
    Activation,
    Convolution2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
    MaxPooling2D,
    BatchNormalization,
)

from handcam.ltt.network.model.DenseNet.custom_layers import Scale


def DenseNet(
    nb_dense_block=4,
    growth_rate=48,
    nb_filter=96,
    reduction=0.0,
    dropout_rate=0.0,
    weight_decay=1e-4,
    classes=1000,
    weights_path=None,
):
    """Instantiate the DenseNet 161 architecture,
    # Arguments
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters
        reduction: reduction factor of transition blocks.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        classes: optional number of classes to classify images
        weights_path: path to pre-trained weights
    # Returns
        A Keras model instance.
    """
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis

    concat_axis = 3
    img_input = Input(
        shape=(224, 224, 4), name="data"
    )  # BGR-D. Treat the BGR and depth with separate pipelines.

    bgr_img_input = crop(dimension=3, start=0, end=3)(img_input)
    depth_img_input = crop(dimension=3, start=3, end=4)(img_input)

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 96
    nb_layers = [6, 12, 36, 24]  # For DenseNet-161

    """
    Begin BGR treatment
    """
    # Initial convolution
    x = ZeroPadding2D((3, 3), name="conv1_zeropadding")(bgr_img_input)
    x = Convolution2D(nb_filter, 7, strides=2, name="conv1", use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name="conv1_bn")(x)
    x = Scale(axis=concat_axis, name="conv1_scale")(x)
    x = Activation("relu", name="relu1")(x)
    x = ZeroPadding2D((1, 1), name="pool1_zeropadding")(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name="pool1")(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx + 2
        x, nb_filter = dense_block(
            x,
            stage,
            nb_layers[block_idx],
            nb_filter,
            growth_rate,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
        )

        # Add transition_block
        x = transition_block(
            x,
            stage,
            nb_filter,
            compression=compression,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
        )
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(
        x,
        final_stage,
        nb_layers[-1],
        nb_filter,
        growth_rate,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
    )

    x = BatchNormalization(
        epsilon=eps, axis=concat_axis, name="conv" + str(final_stage) + "_blk_bn"
    )(x)
    x = Scale(axis=concat_axis, name="conv" + str(final_stage) + "_blk_scale")(x)
    x = Activation("relu", name="relu" + str(final_stage) + "_blk")(x)
    x = GlobalAveragePooling2D(name="pool" + str(final_stage))(x)

    """
    Begin depth treatment
    """
    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 96
    nb_layers = [6, 12, 36, 24]  # For DenseNet-161

    # Initial convolution
    y = ZeroPadding2D((3, 3), name="conv1_depth_zeropadding")(depth_img_input)
    y = Convolution2D(nb_filter, 7, strides=2, name="conv1_depth", use_bias=False)(y)
    y = BatchNormalization(epsilon=eps, axis=concat_axis, name="conv1_depth_bn")(y)
    y = Scale(axis=concat_axis, name="conv1_depth_scale")(y)
    y = Activation("relu", name="relu1_depth")(y)
    y = ZeroPadding2D((1, 1), name="pool1_depth_zeropadding")(y)
    y = MaxPooling2D((3, 3), strides=(2, 2), name="pool1_depth")(y)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx + 2
        y, nb_filter = dense_block(
            y,
            stage,
            nb_layers[block_idx],
            nb_filter,
            growth_rate,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            depth_block=True,
        )

        # Add transition_block
        y = transition_block(
            y,
            stage,
            nb_filter,
            compression=compression,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            depth_block=True,
        )
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    y, nb_filter = dense_block(
        y,
        final_stage,
        nb_layers[-1],
        nb_filter,
        growth_rate,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
        depth_block=True,
    )

    y = BatchNormalization(
        epsilon=eps,
        axis=concat_axis,
        name="conv" + "_depth_" + str(final_stage) + "_blk_bn",
    )(y)
    y = Scale(
        axis=concat_axis, name="conv" + "_depth_" + str(final_stage) + "_blk_scale"
    )(y)
    y = Activation("relu", name="relu" + "_depth_" + str(final_stage) + "_blk")(y)
    y = GlobalAveragePooling2D(name="pool" + "_depth_" + str(final_stage))(y)

    """
    Combine BGR and depth before the fully connected layers.
    """
    x = concatenate([x, y])

    # x = Dense(1024, name='fc1')(x) # Don't use name fc6, because we're loading weights by name.
    x = Dense(classes, name="fc1")(x)
    x = Activation("softmax", name="prob_out")(x)

    model = Model(img_input, x, name="densenet")

    if weights_path is not None:
        model.load_weights(weights_path, by_name=True)

    return model


def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start:end]
        if dimension == 1:
            return x[:, start:end]
        if dimension == 2:
            return x[:, :, start:end]
        if dimension == 3:
            return x[:, :, :, start:end]
        if dimension == 4:
            return x[:, :, :, :, start:end]

    return Lambda(func)


def conv_block(
    x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4, depth_block=False
):
    """Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
    # Arguments
        x: input tensor
        stage: index for dense block
        branch: layer index within each dense block
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    """
    eps = 1.1e-5
    conv_name_base = "conv" + str(stage) + "_" + str(branch)
    relu_name_base = "relu" + str(stage) + "_" + str(branch)

    if depth_block:
        conv_name_base = "conv_depth_" + str(stage) + "_" + str(branch)
        relu_name_base = "relu_depth_" + str(stage) + "_" + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(
        epsilon=eps, axis=concat_axis, name=conv_name_base + "_x1_bn"
    )(x)
    x = Scale(axis=concat_axis, name=conv_name_base + "_x1_scale")(x)
    x = Activation("relu", name=relu_name_base + "_x1")(x)
    x = Convolution2D(inter_channel, 1, name=conv_name_base + "_x1", use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(
        epsilon=eps, axis=concat_axis, name=conv_name_base + "_x2_bn"
    )(x)
    x = Scale(axis=concat_axis, name=conv_name_base + "_x2_scale")(x)
    x = Activation("relu", name=relu_name_base + "_x2")(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base + "_x2_zeropadding")(x)
    x = Convolution2D(nb_filter, 3, name=conv_name_base + "_x2", use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(
    x,
    stage,
    nb_filter,
    compression=1.0,
    dropout_rate=None,
    weight_decay=1e-4,
    depth_block=False,
):
    """Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
    # Arguments
        x: input tensor
        stage: index for dense block
        nb_filter: number of filters
        compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    """

    eps = 1.1e-5
    conv_name_base = "conv" + str(stage) + "_blk"
    relu_name_base = "relu" + str(stage) + "_blk"
    pool_name_base = "pool" + str(stage)

    if depth_block:
        conv_name_base = "conv_depth_" + str(stage) + "_blk"
        relu_name_base = "relu_depth_" + str(stage) + "_blk"
        pool_name_base = "pool_depth_" + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + "_bn")(
        x
    )
    x = Scale(axis=concat_axis, name=conv_name_base + "_scale")(x)
    x = Activation("relu", name=relu_name_base)(x)
    x = Convolution2D(
        int(nb_filter * compression), 1, name=conv_name_base, use_bias=False
    )(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(
    x,
    stage,
    nb_layers,
    nb_filter,
    growth_rate,
    dropout_rate=None,
    weight_decay=1e-4,
    grow_nb_filters=True,
    depth_block=False,
):
    """Build a dense_block where the output of each conv_block is fed to subsequent ones
    # Arguments
        x: input tensor
        stage: index for dense block
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: flag to decide to allow number of filters to grow
    """

    eps = 1.1e-5
    concat_feat = x
    concat_name_base = "concat_"
    if depth_block:
        concat_name_base = "concat_depth_"

    for i in range(nb_layers):
        branch = i + 1
        x = conv_block(
            concat_feat,
            stage,
            branch,
            growth_rate,
            dropout_rate,
            weight_decay,
            depth_block=depth_block,
        )
        concat_feat = concatenate(
            [concat_feat, x],
            axis=concat_axis,
            name=concat_name_base + str(stage) + "_" + str(branch),
        )

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter
