# import tensorflow as tf
# import tensorflow
#
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Add, Activation, Dropout, Flatten, Dense,\
                                        Convolution2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from tensorflow.python.keras import backend as K
# from keras.models import Model
# from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense, AveragePooling2D, MaxPooling2D, BatchNormalization
# from keras.layers.convolutional import Convolution2D
channel_axis = -1
def define_keras_model(input_dim, classes=1000):
    '''' Defining this sequentially, because I need the names and shape to match the PyTorch model'''

    # input
    ip = Input(shape=input_dim)

    # conv0 / initial_conv
    x = Convolution2D(64, (7, 7), padding='same', strides=(2,2), kernel_initializer='he_normal',
                      use_bias=False, name='conv0')(ip)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    # max pool
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid')(x)

    # group0 (stride 1, n=3)
    x = group(x,stride=1,n=3, group_num=0)

    # group1 (stride 2, n=4)
    x = group(x, stride=2, n=4, group_num=1)

    # group2 (stride 2, n=6)
    x = group(x, stride=2, n=6, group_num=2)

    # group3 (stride 2, n=3)
    x = group(x, stride=2, n=3, group_num=3)

    # avgpool2d
    x = AveragePooling2D((7, 7),strides=(1,1),padding='valid')(x)
    x = Flatten()(x)

    x = Dense(classes, name='fc')(x)
    x = Activation('softmax')(x)
    model = Model(ip, x)

    return model

def group(input, stride, n, group_num):
    o = input
    for i in range(0, n):
        b_base = ('%s.block%d.conv') % ('group' + str(group_num), i)
        kernel0 = 2**(7+group_num)
        kernel_1 = 2**(7+group_num)
        kernel_2 = 2**(8+group_num)
        kernel_dim = 2**(8+group_num)


        x = o
        # print(x.shape)
        # conv0
        o = Convolution2D(kernel0, (1, 1), padding='same', strides=(1, 1), kernel_initializer='he_normal',
                                 use_bias=False, name=b_base + '0')(x)
        o = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(o)
        o = Activation('relu')(o)
        # conv1
        stride_0 = i == 0 and stride or 1  # lazy copy from pytorch loop
        o = Convolution2D(kernel_1, (3, 3), padding='same', strides=(stride_0, stride_0), kernel_initializer='he_normal',
                                 use_bias=False, name=b_base + '1')(o)
        o = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(o)
        o = Activation('relu')(o)
        # conv2
        o = Convolution2D(kernel_2, (1, 1), padding='same', strides=(1, 1), kernel_initializer='he_normal',
                                 use_bias=False, name=b_base + '2')(o)
        # print(o.shape)
        if i == 0:

            o = Add()([o, Convolution2D(kernel_dim, (1, 1), padding='same', strides=(stride, stride), kernel_initializer='he_normal',
                                      use_bias=False, name=b_base + '_dim')(x)])
            # o += Convolution2D(2 ** (8 + group_num), (1, 1), padding='same', strides=(stride, stride),
            #               kernel_initializer='he_normal',
            #                           use_bias=True, name=b_base + '_dim')(x)
            # print(o)
        else:
            o = Add()([o,x])
            # o += x
        o = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(o)
        o = Activation('relu')(o)
        # print(o.shape)
    return o
