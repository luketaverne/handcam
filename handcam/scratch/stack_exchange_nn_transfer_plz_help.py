from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import re
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import model_zoo
from nn_transfer import transfer
# from tensorflow.python.keras.models import Model
# from tensorflow.python.keras.layers import Input, Add, Activation, Dropout, Flatten, Dense,\
#                                         Convolution2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.models import Model
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense, AveragePooling2D, MaxPooling2D, Reshape
from keras.layers.convolutional import Convolution2D

# Transfer tutorial <https://github.com/gzuidhof/nn-transfer/blob/master/example.ipynb>
## PyTorch Things
# convert numpy arrays to torch Variables
params = model_zoo.load_url('https://s3.amazonaws.com/modelzoo-networks/wide-resnet-50-2-export-5ae25d50.pth')

for k, v in sorted(params.items()):
    print(k, tuple(v.shape))
    params[k] = Variable(v, requires_grad=True)
print('\nTotal parameters:', sum(v.numel() for v in params.values()))

# PyTorch Model definition, from <https://github.com/szagoruyko/functional-zoo/blob/master/wide-resnet-50-2-export.ipynb>
def define_pytorch_model(params):
    def conv2d(input, params, base, stride=1, pad=0):
        return F.conv2d(input, params[base + '.weight'],
                        params[base + '.bias'], stride, pad)

    def group(input, params, base, stride, n):
        o = input
        for i in range(0, n):
            b_base = ('%s.block%d.conv') % (base, i)
            x = o
            o = conv2d(x, params, b_base + '0')
            o = F.relu(o)
            o = conv2d(o, params, b_base + '1', stride=i == 0 and stride or 1, pad=1)
            o = F.relu(o)
            o = conv2d(o, params, b_base + '2')
            if i == 0:
                o += conv2d(x, params, b_base + '_dim', stride=stride)
            else:
                o += x
            o = F.relu(o)
        return o

    # determine network size by parameters
    blocks = [sum([re.match('group%d.block\d+.conv0.weight' % j, k) is not None
                   for k in params.keys()]) for j in range(4)]

    def f(input, params):
        o = F.conv2d(input, params['conv0.weight'], params['conv0.bias'], 2, 3)
        o = F.relu(o)
        o = F.max_pool2d(o, 3, 2, 1)
        o_g0 = group(o, params, 'group0', 1, blocks[0])
        o_g1 = group(o_g0, params, 'group1', 2, blocks[1])
        o_g2 = group(o_g1, params, 'group2', 2, blocks[2])
        o_g3 = group(o_g2, params, 'group3', 2, blocks[3])
        o = F.avg_pool2d(input=o_g3, kernel_size=7, stride=1, padding=0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params['fc.weight'], params['fc.bias'])
        return o

    return f

## Keras Things
def define_keras_model(input_dim, nb_classes=1000):
    def group(input, stride, n, group_num):
        o = input
        for i in range(0, n):
            b_base = ('%s.block%d.conv') % ('group' + str(group_num), i)
            kernel0 = 2 ** (7 + group_num)
            kernel_1 = 2 ** (7 + group_num)
            kernel_2 = 2 ** (8 + group_num)
            kernel_dim = 2 ** (8 + group_num)

            x = o
            # conv0
            o = Convolution2D(kernel0, (1, 1), padding='same', strides=(1, 1), kernel_initializer='he_normal',
                              use_bias=True, name=b_base + '0', activation='relu')(x)
            # conv1
            stride_0 = i == 0 and stride or 1  # lazy copy from pytorch loop
            o = Convolution2D(kernel_1, (3, 3), padding='same', strides=(stride_0, stride_0),
                              kernel_initializer='he_normal',
                              use_bias=True, name=b_base + '1', activation='relu')(o)
            # conv2
            o = Convolution2D(kernel_2, (1, 1), padding='same', strides=(1, 1), kernel_initializer='he_normal',
                              use_bias=True, name=b_base + '2')(o)
            # print(o.shape)
            if i == 0:

                o = Add()([o, Convolution2D(kernel_dim, (1, 1), padding='same', strides=(stride, stride),
                                            kernel_initializer='he_normal',
                                            use_bias=True, name=b_base + '_dim')(x)])
            else:
                o = Add()([o, x])

            o = Activation('relu')(o)
        return o

    # input
    ip = Input(shape=input_dim)

    # conv0
    x = Convolution2D(64, (7, 7), padding='same', strides=(2,2), kernel_initializer='he_normal',
                      use_bias=True, name='conv0', activation='relu')(ip)
    # x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)

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
    # x = Reshape((2048,))(x)
    x = Dense(nb_classes, name='fc', use_bias=True)(x)
    # x = Activation('linear')(x)
    model = Model(ip, x)

    return model

# Tensorflow
def define_tensorflow_model(inputs, params):
    '''Bottleneck WRN-50-2 model definition
    '''

    def tr(v):
        if v.ndim == 4:
            return v.transpose(2, 3, 1, 0)
        elif v.ndim == 2:
            return v.transpose()
        return v

    params = {k: tf.constant(tr(v)) for k, v in params.items()}

    def conv2d(x, params, name, stride=1, padding=0):
        x = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
        z = tf.nn.conv2d(x, params['%s.weight' % name], [1, stride, stride, 1],
                         padding='VALID')
        if '%s.bias' % name in params:
            return tf.nn.bias_add(z, params['%s.bias' % name])
        else:
            return z

    def group(input, params, base, stride, n):
        o = input
        for i in range(0, n):
            b_base = ('%s.block%d.conv') % (base, i)
            x = o
            o = conv2d(x, params, b_base + '0')
            o = tf.nn.relu(o)
            o = conv2d(o, params, b_base + '1', stride=i == 0 and stride or 1, padding=1)
            o = tf.nn.relu(o)
            o = conv2d(o, params, b_base + '2')
            if i == 0:
                o += conv2d(x, params, b_base + '_dim', stride=stride)
            else:
                o += x
            o = tf.nn.relu(o)
        return o

    # determine network size by parameters
    blocks = [sum([re.match('group%d.block\d+.conv0.weight' % j, k) is not None
                   for k in params.keys()]) for j in range(4)]

    o = conv2d(inputs, params, 'conv0', 2, 3)
    o = tf.nn.relu(o)
    o = tf.pad(o, [[0, 0], [1, 1], [1, 1], [0, 0]])
    o = tf.nn.max_pool(o, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    o_g0 = group(o, params, 'group0', 1, blocks[0])
    o_g1 = group(o_g0, params, 'group1', 2, blocks[1])
    o_g2 = group(o_g1, params, 'group2', 2, blocks[2])
    o_g3 = group(o_g2, params, 'group3', 2, blocks[3])
    o = tf.nn.avg_pool(o_g3, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')
    o = tf.reshape(o, [-1, 2048])
    o = tf.matmul(o, params['fc.weight']) + params['fc.bias']
    return o

# Define the PyTorch model
pytorch_model = define_pytorch_model(params)

# Define the tensorflow model
params_tf = {k: v.numpy() for k, v in model_zoo.load_url('https://s3.amazonaws.com/modelzoo-networks/wide-resnet-50-2-export-5ae25d50.pth').items()}
inputs_tf = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
tf_model = define_tensorflow_model(inputs_tf, params_tf)

# Define the Keras model, reloading the weights since we can't use get_state_dict() on functional pytorch models
input_state_dict_for_keras = model_zoo.load_url('https://s3.amazonaws.com/modelzoo-networks/wide-resnet-50-2-export-5ae25d50.pth')
keras_model = define_keras_model((224,224,3), nb_classes=1000)
# Transfer the pytorch weights to keras
transfer.pytorch_to_keras(pytorch_model, keras_model, state_dict=input_state_dict_for_keras)

# Create dummy data
torch.manual_seed(0)
data = torch.rand(6,3,224,224)
data_keras_and_tf = data.permute(0,2,3,1).numpy()
data_pytorch = Variable(data, requires_grad=False)

# Do a forward pass in all frameworks
pytorch_pred = pytorch_model(data_pytorch, params).data.numpy()
keras_pred = keras_model.predict(data_keras_and_tf)
sess = tf.Session()
tf_pred = sess.run(tf_model, feed_dict={inputs_tf: data_keras_and_tf})

assert keras_pred.shape == pytorch_pred.shape
# check that difference between PyTorch and Tensorflow is small
assert np.abs(tf_pred - pytorch_pred).max() < 1e-4
print(np.abs(keras_pred - pytorch_pred).max())
print(np.abs(keras_pred[...,::-1] - pytorch_pred).max())
print(np.abs(tf_pred - pytorch_pred).max())

plot_comparison = False

if plot_comparison:
    plt.subplot(3, 1, 1)
    plt.axis('Off')
    plt.imshow(pytorch_pred[:,0:40])
    plt.title('pytorch')
    plt.subplot(3, 1, 2)
    plt.axis('Off')
    plt.imshow(tf_pred[:,0:40])
    plt.title('tensorflow')
    plt.subplot(3, 1, 3)
    plt.axis('Off')
    plt.imshow(keras_pred[:,0:40])
    plt.title('keras')
    plt.show()

plot_diff_images = True

if plot_diff_images:
    data_keras_and_tf_flip_ch = keras_model.predict(data_keras_and_tf[...,::-1])
    data_keras_and_tf_flip_lr = keras_model.predict(data_keras_and_tf[:,::-1,:,:])
    data_keras_and_tf_flip_lr_ch = keras_model.predict(data_keras_and_tf[:, ::-1, :, ::-1])
    data_keras_and_tf_flip_ud = keras_model.predict(data_keras_and_tf[:,:,::-1,:])
    data_keras_and_tf_flip_ud_ch = keras_model.predict(data_keras_and_tf[:,:,::-1,::-1])
    data_keras_and_tf_flip_lr_ud = keras_model.predict(data_keras_and_tf[:,::-1,::-1,:])
    data_keras_and_tf_flip_lr_ud_ch = keras_model.predict(data_keras_and_tf[:,::-1,::-1,::-1])
    swap_lr = data_keras_and_tf.transpose(0,2,1,3)
    data_keras_and_tf_swap_lr = keras_model.predict(swap_lr)
    data_keras_and_tf_flip_ch_swap_lr = keras_model.predict(swap_lr[..., ::-1])
    data_keras_and_tf_flip_lr_swap_lr = keras_model.predict(swap_lr[:, ::-1, :, :])
    data_keras_and_tf_flip_lr_ch_swap_lr = keras_model.predict(swap_lr[:, ::-1, :, ::-1])
    data_keras_and_tf_flip_ud_swap_lr = keras_model.predict(swap_lr[:, :, ::-1, :])
    data_keras_and_tf_flip_ud_ch_swap_lr = keras_model.predict(swap_lr[:, :, ::-1, ::-1])
    data_keras_and_tf_flip_lr_ud_swap_lr = keras_model.predict(swap_lr[:, ::-1, ::-1, :])
    data_keras_and_tf_flip_lr_ud_ch_swap_lr = keras_model.predict(swap_lr[:, ::-1, ::-1, ::-1])
    print(np.abs(data_keras_and_tf_flip_ch - pytorch_pred).max())
    print(np.abs(data_keras_and_tf_flip_lr - pytorch_pred).max())
    print(np.abs(data_keras_and_tf_flip_lr_ch - pytorch_pred).max())
    print(np.abs(data_keras_and_tf_flip_ud - pytorch_pred).max())
    print(np.abs(data_keras_and_tf_flip_ud_ch - pytorch_pred).max())
    print(np.abs(data_keras_and_tf_flip_lr_ud - pytorch_pred).max())
    print(np.abs(data_keras_and_tf_flip_lr_ud_ch - pytorch_pred).max())

    print(np.abs(data_keras_and_tf_swap_lr - pytorch_pred).max())
    print(np.abs(data_keras_and_tf_flip_ch_swap_lr - pytorch_pred).max())
    print(np.abs(data_keras_and_tf_flip_lr_swap_lr - pytorch_pred).max())
    print(np.abs(data_keras_and_tf_flip_lr_ch_swap_lr - pytorch_pred).max())
    print(np.abs(data_keras_and_tf_flip_ud_swap_lr - pytorch_pred).max())
    print(np.abs(data_keras_and_tf_flip_ud_ch_swap_lr - pytorch_pred).max())
    print(np.abs(data_keras_and_tf_flip_lr_ud_swap_lr - pytorch_pred).max())
    print(np.abs(data_keras_and_tf_flip_lr_ud_ch_swap_lr - pytorch_pred).max())
    # plt.subplot(3, 1, 1)
    # plt.axis('Off')
    # plt.imshow(pytorch_pred[:, 0:40])
    # plt.title('pytorch')
    # plt.subplot(3, 1, 2)
    # plt.axis('Off')
    # plt.imshow(tf_pred[:, 0:40])
    # plt.title('tensorflow')
    # plt.subplot(3, 1, 3)
    # plt.axis('Off')
    # plt.imshow(keras_pred[:, 0:40])
    # plt.title('keras')
    # plt.show()

assert np.abs(keras_pred - pytorch_pred).max() < 1e-4 # This assertion will fail

print('done!')