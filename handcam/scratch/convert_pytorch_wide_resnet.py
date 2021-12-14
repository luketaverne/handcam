from __future__ import print_function

import tensorflow as tf
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import re
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torchviz import make_dot
from torch.utils import model_zoo
from nn_transfer import transfer, util
from tensorflow.python.keras.layers import Input

from handcam.ltt.network.model.Wide_ResNet import wide_residual_network as wrn
from handcam.ltt.network.model.Wide_ResNet import wrn_keras_luke as wrn_luke

# Transfer tutorial <https://github.com/gzuidhof/nn-transfer/blob/master/example.ipynb>
# convert numpy arrays to torch Variables
params = model_zoo.load_url(
    "https://s3.amazonaws.com/modelzoo-networks/wide-resnet-50-2-export-5ae25d50.pth",
    model_dir="/local/home/luke/",
)
pytorch_layer_names = []
input_state_dict = model_zoo.load_url(
    "https://s3.amazonaws.com/modelzoo-networks/wide-resnet-50-2-export-5ae25d50.pth",
    model_dir="/local/home/luke/",
)

# print(input_state_dict)

for k, v in sorted(params.items()):
    layer_split = k.split(".")
    if layer_split[-1] == "weight":
        pytorch_layer_names.append(".".join(layer_split[0 : len(layer_split) - 1]))
    print(k, tuple(v.shape))
    params[k] = Variable(v, requires_grad=True)
print(pytorch_layer_names)
print("\nTotal parameters:", sum(v.numel() for v in params.values()))
# PyTorch Model definition, from <https://github.com/szagoruyko/functional-zoo/blob/master/wide-resnet-50-2-export.ipynb>
def define_model(params):
    def conv2d(input, params, base, stride=1, pad=0):
        return F.conv2d(
            input, params[base + ".weight"], params[base + ".bias"], stride, pad
        )

    def group(input, params, base, stride, n):
        o = input
        for i in range(0, n):
            b_base = ("%s.block%d.conv") % (base, i)
            x = o
            o = conv2d(x, params, b_base + "0")
            o = F.relu(o)
            o = conv2d(o, params, b_base + "1", stride=i == 0 and stride or 1, pad=1)
            o = F.relu(o)
            o = conv2d(o, params, b_base + "2")
            if i == 0:
                o += conv2d(x, params, b_base + "_dim", stride=stride)
            else:
                o += x
            o = F.relu(o)
        return o

    # determine network size by parameters
    blocks = [
        sum(
            [
                re.match("group%d.block\d+.conv0.weight" % j, k) is not None
                for k in params.keys()
            ]
        )
        for j in range(4)
    ]

    def f(input, params):
        o = F.conv2d(input, params["conv0.weight"], params["conv0.bias"], 2, 3)
        o = F.relu(o)
        o = F.max_pool2d(o, 3, 2, 1)
        o_g0 = group(o, params, "group0", 1, blocks[0])
        o_g1 = group(o_g0, params, "group1", 2, blocks[1])
        o_g2 = group(o_g1, params, "group2", 2, blocks[2])
        o_g3 = group(o_g2, params, "group3", 2, blocks[3])
        o = F.avg_pool2d(input=o_g3, kernel_size=7, stride=1, padding=0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params["fc.weight"], params["fc.bias"])
        return o

    return f


class WideRNPyTorch(nn.Module):
    def __init__(self, params):
        super(WideRNPyTorch, self).__init__()
        self.conv0 = nn.Conv2d(64, 128, 3)

    def define_model(params):
        def conv2d(input, params, base, stride=1, pad=0):
            return F.conv2d(
                input, params[base + ".weight"], params[base + ".bias"], stride, pad
            )

        def group(input, params, base, stride, n):
            o = input
            for i in range(0, n):
                b_base = ("%s.block%d.conv") % (base, i)
                x = o
                o = conv2d(x, params, b_base + "0")
                o = F.relu(o)
                o = conv2d(
                    o, params, b_base + "1", stride=i == 0 and stride or 1, pad=1
                )
                o = F.relu(o)
                o = conv2d(o, params, b_base + "2")
                if i == 0:
                    o += conv2d(x, params, b_base + "_dim", stride=stride)
                else:
                    o += x
                o = F.relu(o)
            return o

        # determine network size by parameters
        blocks = [
            sum(
                [
                    re.match("group%d.block\d+.conv0.weight" % j, k) is not None
                    for k in params.keys()
                ]
            )
            for j in range(4)
        ]

        def forward(input, params):
            o = F.conv2d(input, params["conv0.weight"], params["conv0.bias"], 2, 3)
            o = F.relu(o)
            o = F.max_pool2d(o, 3, 2, 1)
            o_g0 = group(o, params, "group0", 1, blocks[0])
            o_g1 = group(o_g0, params, "group1", 2, blocks[1])
            o_g2 = group(o_g1, params, "group2", 2, blocks[2])
            o_g3 = group(o_g2, params, "group3", 2, blocks[3])
            o = F.avg_pool2d(input=o_g3, kernel_size=7, stride=1, padding=0)
            o = o.view(o.size(0), -1)
            o = F.linear(o, params["fc.weight"], params["fc.bias"])
            return o


# Use pytorch like this
pytorch_model = define_model(params)

# inputs = torch.randn(1,3,224,224)
#
# py2 = WideRNPyTorch(params)
# for key, value in py2.state_dict().items():
#     print(key)
#     print(value)
# print(py2)
# print(params['group0.block0.conv0.weight'].shape)
# state_dict = pytorch_model.state_dict()
# print(pytorch_model((640,480,3),params))

# Create dummy data
# TODO: define keras version of this model
ip = (224, 224, 3)
# wrn_28_10 = wrn.create_wide_residual_network(ip, nb_classes=1000, N=6, k=2, dropout=0.0, verbose=1)
keras_network = wrn_luke.define_keras_model(ip, nb_classes=1000)

# Try with tensorflow
def g(inputs, params):
    """Bottleneck WRN-50-2 model definition"""

    def tr(v):
        if v.ndim == 4:
            return v.transpose(2, 3, 1, 0)
        elif v.ndim == 2:
            return v.transpose()
        return v

    params = {k: tf.constant(tr(v)) for k, v in params.items()}

    def conv2d(x, params, name, stride=1, padding=0):
        x = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
        z = tf.nn.conv2d(
            x, params["%s.weight" % name], [1, stride, stride, 1], padding="VALID"
        )
        if "%s.bias" % name in params:
            return tf.nn.bias_add(z, params["%s.bias" % name])
        else:
            return z

    def group(input, params, base, stride, n):
        o = input
        for i in range(0, n):
            b_base = ("%s.block%d.conv") % (base, i)
            x = o
            o = conv2d(x, params, b_base + "0")
            o = tf.nn.relu(o)
            o = conv2d(
                o, params, b_base + "1", stride=i == 0 and stride or 1, padding=1
            )
            o = tf.nn.relu(o)
            o = conv2d(o, params, b_base + "2")
            if i == 0:
                o += conv2d(x, params, b_base + "_dim", stride=stride)
            else:
                o += x
            o = tf.nn.relu(o)
        return o

    # determine network size by parameters
    blocks = [
        sum(
            [
                re.match("group%d.block\d+.conv0.weight" % j, k) is not None
                for k in params.keys()
            ]
        )
        for j in range(4)
    ]

    o = conv2d(inputs, params, "conv0", 2, 3)
    o = tf.nn.relu(o)
    o = tf.pad(o, [[0, 0], [1, 1], [1, 1], [0, 0]])
    o = tf.nn.max_pool(o, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
    o_g0 = group(o, params, "group0", 1, blocks[0])  # 2
    o_g1 = group(o_g0, params, "group1", 2, blocks[1])  # 3
    o_g2 = group(o_g1, params, "group2", 2, blocks[2])  # 5
    o_g3 = group(o_g2, params, "group3", 2, blocks[3])  # 2
    o = tf.nn.avg_pool(o_g3, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding="VALID")
    o = tf.reshape(o, [-1, 2048])
    o = tf.matmul(o, params["fc.weight"]) + params["fc.bias"]
    return o


params_tf = {
    k: v.numpy()
    for k, v in torch.load(
        "/local/home/luke/wide-resnet-50-2-export-5ae25d50.pth"
    ).items()
}
params_newer_try = {
    k: Variable(v, requires_grad=True)
    for k, v in torch.load(
        "/local/home/luke/wide-resnet-50-2-export-5ae25d50.pth"
    ).items()
}
inputs_tf = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])

tf_model = g(inputs_tf, params_tf)

print(keras_network.summary())

# wrn_28_10.load_weights("/local/home/luke/wide-resnet-50-2-export-5ae25d50.pth")

transfer.pytorch_to_keras(
    pytorch_model,
    keras_network,
    state_dict=input_state_dict,
    flip_filters=False,
    flip_channels=True,
)
# transfer.pytorch_to_keras(pytorch_model, keras_network, state_dict=input_state_dict)
# Create dummy data
torch.manual_seed(0)
data = torch.rand(6, 3, 224, 224)
data_keras = data.permute(0, 2, 3, 1).numpy()
# data_keras = data_keras[...,::-1]
# data_keras = np.swapaxes(data_keras, )
data_pytorch = Variable(data, requires_grad=False)

# Do a forward pass in both frameworks
keras_pred = keras_network.predict(data_keras)
pytorch_pred = pytorch_model(data_pytorch, params).data.numpy()
sess = tf.Session()
tf_pred = sess.run(tf_model, feed_dict={inputs_tf: data_keras})
assert keras_pred.shape == pytorch_pred.shape
# check that difference between PyTorch and Tensorflow is small
# assert np.abs(tf_pred - pytorch_pred).max() < 1e-5
print(np.abs(keras_pred - pytorch_pred).max())
print(np.abs(tf_pred - pytorch_pred).max())
# assert np.abs(keras_pred - pytorch_pred).max() < 1e-5
print(keras_pred.shape)
plt.subplot(3, 1, 1)
plt.axis("Off")
plt.imshow(keras_pred[:, 0:40])
plt.subplot(3, 1, 2)
plt.axis("Off")
plt.imshow(pytorch_pred[:, 0:40])
plt.subplot(3, 1, 3)
plt.axis("Off")
plt.imshow(tf_pred[:, 0:40])
plt.show()

print("done!")

keras_network.save_weights("/local/home/luke/wrn-50-2-keras.h5")
