"""STSnet model"""

from __future__ import division, print_function, absolute_import
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import (
    Conv3D,
    MaxPooling3D,
    AveragePooling3D,
    GlobalAveragePooling2D,
)
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, Reshape
from tensorflow.keras.layers import SpatialDropout2D, TimeDistributed
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Flatten, Permute, Dropout
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf


from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Input,
    Activation,
    Conv2D,
    SeparableConv2D,
    DepthwiseConv2D,
    BatchNormalization,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.constraints import max_norm
import tensorflow as tf


def STSNetModel(
    n_classes,
    n_channels=14,
    n_features=20,
    F1=8,
    D=2,
    F2=16,
    proj_k=9,
    dropout=0.25,
):
    """
    A simplified STS-like CNN.
    Input shape: (n_channels, n_features, 1) with channels_last.
    """

    inp = Input(shape=(n_channels, n_features, 1))
    x = Conv2D(F1, (1, proj_k), padding="same", use_bias=False)(inp)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    x = DepthwiseConv2D((n_channels, 1), depth_multiplier=D, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    x = SeparableConv2D(F2, (1, 7), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout)(x)

    out = Dense(
        n_classes,
        activation="softmax",
        kernel_constraint=max_norm(0.25),
    )(x)

    return Model(inp, out)
