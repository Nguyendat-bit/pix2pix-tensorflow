from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow as tf 

def encode_block(x, filters, use_batchnorm = True):
    initializer = tf.random_normal_initializer(0, 0.02)
    x = Conv2D(filters, 4, 2, padding= 'same', kernel_initializer= initializer)(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    return x

def decode_block(x, y, filters, use_dropout = False):
    initializer = tf.random_normal_initializer(0, 0.02)
    x = Conv2DTranspose(filters, 4, 2, padding= 'same', kernel_initializer= initializer)(x)
    x = BatchNormalization()(x)
    if use_dropout:
        x = Dropout(0.5)(x)
    x = Concatenate()([x, y])
    x = Activation('relu')(x)
    return x

def g_model(inp_shape = (256,256,3)):
    initializer = tf.random_normal_initializer(0, 0.02)
    inputs = Input(inp_shape)
    # encode
    e1 = encode_block(inputs, 64, False)
    e2 = encode_block(e1, 128)
    e3 = encode_block(e2, 256)
    e4 = encode_block(e3, 512)
    e5 = encode_block(e4, 512)
    e6 = encode_block(e5, 512)
    e7 = encode_block(e6, 512)
   
    # bridge 
    x = encode_block(e7, 512)

    # decode
    d1 = decode_block(x, e7, 512, True)
    d2 = decode_block(d1, e6, 512, True)
    d3 = decode_block(d2, e5, 512, True)
    d4 = decode_block(d3, e4, 512)
    d5 = decode_block(d4, e3, 256)
    d6 = decode_block(d5, e2, 128)
    d7 = decode_block(d6, e1, 64)

    outputs = Conv2DTranspose(3, 4, 2, padding= 'same', activation= 'tanh', kernel_initializer= initializer)(d7)
    return Model(inputs, outputs, name = 'g_model')

if __name__ == '__main__':
    model = g_model()
    model.summary()