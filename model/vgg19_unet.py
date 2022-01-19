from replace_activation_in_pretrain_model import * 
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def decoder_block(x, y, filters):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = Conv2DTranspose(filters, 4, strides= 2, padding= 'same', kernel_initializer= initializer)(x)
    x = Concatenate(axis = 3)([x,y])
    x = Conv2D(filters, 4, padding= 'same', kernel_initializer= initializer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, 4, padding= 'same', kernel_initializer= initializer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def g_model(input_shape= (512,512,3), trainable = True, weights = None):
    inputs = Input(shape = input_shape)
    initializer = tf.random_normal_initializer(0., 0.02)

    vgg19 = VGG19(include_top=False, weights= weights, input_tensor=inputs)
    replace_swish_with_relu(vgg19)
    vgg19.trainable = trainable
    """ Encoder """
    s1 = vgg19.get_layer("block1_conv2").output         ## (512 x 512)
    s2 = vgg19.get_layer("block2_conv2").output         ## (256 x 256)
    s3 = vgg19.get_layer("block3_conv4").output         ## (128 x 128)
    s4 = vgg19.get_layer("block4_conv4").output         ## (64 x 64)

    """ Bridge """
    b1 = vgg19.get_layer("block5_conv4").output         ## (32 x 32)
    
    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d1 = Dropout(0.3)(d1)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d2 = Dropout(0.3)(d2)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d3 = Dropout(0.3)(d3)
    d4 = decoder_block(d3, s1, 64)          ## 512

    outputs = Conv2D(3,4, padding= 'same', activation= 'tanh', kernel_initializer= initializer)(d4)
    return Model(inputs, outputs, name = 'g_model')

if __name__ == '__main__':
    input_shape = (224,224,3)
    model = g_model(input_shape, False)
    model.summary()