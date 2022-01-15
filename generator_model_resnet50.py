from replace_activation_in_pretrain_model import * 
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def decoder_block(x, y, filters):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = Conv2DTranspose(filters, 4, strides= 2, padding= 'same', kernel_initializer= initializer)(x)
    x = Concatenate(axis = 3)([x,y])
    x = Conv2D(filters, 4, padding= 'same', kernel_initializer= initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters, 4, padding= 'same', kernel_initializer= initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    return x

def generator_model(input_shape= (512,512,3), trainable = True):
    inputs = Input(shape = input_shape)
    initializer = tf.random_normal_initializer(0., 0.02)

   
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
    resnet50.trainable = trainable
    replace_swish_with_relu(resnet50)
    """ Encoder """
    s1 = resnet50.get_layer("input_1").output           ## (512 x 512)
    s2 = resnet50.get_layer("conv1_relu").output        ## (256 x 256)
    s3 = resnet50.get_layer("conv2_block3_out").output  ## (128 x 128)
    s4 = resnet50.get_layer("conv3_block4_out").output  ## (64 x 64)

    """ Bridge """
    b1 = resnet50.get_layer("conv4_block6_out").output  ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d1 = Dropout(0.3)(d1)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d2 = Dropout(0.3)(d2)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d3 = Dropout(0.3)(d3)
    d4 = decoder_block(d3, s1, 64)          ## 512            ## 512

    outputs = Conv2D(3,4, padding= 'same', activation= 'tanh', kernel_initializer= initializer)(d4)
    return Model(inputs, outputs, name = 'g_model')

if __name__ == '__main__':
    input_shape = (224,224,3)
    model = generator_model(input_shape, False)
    model.summary()

