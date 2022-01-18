from tensorflow.keras.models import Model
import tensorflow as tf 
from tensorflow.keras.layers import *
def down_block(x, filter, strides = 2, use_batch_norm = True):
    x = Conv2D(filter, 4, strides= strides)(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    return x 
def d_model(inp_shape = (256,256,3), feature = [64,128,256,512]):
    initializer = tf.random_normal_initializer(0., 0.02)
    inp1 = Input(shape= inp_shape)
    inp2 = Input(shape= inp_shape)    
    x = Concatenate()([inp1, inp2])
    x = down_block(x, feature[0], use_batch_norm= True)
    x = down_block(x, feature[1])
    x = down_block(x, feature[2])
    x = ZeroPadding2D()(x)
    x = down_block(x, feature[-1], 1)
    x = ZeroPadding2D()(x)
    x = Conv2D(1,4, strides= 1, kernel_initializer= initializer)(x)
    return Model([inp1,inp2], x, name = 'd_model')

if __name__ == '__main__':
    model = d_model()
    model.summary()