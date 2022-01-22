import tensorflow as tf 
def replace_swish_with_relu(model):
    '''
    Modify passed model by replacing swish activation with relu
    '''
    for layer in tuple(model.layers):
        layer_type = type(layer).__name__
        if hasattr(layer, 'activation') and layer.activation.__name__ == 'relu':
            if layer_type == "Conv2D":
                # conv layer with swish activation
                layer.activation = tf.keras.layers.LeakyReLU(0.2)
            else:
                # activation layer
                layer.activation = tf.keras.layers.LeakyReLU(0.2)
    return model