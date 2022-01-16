from tensorflow import keras
import tensorflow as tf 
# binary_crossentropy
# L1 or L2
@tf.function
def disc_loss(loss_func, disc_real_output, disc_gen_output):
    real_loss = loss_func(tf.experimental.numpy.full_like(disc_real_output, 0.9, dtype= tf.float32), disc_real_output)
    fake_loss = loss_func(tf.experimental.numpy.full_like(disc_gen_output, 0.1, dtype= tf.float32), disc_gen_output)
    total_loss = real_loss + fake_loss
    return total_loss
def gen_loss(loss_func, disc_gen_output, gen_output, target, l1 = True, LAMBDA = 100):
    gan_loss = loss_func(tf.ones_like(disc_gen_output), disc_gen_output)

    if l1: # Mean absolute error
        l_loss = tf.reduce_mean(tf.abs(target - gen_output))
    else: # Mean square error
        l_loss = tf.reduce_sum(tf.pow(gen_output - target, 2)) / (2 * gen_output.shape[0])
    total_loss = gan_loss + LAMBDA * l_loss
    return total_loss