from telnetlib import GA
from loss_model import *
class Gan_model(keras.Model):
    def __init__(self, discriminator, generator):
        super(Gan_model, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
    def complie(self, d_optimizer, g_optimizer, loss_func):
        super(Gan_model, self).complie()
        self.loss_func = loss_func
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(x, training = True)
            disc_real_output = self.discriminator([x, y], training = True)
            disc_generated_output = self.discriminator([x, gen_output], training = True)

            generator_loss = gen_loss(self.loss_func, disc_generated_output, gen_output, y)
            discriminator_loss = disc_loss(self.loss_func, disc_real_output, disc_generated_output)
            
        generator_gradient = gen_tape.gradient(generator_loss, self.generator.trainable_variables)
        discriminator_gradient = disc_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(discriminator_gradient, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(generator_gradient, self.generator.trainable_variables))
        return {"d_loss": discriminator_loss, "g_loss": generator_loss}


        