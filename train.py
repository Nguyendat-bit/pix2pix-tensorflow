from argparse import ArgumentParser
from generator_model import resnet50_unet, vgg19_unet, unet
import discriminator_model
from GeneratorCustom import *
import sys 
import glob2
from custom_model import Gan_model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--all-train', action= 'append', required= True)
    parser.add_argument('--batch-size',type = int, default= 8 )
    parser.add_argument('--bone', type= str, default= 'unet', help = 'unet, resnet50_unet, vgg19_unet')
    parser.add_argument('--lr-g', type = float, default= 2e-4)
    parser.add_argument('--lr-d', type = float, default= 2e-4)
    parser.add_argument('--beta1-g', type= float, default= 0.5)
    parser.add_argument('--beta1-d', type= float, default= 0.5)
    parser.add_argument('--image-size', default= 256, type= int)    
    parser.add_argument('--model-save', default= 'g_model.h5', type= str)
    parser.add_argument('--shuffle', default= True, type= bool)
    parser.add_argument('--epochs', type = int, required= True)   
    parser.add_argument('--random-brightness', type= bool, default= False)
    parser.add_argument('--random-flip', type= bool, default= True)
    parser.add_argument('--rotation', type= float, default= None)
    parser.add_argument('--pretrain', type = bool, default= True)
    parser.add_argument('--weights', type = bool, default= True)
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    print('---------------------Welcome to Pix2Pix-------------------')
    print('Author')
    print('Github: Nguyendat-bit')
    print('Email: nduc0231@gmail')
    print('---------------------------------------------------------------------')
    print('Training Pix2Pix model with hyper-params:')
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    
    if args.random_brightness:
        brightness_range = (0.75, 1.0)
    else:
        brightness_range = None

    if args.pretrain:
        args.weights = 'imagenet'   

    train_domainA  = sorted(glob2.glob(args.all_train[0]))
    train_domainB = sorted(glob2.glob(args.all_train[1]))
    all_train_filenames = list(zip(train_domainA, train_domainB))
    data = DataGenerator(all_train_filenames, (args.image_size, args.image_size), args.batch_size, brightness_range, args.random_flip, args.shuffle)
    inp_shape=(args.image_size, args.image_size, 3)
    if args.bone == 'unet':
        g_model = unet.g_model(inp_shape)
    elif args.bone == 'resnet50_unet':
        g_model = resnet50_unet.g_model(inp_shape, args.pretrain, args.weights)
    elif args.bone == 'vgg19_unet':
        g_model = vgg19_unet.g_model(inp_shape, args.pretrain, args.weights)

    d_model = discriminator_model.d_model(inp_shape)

    pix2pix = Gan_model(d_model, g_model)

    # optimizer
    d_optimizer = Adam(learning_rate= args.lr_d, beta_1= args.beta1_d)
    g_optimizer = Adam(learning_rate= args.lr_g, beta_1= args.beta1_g)
    bce = BinaryCrossentropy(from_logits= True)

    pix2pix.complie(d_optimizer, g_optimizer, loss_func= bce)
    pix2pix.fit(data, epochs = args.epochs)
    tf.keras.models.save_model()

    