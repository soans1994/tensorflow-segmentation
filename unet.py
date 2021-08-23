import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
#import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

#conv block conv-bn-act-conv-bn-act-dropout(if enabled)

def conv_block(x, filter_size, filters, dropout, batch_norm=False):
    conv = layers.Conv2D(filters, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    conv = layers.Conv2D(filters, (filter_size, filter_size), padding="same")(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    if dropout>0:
        conv = layers.Dropout(dropout)(conv)
    return conv

def repeat_elem(tensor, rep):
    return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={"repnum": rep})(tensor)

def unet(input_shape, num_classes=1, dropout_rate=0.0, batch_norm=True):
    filter_num = 64#first layer filters
    filter_size = 3# size of conv filter
    up_sample_size = 2#size of upsample filter

    inputs = layers.Input(input_shape, dtype=tf.float32)
    #downsamples layers
    #downres 1, conv+pooling
    conv_128 = conv_block(inputs, filter_size, filter_num, dropout_rate, batch_norm)
    pool_64 = layers.MaxPool2D(pool_size=(2,2))(conv_128)
    #downres 2
    conv_64 = conv_block(pool_64, filter_size, filter_num*2, dropout_rate, batch_norm)
    pool_32 = layers.MaxPool2D(pool_size=(2,2))(conv_64)
    #downres 3
    conv_32 = conv_block(pool_32, filter_size, filter_num*4, dropout_rate, batch_norm)
    pool_16 = layers.MaxPool2D(pool_size=(2,2))(conv_32)
    #downres 4
    conv_16 = conv_block(pool_16, filter_size, filter_num*8, dropout_rate, batch_norm)
    pool_8 = layers.MaxPool2D(pool_size=(2,2))(conv_16)
    #downres 5 downsampling conv only
    conv_8 = conv_block(pool_8, filter_size, filter_num*16, dropout_rate, batch_norm)

    #upsampling layers
    #upres 6
    up_16 = layers.UpSampling2D(size=(up_sample_size, up_sample_size), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, conv_16], axis=3)
    up_conv_16 = conv_block(up_16, filter_size, filter_num*8, dropout_rate, batch_norm)

    #upres 7
    up_32 = layers.UpSampling2D(size=(up_sample_size, up_sample_size), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, conv_32], axis=3)
    up_conv_32 = conv_block(up_32, filter_size, filter_num*4, dropout_rate, batch_norm)

    #upres 8
    up_64 = layers.UpSampling2D(size=(up_sample_size, up_sample_size), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, conv_64 ], axis=3)
    up_conv_64 = conv_block(up_64, filter_size, filter_num*2, dropout_rate, batch_norm)

    #upres 9
    up_128 = layers.UpSampling2D(size=(up_sample_size, up_sample_size), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, conv_128], axis=3)
    up_conv_128 = conv_block(up_128, filter_size, filter_num, dropout_rate, batch_norm)

    #1*1 conv layers
    conv_final = layers.Conv2D(num_classes, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation("softmax")(conv_final)

    #model
    model = models.Model(inputs, conv_final, name="unet")
    print(model.summary())
    return model
'''
if __name__=="__main__":
    input_shape = (128,128,3)
    model = unet((input_shape), num_classes=10, dropout_rate=0.0, batch_norm=True)
    #tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
'''