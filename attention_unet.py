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

def res_conv_block(x, filter_size, filters, dropout, batch_norm=False)
    #conv-bn-act-conv-bn-shortcut-bn-(shortcut+bn)+act
    conv = layers.Conv2D(filters, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    conv = layers.Conv2D(filters, (filter_size, filter_size), padding="same")(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    #conv = layers.Activation("relu")(conv)
    if dropout>0:
        conv = layers.Dropout(dropout)(conv)
    shortcut = layers.Conv2D(filters, kernel_size=(1,1), padding="same")(x)# not direct x , but conv of x
    if batch_norm=True:
        shortcut = layers.BatchNormalization(axis=3)(shortcut)
    res_path = layers.add([shortcut, conv])
    res_path = layers.Activation("relu")(res_path)
    return res_path

def gating_signal(input, output_size, batch_norm=False):#resize downlayer feature to same dim as up layer feature map using 1x1 conv(return gating signal feature map with dim as up layer
    x = layers.Conv2D(output_size, kernel_size=(1,1), padding="same")(input)
    if batch_norm is True:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")
    return x

def attention_block(x, gating, filter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)
    # make x signal same as gating signal since x is 1 dimension up 128x128x128 to 64x64x128========reduce image dimension
    theta_x = layers.Conv2D(filter_shape, (2,2), padding="same")(x)#stride 2 like max pool
    shape_theta_x = K.int_shape(theta_x)
    # make gating signals filters same as x signals filters 64x64x64 to 64x64x128============increase filters
    phi_g = layers.Conv2D(filter_shape, (1,1), padding="same")(gating)
    upsample_g = layers.Conv2DTranspose(filter_shape, (3,3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]), padding="same")(phi_g)#this is extra not clear
    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation("relu")(concat_xg)
    psi = layers.Conv2D(1, (1,1), padding="same")(act_xg)
    sigmoid_xg = layers.Activation("sigmoid")(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)
    upsample_psi = repeat_elem(upsample_psi, shape_x[3])
    y = layers.multiply([upsample_psi, x])
    result = layers.Conv2D(shape_x[3], (1,1), padding="same")(y)
    result = layers.BatchNormalization()(result)
    return result


def attention_unet(input_shape, num_classes=1, dropout_rate=0.0, batch_norm=True):
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
    gating_16 = gating_signal(conv_8, filter_num*8, batch_norm)
    att_16 = attention_block(conv_16, gating_16, filter_num*8)
    up_16 = layers.UpSampling2D(size=(up_sample_size, up_sample_size), data_format="channels_last")(att_16)
    up_16 = layers.concatenate([up_16, conv_16], axis=3)
    up_conv_16 = conv_block(up_16, filter_size, filter_num*8, dropout_rate, batch_norm)

    #upres 7
    gating_32 = gating_signal(up_conv_16, filter_num*4, batch_norm)
    att_32 = attention_block(conv_32, gating_32, filter_num*4)
    up_32 = layers.UpSampling2D(size=(up_sample_size, up_sample_size), data_format="channels_last")(att_32)
    up_32 = layers.concatenate([up_32, conv_32], axis=3)
    up_conv_32 = conv_block(up_32, filter_size, filter_num*4, dropout_rate, batch_norm)

    #upres 8
    gating_64 = gating_signal(up_conv_32, filter_num*2, batch_norm)
    att_64 = attention_block(conv_64, gating_64, filter_num*2)
    up_64 = layers.UpSampling2D(size=(up_sample_size, up_sample_size), data_format="channels_last")(att_64)
    up_64 = layers.concatenate([up_64, conv_64 ], axis=3)
    up_conv_64 = conv_block(up_64, filter_size, filter_num*2, dropout_rate, batch_norm)

    #upres 9
    gating_128 = gating_signal(up_conv_64, filter_num, batch_norm)
    att_128 = attention_block(conv_128, gating_128, filter_num)
    up_128 = layers.UpSampling2D(size=(up_sample_size, up_sample_size), data_format="channels_last")(att_128)
    up_128 = layers.concatenate([up_128, conv_128], axis=3)
    up_conv_128 = conv_block(up_128, filter_size, filter_num, dropout_rate, batch_norm)

    #1*1 conv layers
    conv_final = layers.Conv2D(num_classes, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation("softmax")(conv_final)

    #model
    model = models.Model(inputs, conv_final, name="attention_unet")
    print(model.summary())
    return model

if __name__=="__main__":
    input_shape = (128,128,3)
    model = attention_unet((input_shape), num_classes=10, dropout_rate=0.0, batch_norm=True)
    #tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
