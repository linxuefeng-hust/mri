import keras
from keras.models import Sequential
from keras.models import save_model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K
import numpy as np
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from matplotlib import rc
from keras.utils import plot_model
from PIL import Image
from sklearn.utils import shuffle
import os
from keras.utils import plot_model
import sys
import load_easydata
import c3d_lstm_model
from keras import regularizers
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, Activation

import scipy.io
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
#import read_npy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure(figsize=(6.3, 4.7))
        plt.rcParams.update(params)
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.subplots_adjust(0.06, 0.1, 0.96, 0.94, 0.2, 0.3)
        plt.savefig("result.png")
        plt.show()


'''
# mat290
def load_3D_from_mat():
    Data_tmp = sio.loadmat('Data_290.mat')

    Data = Data_tmp['Data'].astype('float32')
    Labels = Data_tmp['Labels'].astype('int32')

    Data, Labels = shuffle(Data, Labels, random_state=0)

    #sio.savemat('Data_290.mat', {'Data': Data, 'Labels': Labels})

    x_train = Data[:260, :, :, :]
    y_train = Labels[:260]
    x_test = Data[260:, :, :, :]
    y_test = Labels[260:]

    return x_train, y_train, x_test, y_test
'''
#cur_filepath = r'D:/AD_Norm/12_5_nc_ad_train_3d_data.mat' # 9.18日修改，为测试重复项是否对结果有影响
#cur_filepath = r'D:/AD_Norm/12_5_nc_mci_train_3d_data.mat'
#cur_filepath =r'D:/AD_Norm/12_14_for_train_nc_mci_train_3d_data.mat'
#cur_filepath = r'D:/AD_Norm/12_17_for_train_nc_mci_train_3d_data.mat'
#cur_filepath = r'D:/AD_Norm/12_19_ad_mci_train_3d_data.mat'
cur_filepath = r'D:/AD_Norm/12_21_ad_mci_train_3d_data.mat'
train_data = scipy.io.loadmat(cur_filepath)  # 读入存放数据的字典
x_train = np.array(train_data['x_train'])
x_train = x_train.astype('float32')
y_train = np.array(train_data['y_train'])
y_train = y_train.astype('float32')
x_test = np.array(train_data['x_test'])
y_test = np.array(train_data['y_test'])

# input image dimensions
img_rows, img_cols, img_depth = 61, 73, 61
num_classes = 2 # 二分类问题

# the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    print("i am first")
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols, img_depth)
    x_test = x_test.reshape(x_test.shape[0],  1, img_rows, img_cols, img_depth)
    input_shape = ( 1, img_rows, img_cols, img_depth)

else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_depth, 1)
    x_test = x_test.reshape(x_test.shape[0],  img_rows, img_cols, img_depth, 1)
    y_test = y_test.reshape((-1, 1))
    y_train = y_train.reshape((-1, 1))
    input_shape = ( img_rows, img_cols, img_depth, 1)

'''
else:
    x_train = x_train.reshape(x_train.shape[0],  img_rows, img_cols, img_depth, 1)
    x_test = x_test.reshape(x_test.shape[0],  img_rows, img_cols, img_depth, 1)
    input_shape = ( img_rows, img_cols, img_depth, 1)kernel_regularizer=regularizers.l2(0.01)
'''

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test,  num_classes)


batch_size = 4  # batch为8的训练过程收敛效率明显不如batch为2，timeamount改为8之后，就跑不了8个batch了
epochs = 150


print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

def fmri_c3d(input_shape, nb_classes):
    """
    Build a 3D convolutional network, based loosely on C3D.
        https://arxiv.org/pdf/1412.0767.pdf
    """
    # 8.27加入一些正则项
    # Model.
    weight_decay = 0.00001  # 0.0005 # 0.0001 #0.001
    model = Sequential()
    model.add(Conv3D(32, (3, 3, 3),  padding='same', input_shape=input_shape, activation='relu'))
    #model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
    model.add(Conv3D(32, (3, 3, 3),  padding='same', input_shape=input_shape, activation='relu')) # 8.31加深层数
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
    # 9.17 去掉卷积层的正则约束
    model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu'))
    #model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))  # 9.17去掉maxpooling
    model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu')) # 8.29加入
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
    model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu'))
    model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu',))

    model.add(BatchNormalization()) # 9.17

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
    #model.add(Conv3D(256, (2, 2, 2), activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv3D(256, (3, 3, 3), padding='same', activation='relu')) # 9.10 8.31修改为3，3，3卷积核
    model.add(Conv3D(256, (3, 3, 3), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))) # 9.10 加入最大池化和512个卷积核
    model.add(Conv3D(512, (3, 3, 3), padding='same', activation='relu'))
    model.add(Conv3D(512, (3, 3, 3), padding='same', activation='relu'))
    #model.add(Conv3D(256, (2, 2, 2), activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
    #model.add(Conv3D(256, (2, 2, 2), activation='relu')) #8.31.去掉正则，加深层数

    model.add(BatchNormalization()) #9.17
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    # 9.17
    model.add(Conv3D(512, (3, 3, 3), padding='same', activation='relu'))
    model.add(Conv3D(512, (3, 3, 3), padding='same', activation='relu'))
    #model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Flatten())
    #model.add(Dense(1024))
    #model.add(Dropout(0.5))
    model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay))) #1024
    #model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model

def fmri_vgg_3d(input_shape, num_classes):
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
    # change padding from same to valid 18.9.8
    model = Sequential()
    weight_decay = 0.0005#0.0005 # 0.0001

    model.add(Conv3D(32, (3, 3, 3), padding='same',
                     input_shape=input_shape)) # 9.13 64 to 32
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv3D(32, (3, 3, 3), padding='same')) # 9.13 64 to 32
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(64, (3, 3, 3), padding='same')) # 9.13 128 to 64
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv3D(64, (3, 3, 3), padding='same'))# 9.13 128 to 64
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(128, (3, 3, 3), padding='same')) # 9.13 256 to 128
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv3D(128, (3, 3, 3), padding='same')) # 9.13 256 to 128
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv3D(256, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(256, (3, 3, 3), padding='same')) # 9.13 512 to 256 same to valid
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    #model.add(Conv3D(512, (3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv3D(512, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    #model.add(Conv3D(512, (3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv3D(512, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    #model.add(Conv3D(512, (3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv3D(512, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv3D(512, (3, 3, 3), padding='same'))
    model.add(Conv3D(512, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv3D(512, (3, 3, 3), padding='same'))
    model.add(Conv3D(512, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
    #model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.2)) # 0.5
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


model = fmri_c3d(input_shape, num_classes)   # 9.17添加
filepath = './12_19_3d_c3d_ad_mci_weight.hdf5'
#filepath = './19_5_7_see_para_amounts_3d_c3d_ad_mci_weight.hdf5'
def get_callbacks(filepath, patience=4):
    lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=1e-5, patience=patience, verbose=1, min_lr = 0.00001)
    msave = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True) #after every epoch save model to filepath
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=patience*3+2, verbose=1, mode='auto')
    return [lr_reduce, msave]#, earlystop]
callbacks = get_callbacks(filepath)
model.summary()
# plot_model(model, to_file='model.png')
learning_rate = 0.00001  # 6.25lr从0.001修改为0.0004
#model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=learning_rate, decay=learning_rate/20, momentum=0.9, nesterov=True), metrics=['accuracy'])
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08), metrics=['accuracy'])  # 6.25修改SGD为adam
#json_string = model.to_json()
#with open('mlp_model.json','w') as of:
    #of.write(json_string)
history = LossHistory()
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(x_test, y_test),
          callbacks=callbacks)
# score = model.evaluate(x_test, y_test, verbose=1)
# print(model.metrics_names)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# save_model('D:/AD_Norm/c3dlstm.h5')
