import numpy as np

from keras.models import Sequential
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from sklearn.metrics import log_loss
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling3D
from keras import regularizers

from keras.utils import plot_model
from collections import deque
import sys
import load_data

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers
import os
import gc
import keras
from keras.models import Sequential
from keras.models import save_model, load_model
from keras.layers import Dense, Dropout, Flatten

import random
import matplotlib.pyplot as plt
#from matplotlib import rc
from keras.utils import plot_model
#from PIL import Image
from sklearn.utils import shuffle
import os
from keras.utils import plot_model
import load_easydata
import c3d_lstm_model
#import scipy.io
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
import c3d_lstm_model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers.advanced_activations import LeakyReLU

# 实现data_generator



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




#ad_train_save_path = r'E:\small_adni_generator\train\AD'
# 9.15 修改文件路径，选择除了转换时间点和转换时间点之前的数据
#ad_train_save_path = r'E:\all_small_adni_generator\tpm_smooth_unmax\train\AD'
#ad_train_save_path  = r'E:\all_small_adni_generator\train\AD'
#ad_train_save_path = r'E:\matlab_adni_generator_std\train\AD'
ad_train_save_path = r'E:\allgroup_matlab_adni_generator_std\train\AD'
ad_validation_save_path= r'E:\allgroup_matlab_adni_generator_std\validation\AD'
cn_train_save_path = r'E:\allgroup_matlab_adni_generator_std\train\CN'
cn_validation_save_path= r'E:\allgroup_matlab_adni_generator_std\validation\CN'
'''
# 尝试迁移学习，先使用AD，CN的分类训练结果
ad_train_save_path = r'E:\MCI_compa_std\train\convert'
ad_validation_save_path= r'E:\MCI_compa_std\validation\convert'
cn_train_save_path = r'E:\MCI_compa_std\train\no_convert'
cn_validation_save_path= r'E:\MCI_compa_std\validation\no_convert'
mci_train_save_path = r'E:\allgroup_matlab_adni_generator_std\train\MCI'
mci_validation_save_path= r'E:\allgroup_matlab_adni_generator_std\validation\MCI'

'''
#mci_train_save_path = r'E:\all_small_adni_generator\tpm_smooth_unmax\train\MCI'
#mci_validation_save_path= r'E:\all_small_adni_generator\tpm_smooth_unmax\validation\MCI'  # 9.26 remove mci data

train_path=r'E:\allgroup_matlab_adni_generator_std\train'
validation_path=r'E:\allgroup_matlab_adni_generator_std\validation'
'''
train_path=r'E:\MCI_compa_std\train'
validation_path=r'E:\MCI_compa_std\validation'
'''

#img_rows, img_cols, img_depth = 256, 256, 192
#img_rows, img_cols, img_depth = 192, 192, 160   # 9.18修改
#img_rows, img_cols, img_depth = 128, 128, 96
#img_rows, img_cols, img_depth = 91, 109, 91
#img_rows, img_cols, img_depth = 121, 145, 121
#img_rows, img_cols, img_depth = 256, 256, 160
#img_rows, img_cols, img_depth = 256, 256, 166
img_rows, img_cols, img_depth = 121, 145 ,121
batch_size = 8
image_shape=(img_rows, img_cols, img_depth, 1)
#nb_classes=3
nb_classes = 2 # 9.12测试一下二分类效果
epochs = 110
learning_rate = 0.0001 # 0.0001

nb_train_samples=len(os.listdir(ad_train_save_path))+len(os.listdir(cn_train_save_path))  #+len(os.listdir(mci_train_save_path))
nb_validation_samples=len(os.listdir(ad_validation_save_path))+len(os.listdir(cn_validation_save_path))  #+len(os.listdir(mci_validation_save_path))
#nb_train_samples=len(os.listdir(cn_train_save_path))+len(os.listdir(mci_train_save_path)) # 9.12试一下cn和mci/cn和ad的二分类
#nb_validation_samples=len(os.listdir(cn_train_save_path))+len(os.listdir(mci_validation_save_path))


'''
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=image_shape,
        batch_size=batch_size,
        )

validation_generator = test_datagen.flow_from_directory(
        validation_path,
        target_size=image_shape,
        batch_size=batch_size,
        )
'''






def mygenerator(gen_path, batch_size, nb_classes, img_rows, img_cols, img_depth):
    gen_ad_path=os.path.join(gen_path, 'AD')
    #gen_ad_path = os.path.join(gen_path, 'convert')
    #gen_mci_path=os.path.join(gen_path, 'MCI')
    gen_cn_path=os.path.join(gen_path, 'CN')
    #gen_cn_path = os.path.join(gen_path, 'no_convert')
    gen_train_path=[]
    gen_label_path=[]
    temp_gen_train_path=[]
    temp_gen_label_path=[]


    for file in os.listdir(gen_ad_path):
        cur_gen_path=os.path.join(gen_ad_path,file)
        temp_gen_train_path.append(cur_gen_path)
        temp_gen_label_path.append(0)   # AD的标签作为0
    '''
    for file in os.listdir(gen_mci_path):
        cur_gen_path = os.path.join(gen_mci_path, file)
        temp_gen_train_path.append(cur_gen_path)
        temp_gen_label_path.append(1)  # mci的标签作为1
        #temp_gen_label_path.append(0)
    '''

    for file in os.listdir(gen_cn_path):
        cur_gen_path = os.path.join(gen_cn_path, file)
        temp_gen_train_path.append(cur_gen_path)
        #temp_gen_label_path.append(2)  # cn的标签作为2
        temp_gen_label_path.append(1)
    # 将数据与标签成对打散

    #shu_list = list(zip(gen_train_path, gen_label_path))
    #random.shuffle(shu_list)
    #gen_train_path[:], gen_label_path[:] = zip(shu_list) # 当list过大时，此语句会报错
    index_shuf=list(range(len(temp_gen_label_path)))
    random.shuffle(index_shuf)
    j=0
    for j in index_shuf:
        gen_train_path.append(temp_gen_train_path[j])
        gen_label_path.append(temp_gen_label_path[j])

    del temp_gen_label_path, temp_gen_train_path
    gc.collect()  # 清除缓存

    batchcount=0
    i=0
    yeild_count = 0  # 记录产生了多少个batch
    while True:
        for gen_i in range(len(gen_train_path)):
            if batchcount ==0:
                inputs = []
                labels = []
            temp_img = np.load(gen_train_path[gen_i])
            temp_label = gen_label_path[gen_i]
            inputs.append(temp_img)
            labels.append(temp_label)
            batchcount += 1
            if gen_i == 222:
                stop_flag = 0


            # 为了保证当剩余的样本量小于一个batchsize时仍然能够产生数据
            if batchcount >= batch_size:
                batchcount = 0
                inputs = np.array(inputs)
                inputs = inputs.astype('float32')
                labels = np.array(labels)
                #labels = inputs.astype('float32')
                #print(labels.shape[0])
                inputs = inputs.reshape(inputs.shape[0], img_rows, img_cols, img_depth, 1)
                labels = labels.reshape((-1, 1))
                labels = keras.utils.to_categorical(labels, nb_classes)
                yeild_count += 1
                yield inputs, labels
                #test_flag=0
            elif (batchcount < batch_size) and (yeild_count*batch_size + batchcount >= len(gen_train_path)):
                batchcount = 0
                inputs = np.array(inputs)
                labels = np.array(labels)
                print(labels.shape[0])
                inputs = inputs.reshape(inputs.shape[0], img_rows, img_cols, img_depth, 1)
                labels = labels.reshape((-1, 1))
                labels = keras.utils.to_categorical(labels, nb_classes)
                yeild_count = 0 # 当进入此条件时，一个epoch所有的数据皆以完全产生
                yield inputs, labels
                #test_flag=0
            gc.collect()  # 清除缓存



'''
if __name__=='__main__':
    train_path = r'E:\adni_generator\train'
    validation_path = r'E:\adni_generator\validation'

    img_rows, img_cols, img_depth = 256, 256, 192
    batch_size = 4
    image_shape = (img_rows, img_cols, img_depth, 1)
    nb_classes = 3
    epochs = 150
    learning_rate = 0.0001
    test_gen=mygenerator(train_path, batch_size, nb_classes, img_rows, img_cols, img_depth)
    flag=0

'''



train_generator=mygenerator(train_path, batch_size, nb_classes, img_rows, img_cols, img_depth)
validation_generator=mygenerator(validation_path, batch_size, nb_classes, img_rows, img_cols, img_depth)




def vgg_3d(input_shape, num_classes):
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
    # change padding from same to valid 18.9.8
    model = Sequential()
    weight_decay = 0.0001#0.0005 # 0.0001

    model.add(Conv3D(64, (3, 3, 3), padding='same',
                     input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay))) # 9.13 64 to 32
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv3D(64, (3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))) # 9.13 64 to 32
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(128, (3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))) # 9.13 128 to 64
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv3D(128, (3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))# 9.13 128 to 64
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(256, (3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))) # 9.13 256 to 128
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv3D(256, (3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))) # 9.13 256 to 128
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv3D(256, (3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(512, (3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))) # 9.13 512 to 256 same to valid
    # 10.2 此层之前的padding方式都曾被改为valid
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv3D(512, (3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    #model.add(Conv3D(512, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv3D(512, (3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    #model.add(Conv3D(512, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(512, (3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    #model.add(Conv3D(512, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv3D(512, (3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    #model.add(Conv3D(512, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv3D(512, (3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    #model.add(Conv3D(512, (3, 3, 3), padding='same'))
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


def mri_vgg_3d(input_shape, num_classes):
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
    # change padding from same to valid 18.9.8
    # 9.20 divided kernel numbers with 8
    model = Sequential()
    weight_decay = 0.0001#0.0005 # 0.0001

    model.add(Conv3D(4, (3, 3, 3), padding='same',
                     input_shape=input_shape)) # 9.13 64 to 32
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.3))

    model.add(Conv3D(4, (3, 3, 3), padding='same')) # 9.13 64 to 32
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(8, (3, 3, 3), padding='same')) # 9.13 128 to 64
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv3D(8, (3, 3, 3), padding='same'))# 9.13 128 to 64
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(16, (3, 3, 3), padding='same')) # 9.13 256 to 128
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv3D(16, (3, 3, 3), padding='same')) # 9.13 256 to 128
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv3D(32, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(32, (3, 3, 3), padding='same')) # 9.13 512 to 256 same to valid
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    #model.add(Conv3D(512, (3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv3D(64, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    #model.add(Conv3D(512, (3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv3D(64, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    #model.add(Conv3D(512, (3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv3D(64, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv3D(128, (3, 3, 3), padding='same'))  # 这也是512改的
    model.add(Conv3D(128, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv3D(256, (3, 3, 3), padding='same')) # 512 to 256
    model.add(Conv3D(256, (3, 3, 3), padding='same'))
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




def c3d(input_shape, nb_classes):
    """
    Build a 3D convolutional network, based loosely on C3D.
        https://arxiv.org/pdf/1412.0767.pdf
    """
    # 8.27加入一些正则项
    # Model.
    weight_decay = 0.0001  # 0.0005 # 0.0001 #0.001
    model = Sequential()
    # 10.2 divided fliters with two
    model.add(Conv3D(8, (3, 3, 3),  padding='same', input_shape=input_shape, activation='relu'))  # 9.20 change kernel number from 32 to 8 # 10.2 change same to valid; fliter size 8 to 4
    #model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
    model.add(Conv3D(8, (3, 3, 3),  padding='same', activation='relu')) # 8.31加深层数 #9.20 change kernel number from 32 to 8
    model.add(BatchNormalization())  # 10.2 remove
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(16, (3, 3, 3), padding='same', activation='relu'))
    model.add(Conv3D(16, (3, 3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())  # 10.2 remove
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    # 9.17 去掉卷积层的正则约束
    model.add(Conv3D(32, (3, 3, 3), padding='same', activation='relu')) # 8.31加深层数 #9.20 change kernel number from 64 to 16
    model.add(Conv3D(32, (3, 3, 3), padding='same', activation='relu')) # 8.29加入
    model.add(BatchNormalization())  # 9.20加入，试图加快收敛
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu'))
    model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu',))  # 128 to 32
    model.add(BatchNormalization())  # 9.20加入，试图加快收敛  # 10.2 remove
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    #model.add(Conv3D(256, (2, 2, 2), activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
    # 9.18为以下四个卷积层加入正则化方法
    model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay))) # 9.10 8.31修改为3，3，3卷积核
    #model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay))) # 256 to 64\

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))) # 9.10 加入最大池化和512个卷积核
    #model.add(Conv3D(256, (3, 3, 3), padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))  # 512 to 128
    #model.add(Conv3D(256, (3, 3, 3), padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))  #  512 to 128
    model.add(BatchNormalization())  # 9.20加入，试图加快收敛

    model.add(Flatten())
    #model.add(GlobalAveragePooling3D())
    #model.add(Dense(1024)) # add 10.17
    model.add(Dropout(0.5))
    #model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay))) #1024
    model.add(Dense(512))
    #model.add(Dropout(0.5))
    #model.add(Dense(nb_classes, activation='softmax',kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dense(nb_classes, activation='softmax'))

    return model


def matlab_mri_vgg_3d(input_shape, num_classes):
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
    # change padding from same to valid 18.9.8
    # 9.20 divided kernel numbers with 8
    model = Sequential()
    weight_decay = 0.0001#0.0005 # 0.0001


    model.add(Conv3D(64, (3, 3, 3), padding='same',
                     input_shape=input_shape)) # 9.13 64 to 32
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.3))

    model.add(Conv3D(64, (3, 3, 3), padding='same')) # 9.13 64 to 32
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(128, (3, 3, 3), padding='same')) # 9.13 128 to 64
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv3D(128, (3, 3, 3), padding='same'))# 9.13 128 to 64
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(256, (3, 3, 3), padding='same')) # 9.13 256 to 128
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv3D(256, (3, 3, 3), padding='same')) # 9.13 256 to 128
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv3D(256, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(512, (3, 3, 3), padding='same')) # 9.13 512 to 256 same to valid
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    #model.add(Conv3D(512, (3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv3D(512, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    #model.add(Conv3D(512, (3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv3D(512, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(512, (3, 3, 3), padding='same'))  # 这也是512改的
    #model.add(Conv3D(512, (3, 3, 3), padding='same')) # 10.2 22:14跑出0.7031之后去除
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))


    model.add(Conv3D(512, (3, 3, 3), padding='same')) # 512 to 256
    #model.add(Conv3D(512, (3, 3, 3), padding='same')) # 10.2 22:14跑出0.7031之后去除
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
    #model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.2)) # 0.5  10.2 22:14跑出0.7031之后从0.2修改为0.5
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model

def globalpool_matlab_mri_vgg_3d(input_shape, num_classes):
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
    # change padding from same to valid 18.9.8
    # 9.20 divided kernel numbers with 8
    model = Sequential()
    weight_decay = 0.0001#0.0005 # 0.0001


    model.add(Conv3D(8, (3, 3, 3), padding='same',
                     input_shape=input_shape)) # 9.13 64 to 32 除以2
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.3))

    model.add(Conv3D(8, (3, 3, 3), padding='same')) # 9.13 64 to 32
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(32, (3, 3, 3), padding='same')) # 9.13 128 to 64
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv3D(32, (3, 3, 3), padding='same'))# 9.13 128 to 64
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(64, (3, 3, 3), padding='same')) # 9.13 256 to 128
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv3D(64, (3, 3, 3), padding='same')) # 9.13 256 to 128
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv3D(128, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(128, (3, 3, 3), padding='same')) # 9.13 512 to 256 same to valid
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    #model.add(Conv3D(512, (3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv3D(256, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))


    model.add(Conv3D(256, (3, 3, 3), padding='same')) # 512 to 256
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(GlobalAveragePooling3D())

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


def lenet3d(input_shape, num_classes):
    model = Sequential()
    model.add(Conv3D(6, (3,3,3), padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))
    model.add(Conv3D(16, (3,3,3), padding='same', activation='relu'))  # filter size5 to size3

    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))
    #model.add(Conv3D(120, (5,5,5), padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(120,activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model



filepath = './ad_nc_pretrain_mci_convert_weight.hdf5'


def get_callbacks(filepath, patience=2):
    lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=1e-5, patience=patience, verbose=1, min_lr = 0.00001)
    msave = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True) #after every epoch save model to filepath
    #earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=patience*3+2, verbose=1, mode='auto')
    return [lr_reduce, msave]#, earlystop]

callbacks = get_callbacks(filepath)
#model=lenet3d(image_shape, nb_classes) #即使进行了修改，网络依然无法本地运行
model=c3d(image_shape, nb_classes)   # 过拟合很严重。9.12与9.13测试了两种二分类，效果也很差，不确定是数据的问题还是网络的问题
#model = c3d_lstm_model.vgg_3d(image_shape)
#model = lenet3d(image_shape, nb_classes)#c3d(image_shape, nb_classes)
model.summary()
print('train numbers: %s'%(nb_train_samples))
print('validation numbers :%s'%(nb_validation_samples))
#model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=learning_rate, decay=learning_rate/20, momentum=0.9, nesterov=True), metrics=['accuracy'])
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08), metrics=['accuracy'])
history = LossHistory()
model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples//batch_size,
        verbose=2, # 2为每个epoch输出一次
        epochs=epochs,
        callbacks=callbacks,   # 9.21 新增callbacks
        validation_data=validation_generator,
        validation_steps=nb_validation_samples//batch_size)
#model.save('./avepool_twoclasses_ad_weight.hdf5')

#history.loss_plot('epoch')