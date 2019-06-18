'''
模仿neurocomputing那篇文章，使用包含残差与separable convolutions构建的深度模型对pMCI与sMCI进行分类
'''
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


from keras.layers import Input, Conv3D, MaxPooling3D, Dropout, BatchNormalization, Reshape, Dense, ELU, concatenate, add, Lambda, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
from keras.metrics import binary_crossentropy
import numpy as np
import math

import sys
#sys.path.append('/home/ses88/venv/PropagAgeing')
sys.path.append('/Users/simeonspasov/DL files/MCI advanced')
from sepconv3D import SeparableConv3D

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
'''
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
#mci_train_save_path = r'E:\allgroup_matlab_adni_generator_std\train\MCI'
#mci_validation_save_path= r'E:\allgroup_matlab_adni_generator_std\validation\MCI'


#mci_train_save_path = r'E:\all_small_adni_generator\tpm_smooth_unmax\train\MCI'
#mci_validation_save_path= r'E:\all_small_adni_generator\tpm_smooth_unmax\validation\MCI'  # 9.26 remove mci data
'''
train_path=r'E:\allgroup_matlab_adni_generator_std\train'
validation_path=r'E:\allgroup_matlab_adni_generator_std\validation'
'''
train_path=r'E:\MCI_compa_std\train'
validation_path=r'E:\MCI_compa_std\validation'


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
    #gen_ad_path=os.path.join(gen_path, 'AD')
    gen_ad_path = os.path.join(gen_path, 'convert')
    #gen_mci_path=os.path.join(gen_path, 'MCI')
    #gen_cn_path=os.path.join(gen_path, 'CN')
    gen_cn_path = os.path.join(gen_path, 'no_convert')
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


train_generator=mygenerator(train_path, batch_size, nb_classes, img_rows, img_cols, img_depth)
validation_generator=mygenerator(validation_path, batch_size, nb_classes, img_rows, img_cols, img_depth)



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


def XAlex3D(mri_volume, nb_classes, w_regularizer=None, drop_rate=0.):
    # 3D Multi-modal deep learning neural network (refer to fig. 4 for chain graph of architecture)
    #conv1_left = _conv_bn_relu_pool_drop(24, 11, 13, 11, strides=(4, 4, 4), w_regularizer=w_regularizer,
                                         #drop_rate=drop_rate, pool=True)(mri_volume)  # 左右通道
    conv1_left = Conv3D(24, (11, 13, 11), strides=(2, 2, 2), kernel_initializer="he_normal",
                             padding='same', kernel_regularizer = w_regularizer)(mri_volume)
    norm = BatchNormalization()(conv1_left)
    elu = ELU()(norm)
    elu=MaxPooling3D(pool_size=3, strides=2)(elu)
    block1=Dropout(drop_rate) (elu)

    # Second layer
    conv2_left = Conv3D(48, (5, 6, 5), strides=(2, 2, 2), kernel_initializer="he_normal",
                             padding='same', kernel_regularizer = w_regularizer)(block1)
    norm = BatchNormalization()(conv2_left)
    elu = ELU()(norm)
    elu = MaxPooling3D(pool_size=3, strides=2)(elu)
    block2=Dropout(drop_rate) (elu)




    # Introduce Middle Flow (separable convolutions with a residual connection)
    conv_mid_1 = mid_flow(block2, drop_rate, w_regularizer, filters=48)

    # Split channels for grouped-style convolution
    conv_mid_1_1 = Lambda(lambda x: x[:, :, :, :, :24])(conv_mid_1)   #48到24
    conv_mid_1_2 = Lambda(lambda x: x[:, :, :, :, 24:])(conv_mid_1)

    #conv5_left = _conv_bn_relu_pool_drop(24, 3, 4, 3, w_regularizer=w_regularizer, drop_rate=drop_rate, pool=True)(
        #conv_mid_1_1)
    conv5_left = Conv3D(24, (3, 4, 3), strides=(1, 1, 1), kernel_initializer="he_normal",
                             padding='same', kernel_regularizer = w_regularizer)(conv_mid_1_1)
    norm = BatchNormalization()(conv5_left)
    elu = ELU()(norm)
    elu = MaxPooling3D(pool_size=3, strides=2)(elu)
    conv5_left=Dropout(drop_rate) (elu)

    #conv5_right = _conv_bn_relu_pool_drop(24, 3, 4, 3, w_regularizer=w_regularizer, drop_rate=drop_rate, pool=True)(
        #conv_mid_1_2)
    conv5_right = Conv3D(24, (3, 4, 3), strides=(1, 1, 1), kernel_initializer="he_normal",
                             padding='same', kernel_regularizer = w_regularizer)(conv_mid_1_2)
    norm = BatchNormalization()(conv5_right)
    elu = ELU()(norm)
    elu = MaxPooling3D(pool_size=3, strides=2)(elu)
    conv5_right=Dropout(drop_rate) (elu)

    #conv6_left = _conv_bn_relu_pool_drop(8, 3, 4, 3, w_regularizer=w_regularizer, drop_rate=drop_rate, pool=True)(
        #conv5_left)
    conv6_left = Conv3D(8, (3, 4, 3), strides=(1, 1, 1), kernel_initializer="he_normal",
                             padding='same', kernel_regularizer = w_regularizer)(conv5_left)
    norm = BatchNormalization()(conv6_left)
    elu = ELU()(norm)
    elu = MaxPooling3D(pool_size=3, strides=2)(elu)
    conv6_left=Dropout(drop_rate) (elu)

    #conv6_right = _conv_bn_relu_pool_drop(8, 3, 4, 3, w_regularizer=w_regularizer, drop_rate=drop_rate, pool=True)(
        #conv5_right)
    conv6_right = Conv3D(8, (3, 4, 3), strides=(1, 1, 1), kernel_initializer="he_normal",
                             padding='same', kernel_regularizer = w_regularizer)(conv5_right)
    norm = BatchNormalization()(conv6_right)
    elu = ELU()(norm)
    elu = MaxPooling3D(pool_size=3, strides=2)(elu)
    conv6_right=Dropout(drop_rate) (elu)

    conv6_concat = concatenate([conv6_left, conv6_right], axis=-1)

    # Flatten 3D conv network representations
    flat_conv_6 = Reshape((np.prod(K.int_shape(conv6_concat)[1:]),))(conv6_concat)

    # 2-layer Dense network for clinical features
    # vol_fc1 = _fc_bn_relu_drop(32, w_regularizer=w_regularizer,
    # drop_rate=drop_rate)(clinical_inputs)

    # flat_volume = _fc_bn_relu_drop(10, w_regularizer=w_regularizer,
    # drop_rate=drop_rate)(vol_fc1)

    # Combine image and clinical features embeddings

    fc1 = _fc_bn_relu_drop(10, w_regularizer, drop_rate=drop_rate)(flat_conv_6)
    # flat = concatenate([fc1, flat_volume])

    # Final 4D embedding
    # fc2 = _fc_bn_relu_drop(4, w_regularizer, drop_rate=drop_rate)(flat) #AD/HC, pMCI/sMCI
    #fc2 = _fc_bn_relu_drop(2, w_regularizer, drop_rate=drop_rate)(fc1)  #
    fc2 =Dense(nb_classes, activation='softmax')(fc1)

    return fc2


def _fc_bn_relu_drop(units, w_regularizer=None, drop_rate=0., name=None):
    # Defines Fully connected block (see fig. 3 in paper)
    def f(input):
        fc = Dense(units=units, activation='linear', kernel_regularizer=w_regularizer, name=name)(
            input)  # was 2048 initially
        fc = BatchNormalization()(fc)
        fc = ELU()(fc)
        fc = Dropout(drop_rate)(fc)
        return fc

    return f


def _conv_bn_relu_pool_drop(filters, height, width, depth, strides=(1, 1, 1), padding='same', w_regularizer=None,
                            drop_rate=None, name=None, pool=False):
    # Defines convolutional block (see fig. 3 in paper)
    def f(input):
        conv = Conv3D(filters, (height, width, depth),
                      strides=strides, kernel_initializer="he_normal",
                      padding=padding, kernel_regularizer=w_regularizer, name=name)(input)
        norm = BatchNormalization()(conv)
        elu = ELU()(norm)
        if pool == True:
            elu = MaxPooling3D(pool_size=3, strides=2)(elu)
        return Dropout(drop_rate)(elu)

    return f


def _sepconv_bn_relu_pool_drop(filters, height, width, depth, strides=(1, 1, 1), padding='same', depth_multiplier=1,
                               w_regularizer=None,
                               drop_rate=None, name=None, pool=False):
    # Defines separable convolutional block (see fig. 3 in paper)
    def f(input):
        sep_conv = SeparableConv3D(filters, (height, width, depth),
                                   strides=strides, depth_multiplier=depth_multiplier, kernel_initializer="he_normal",
                                   padding=padding, kernel_regularizer=w_regularizer, name=name)(input)
        sep_conv = BatchNormalization()(sep_conv)
        elu = ELU()(sep_conv)
        if pool == True:
            elu = MaxPooling2D(pool_size=3, strides=2, padding='same')(elu)
        return Dropout(drop_rate)(elu)

    return f


def mid_flow(x, drop_rate, w_regularizer, filters=48):
    #分离卷积加残差模块，对应的96个通道相加
    #这里因为从没用JD,所以卷积数量改为48
    # 3 consecutive separable blocks with a residual connection (refer to fig. 4)
    residual = x
    x = _sepconv_bn_relu_pool_drop(filters, 3, 3, 3, padding='same', depth_multiplier=1, drop_rate=drop_rate,
                                   w_regularizer=w_regularizer)(x)
    x = _sepconv_bn_relu_pool_drop(filters, 3, 3, 3, padding='same', depth_multiplier=1, drop_rate=drop_rate,
                                   w_regularizer=w_regularizer)(x)
    x = _sepconv_bn_relu_pool_drop(filters, 3, 3, 3, padding='same', depth_multiplier=1, drop_rate=drop_rate,
                                   w_regularizer=w_regularizer)(x)
    x = add([x, residual])
    return x





w_regularizer = regularizers.l2(5e-5)
drop_rate = 0.1


#filepath = './ad_nc_pretrain_mci_convert_weight.hdf5'
#pre_train_path= './ad_nc_pretrain_mci_convert_weight.hdf5'
filepath='./XAlex3D_mci_convert_weight.hdf5'
def get_callbacks(filepath, patience=2):
    lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=1e-5, patience=patience, verbose=1, min_lr = 0.00001)
    msave = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True) #after every epoch save model to filepath
    #earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=patience*3+2, verbose=1, mode='auto')
    return [lr_reduce, msave]#, earlystop]

callbacks = get_callbacks(filepath)
mri=Input (shape = (image_shape)) # 需要用Input对输入维度包装成tensor形式
predictions = XAlex3D(mri, nb_classes, w_regularizer, drop_rate)
model=Model(inputs=mri, outputs=predictions)
#model =load_model(pre_train_path)
#model=lenet3d(image_shape, nb_classes) #即使进行了修改，网络依然无法本地运行
#model=c3d(image_shape, nb_classes)   # 过拟合很严重。9.12与9.13测试了两种二分类，效果也很差，不确定是数据的问题还是网络的问题
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