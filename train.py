import keras
from keras.models import Sequential
from keras.models import save_model,load_model
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
import scipy.io
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras.utils.vis_utils import plot_model
import pydot_ng as pydot
pydot.Dot.create(pydot.Dot())
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

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





batch_size = 4 # batch为8的训练过程收敛效率明显不如batch为2，timeamount改为8之后，就跑不了8个batch了
num_classes =3
epochs = 150 # 300


time_amount=5# 针对每个人的一个时间段的数据 0: time_amount-1
#time_amount=1   #测试3d结构


'''
# x_train, y_train, x_test, y_test = get_basic_data()
x_train, y_train, x_test, y_test = load_3D_from_mat()

# input image dimensions
img_rows, img_cols, img_depth = 52, 58, 52
num_classes=2 # 二分类问题
'''

#x_train,y_train,x_test,y_test=load_easydata.load_easy(time_amount)
# 这里为了节省时间，将time_amount在load_easydata中改为固定值4
#cur_filepath = r'D:/AD_Norm/11_30_time5_NC_MCI_train_data.mat' # 之前使用的这个进行训练
#cur_filepath =r'D:/AD_Norm/12_13_time5_NC_MCI_train_data.mat'
#cur_filepath =r'D:/AD_Norm/11_30_time5_NC_AD_train_data.mat'
#cur_filepath = r'D:/AD_Norm/11_30_time5_AD_MCI_train_data.mat'
#cur_filepath = r'D:/AD_Norm/12_21_time5_AD_MCI_train_data.mat'
cur_filepath = r'D:/AD_Norm/time_5_AD_NC_MCI_train_data.mat'
train_data = scipy.io.loadmat(cur_filepath) #读入存放数据的字典
x_train = np.array(train_data['x_train'])
x_train=x_train.astype('float32')
y_train = train_data['y_train']
y_train=y_train.astype('float32')
x_test = np.array(train_data['x_test'])
y_test = train_data['y_test']


# input image dimensions
img_rows, img_cols, img_depth = 61, 73, 61
#num_classes=2 # 二分类问题



# the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    print("i am first")
    x_train = x_train.reshape(x_train.shape[0], time_amount, 1, img_rows, img_cols, img_depth)
    x_test = x_test.reshape(x_test.shape[0], time_amount, 1, img_rows, img_cols, img_depth)
    input_shape = (time_amount, 1, img_rows, img_cols, img_depth)

else:
    x_train = x_train.reshape(x_train.shape[0], time_amount, img_rows, img_cols, img_depth, 1)
    x_test = x_test.reshape(x_test.shape[0], time_amount, img_rows, img_cols, img_depth, 1)
    y_test = y_test.reshape((-1, 1))
    y_train = y_train.reshape((-1, 1))


    input_shape = (time_amount, img_rows, img_cols, img_depth, 1)

'''
else:
    x_train = x_train.reshape(x_train.shape[0],  img_rows, img_cols, img_depth, 1)
    x_test = x_test.reshape(x_test.shape[0],  img_rows, img_cols, img_depth, 1)
    input_shape = ( img_rows, img_cols, img_depth, 1)kernel_regularizer=regularizers.l2(0.01)
'''






x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


model=c3d_lstm_model.c3d_lstm(input_shape)
#model=c3d_lstm_model.c3d(input_shape)

#filepath = './12_31_time5_ad_nc_mci_gru_weight.hdf5'
filepath = './19_5_7_see_para_amounts_3d_c3d_ad_mci_weight.hdf5'

def get_callbacks(filepath, patience=4):
    lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=1e-5, patience=patience, verbose=1, min_lr = 0.00001)
    msave = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True) #after every epoch save model to filepath
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=patience*3+2, verbose=1, mode='auto')
    return [lr_reduce, msave]#, earlystop]
callbacks = get_callbacks(filepath)

###################
model.summary()
#plot_model(model, to_file='model.png')
learning_rate = 0.00001 # 6.25lr从0.001修改为0.0004
#model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=learning_rate, decay=learning_rate/20, momentum=0.9, nesterov=True), metrics=['accuracy'])
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08), metrics=['accuracy']) # 6.25修改SGD为adam
#history = LossHistory()
#plot_model(model,to_file='D:/ADNI_clean/6_24/untitled/new_time_dis_structure_fmri_c3dlstm.png',show_shapes=True)
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(x_test, y_test), callbacks=callbacks)
#score = model.evaluate(x_test, y_test, verbose=1)
#print(model.metrics_names)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
#history.loss_plot('epoch')
#save_model('D:/AD_Norm/c3dlstm.h5')
