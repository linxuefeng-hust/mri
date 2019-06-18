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
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet169, DenseNet121
#from keras.applications.densenet import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
#from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import PIL
from PIL import Image as pil_image
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.utils.vis_utils import plot_model
import pydot_ng as pydot
from sklearn.metrics import roc_curve, auc  ###计算roc和auc

batch_size = 4
num_classes = 3

time_amount=5# 针对每个人的一个时间段的数据 0: time_amount-1

#cur_filepath = r'D:/AD_Norm/12_12_time5_NC_AD_train_data.mat'   #12.12 这里使用的数据比例是train_num = math.ceil(file_count * 1)，AD与NC的测试
#final#  cur_filepath = r'D:/AD_Norm/11_30_time5_NC_MCI_train_data.mat' #mci_nc final#  #用11_30_time5_NC_MCI_train_data.mat来训练，用12_13_time5_NC_MCI_train_data来测试
#cur_filepath = r'D:/AD_Norm/12_13_time5_NC_MCI_train_data.mat'
#cur_filepath = r'D:/AD_Norm/11_30_time5_AD_MCI_train_data.mat'
#cur_filepath = r'D:/AD_Norm/12_21_time5_AD_MCI_train_data.mat' #不是
cur_filepath =r'D:/AD_Norm/12_25_paint_roc_time5_AD_MCI_train_data.mat' #
train_data = scipy.io.loadmat(cur_filepath) #读入存放数据的字典
'''
x_test = []
y_test = []
x_test.append(train_data['x_train'])
x_test.append(train_data['x_test'])
x_test = np.array(x_test)
y_test.extend(np.array(train_data['y_train']))
y_test.extend(np.array(train_data['y_test']))
y_test = np.array(y_test)
'''

x_test = np.array(train_data['x_test'])#x_test = np.array(train_data['x_test'])
y_test = np.array(train_data['y_test'])#y_test = np.array(train_data['y_test'])
img_rows, img_cols, img_depth = 61, 73, 61
if K.image_data_format() == 'channels_first':
    print("i am first")
    x_test = x_test.reshape(x_test.shape[0], time_amount, 1, img_rows, img_cols, img_depth)
    input_shape = (time_amount, 1, img_rows, img_cols, img_depth)

else:
    x_test = x_test.reshape(x_test.shape[0], time_amount, img_rows, img_cols, img_depth, 1)
    y_test = y_test.reshape((-1, 1))
    input_shape = (time_amount, img_rows, img_cols, img_depth, 1)

x_test = x_test.astype('float32')
#y_test = keras.utils.to_categorical(y_test, num_classes)   # y_test不需要修改成one-hot标签，因为roc不支持
print(x_test.shape[0], 'test samples')

#filepath1 = './11_30_time5_ad_nc_weight_100_9737.hdf5' # AD与NC分类最终展示的结果所用的权重
#filepath1='./12_13_c3d_lstm_time5_mci_nc_weight_9211.hdf5'  #mci nc fianl#
#filepath1 = './11_30_time5_ad_mci_weight_9474.hdf5'   # 12.20 测试用例
#filepath1 ='./12_21_time5_mci_ad_weight_8421.hdf5' #0.97 搭配D:/AD_Norm/12_25_paint_roc_time5_AD_MCI_train_data.mat
#filepath1 = './12_21_time5_mci_ad_weight_dense1024_8684.hdf5'
filepath1 ='./12_21_time5_mci_ad_weight.hdf5'
model=load_model(filepath1)
prediction=model.predict(x_test,batch_size=4,)
#y_score = np.argmax(prediction, axis=1)
y_score = prediction[:,1]
fpr1, tpr1, threshold=roc_curve(y_test, y_score)
roc_auc1 =auc(fpr1,tpr1)

### 3d fmri data #############################
#cur_filepath = r'D:/AD_Norm/12_12_nc_ad_train_3d_data.mat'  # 12.12 这里使用的数据比例是train_num = math.ceil(file_count * 1)
#final# cur_filepath = r'D:/AD_Norm/12_13_nc_mci_train_3d_data.mat' #final nc mci#
#cur_filepath = r'D:/AD_Norm/12_21_ad_mci_train_3d_data.mat' # 0.94太高排除
cur_filepath = r'D:/AD_Norm/12_27_test_roc_ad_mci_train_3d_data.mat' #0.94搭配nc mci 12_14_3d_c3d_nc_mci_weight.hdf5
train_data = scipy.io.loadmat(cur_filepath)  # 读入存放数据的字典
x_test = np.array(train_data['x_test'])#x_test = np.array(train_data['x_test'])
y_test = np.array(train_data['y_test'])#y_test = np.array(train_data['y_test'])
img_rows, img_cols, img_depth = 61, 73, 61
if K.image_data_format() == 'channels_first':
    print("i am first")
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols, img_depth)
    input_shape = ( 1, img_rows, img_cols, img_depth)

else:
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_depth, 1)
    y_test = y_test.reshape((-1, 1))
    input_shape = ( img_rows, img_cols, img_depth, 1)

x_test = x_test.astype('float32')
print(x_test.shape[0], 'test samples')

#filepath2 = './12_12_3d_c3d_ad_nc_weight.hdf5'#filepath2 = './12_6_3d_c3d_ad_nc_weight_8947.hdf5'
#filepath2 = './12_14_3d_c3d_nc_mci_weight.hdf5'  # 在绘制ad_mci的roc曲线时使用的模型权重也是这个
filepath2 = './12_19_3d_c3d_ad_mci_weight_8684.hdf5' # 12.20 测试
model = load_model(filepath2)
prediction=model.predict(x_test,batch_size=4,)
#y_score = np.argmax(prediction, axis=1)
y_score = prediction[:,1]
fpr2, tpr2, threshold=roc_curve(y_test, y_score)
roc_auc2 =auc(fpr2,tpr2)


##### picture fmri data ##############

#VALID_DIR = r'D:\ADNI_clean\6_24\validation'
VALID_DIR = r'D:\ADNI_clean\6_24\validate'
RANDOM_STATE = 2018
learning_rate = 0.00001  # 0.003 # change to lr = 1e-4
NB_CLASSES = 2
BATCH_SIZE =8
IN_WIDTH, INT_HEIGHT = 61, 73
y_test=[]
x_test=[]

ad_path = os.path.join(VALID_DIR,'AD')
for file in os.listdir(ad_path):
    ad_file_path = os.path.join(ad_path, file)
    img = Image.open(ad_file_path)
    img = img.resize((INT_HEIGHT, IN_WIDTH), PIL.Image.ANTIALIAS)
    img = img_to_array(img)
    img = preprocess_input(img)
    x_test.append(img)
    y_test.append(0)  # 谁第一个加入，谁就标0

nc_path = os.path.join(VALID_DIR,'MCI')
for file in os.listdir(nc_path):
    nc_file_path = os.path.join(nc_path, file)
    img = Image.open(nc_file_path)
    img = img.resize((INT_HEIGHT, IN_WIDTH), PIL.Image.ANTIALIAS)
    img = img_to_array(img)
    img = preprocess_input(img)
    x_test.append(img)
    y_test.append(1)

x_test, y_test = shuffle(x_test, y_test, random_state=1)
x_test = np.array(x_test)
y_test = np.array(y_test)
'''
valid_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)
valid_generator = valid_datagen.flow_from_directory(
    directory=VALID_DIR,
    target_size=(IN_WIDTH, INT_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    seed=2018,
    interpolation='antialias',
    # interpolation=PIL.Image.ANTIALIAS
)
'''


#filepath3 = './12_5_vgg19_picture_ad_nc_weight_9500.hdf5' # 画AD-NC的ROC曲线时使用的是这个数据
#filepath3 = './12_13_vgg19_picture_mci_nc_weight.hdf5' # mci nc final
#filepath3='./12_12_vgg19_picture_ad_nc_weight.hdf5'
filepath3 = './12_21_vgg19_picture_ad_mci_weight.hdf5'
model = load_model(filepath3)
prediction=model.predict(x_test)
#y_score = np.max(prediction, axis=1) # 改成max
y_score = prediction[:,1]   # NC的标签为1，以NC为正例，这个弄清楚非常重要
fpr3, tpr3, threshold=roc_curve(y_test, y_score)
roc_auc3 =auc(fpr3,tpr3)


#filepath4 = './12_5_resnet50_picture_ad_nc_weight_95.hdf5'
#filepath4 = './12_12_resnet50_picture_ad_nc_weight.hdf5'  # 12.12这个是画ROC时用到的
#filepath4 = './12_13_resnet50_picture_mci_nc_weight.hdf5' # mci nc final
filepath4 = './12_21_resenet50_picture_ad_mci_weight.hdf5'

model = load_model(filepath4)
prediction=model.predict(x_test)
#y_score = np.argmax(prediction, axis=1)
y_score = prediction[:,1]
fpr4, tpr4, threshold=roc_curve(y_test, y_score)
roc_auc4 =auc(fpr4,tpr4)


#filepath5 = './12_12_densenet_picture_ad_nc_weight.hdf5' # 这个好结果是使用densenet121得到的
#filepath5 = './12_13_densenet121_picture_mci_nc_weight.hdf5'  # mci nc final
filepath5 = './12_21_densenet121_picture_ad_mci_weight.hdf5'
model = load_model(filepath5)
prediction=model.predict(x_test)
#y_score = np.argmax(prediction, axis=1)
y_score = prediction[:,1]
fpr5, tpr5, threshold=roc_curve(y_test, y_score)
roc_auc5 =auc(fpr5,tpr5)

'''
#filepath5 = './12_5_densenet169_picture_ad_nc_weight.hdf5'
filepath5 = './12_12_vgg19_picture_ad_nc_weight.hdf5'
model = load_model(filepath5)
#model.summary()
prediction=model.predict(x_test)
#y_score = np.argmax(prediction, axis=1)
y_score = prediction[:,1]
fpr5, tpr5, threshold=roc_curve(y_test, y_score)
roc_auc5 =auc(fpr5,tpr5)
'''


filepath6 = './4_18_res_brain_picture_ad_mci_weight.hdf5'
model = load_model(filepath6)
prediction=model.predict(x_test)
y_score = prediction[:,1]
fpr6, tpr6, threshold=roc_curve(y_test, y_score)
roc_auc6 =auc(fpr6,tpr6)






############ plot roc curve ###################
#plt.figure()
lw = 2 # 线的宽度
plt.figure(figsize=(6,6))
plt.grid(True)
plt.plot(fpr1, tpr1, color='darkred',
         lw=lw, label='c3d-lstm ROC curve (area = %0.2f)' % roc_auc1) ###假正率为横坐标，真正率为纵坐标做曲线

plt.plot(fpr2, tpr2, color='darkorange',
         lw=lw, label='c3d ROC curve (area = %0.2f)' % roc_auc2) ###假正率为横坐标，真正率为纵坐标做曲线

plt.plot(fpr3, tpr3, color='darkblue',
         lw=lw, label='vgg19 ROC curve (area = %0.2f)' % roc_auc3) ###假正率为横坐标，真正率为纵坐标做曲线

plt.plot(fpr4, tpr4, color='darkgreen',
         lw=lw, label='resnet50 ROC curve (area = %0.2f)' % roc_auc4) ###假正率为横坐标，真正率为纵坐标做曲线

plt.plot(fpr5, tpr5, color='purple',
         lw=lw, label='densenet121 ROC curve (area = %0.2f)' % roc_auc5) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot(fpr6, tpr6, color='red',
         lw=lw, label='brain networkss model ROC curve (area = %0.2f)' % roc_auc6) ###假正率为横坐标，真正率为纵坐标做曲线

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig("4_18_ad_mci_roc.png")
plt.show()

pause=0