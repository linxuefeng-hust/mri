'''
就是将all_group_matlab中的nifti文件转换成npy文件，但是要保留文件名，方便后面根据npy数据的文件名筛选出对应的clinical数据
'''

from nilearn.image import load_img
from nilearn.datasets import load_mni152_template
import os
import gc
import numpy as np
import scipy.io
import pandas as pd
import nilearn
from sklearn.utils import shuffle
import math
import nibabel as nib
import nilearn.image as niimage
# 使用循环或者if语句能更好的实现程序
# 10月1日，对matlab预处理之后的数据进行处理
# 9.15更改文件目录
AD_train_path = r'D:\all_group_matlab\train\AD\T1Img'
CN_train_path = r'D:\all_group_matlab\train\CN\T1Img'
AD_validation_path = r'D:\all_group_matlab\validation\AD\T1Img'
CN_validation_path = r'D:\all_group_matlab\validation\CN\T1Img'
MCI_train_path = r'D:\all_group_matlab\train\MCI\T1Img'
MCI_validation_path = r'D:\all_group_matlab\validation\MCI\T1Img'

MCI_convert_train_path = r'E:\MCI_comparision\train\mci_convert'
MCI_unconvert_train_path = r'E:\MCI_comparision\train\mci_unconvert'
MCI_convert_validation_path = r'E:\MCI_comparision\validation\mci_convert'
MCI_unconvert_validation_path = r'E:\MCI_comparision\validation\mci_unconvert'



ad_train_save_path = r'E:\clinical_allgroup_matlab_adni_generator_std\train\AD'
ad_validation_save_path = r'E:\clinical_allgroup_matlab_adni_generator_std\validation\AD'
cn_train_save_path = r'E:\clinical_allgroup_matlab_adni_generator_std\train\CN'
cn_validation_save_path = r'E:\clinical_allgroup_matlab_adni_generator_std\validation\CN'
mci_train_save_path = r'E:\clinical_allgroup_matlab_adni_generator_std\train\MCI'
mci_validation_save_path = r'E:\clinical_allgroup_matlab_adni_generator_std\validation\MCI'



mci_convert_train_save_path = r'E:\MCI_compa_std\train\convert'
mci_convert_validation_save_path =r'E:\MCI_compa_std\validation\convert'
mci_unconvert_train_save_path = r'E:\MCI_compa_std\train\no_convert'
mci_unconvert_validation_save_path = r'E:\MCI_compa_std\validation\no_convert'



if __name__ == '__main__':
    #give_shape = (256, 256, 192)
    #give_shape = (192, 192 ,160) # 9.18修改
    #give_shape = (128, 128, 96)   # 9.18 修改。减小图片尺寸，以期使用更大的batch
    give_shape = (121, 145, 121)   # 9.26 加入最大值标准化，将维度改为(256, 256, 160) ，将数据存入all_adni_generator


    for dirs in os.listdir(MCI_train_path):
        train_path = os.path.join(MCI_train_path, dirs)
        train_path = os.path.join(train_path, 'mri')
        mri_img = 0
        for file in os.listdir(train_path):
            train_file_path = os.path.join(train_path, file)
            img = nib.load(train_file_path)
            img_arr = img.get_data()
            img_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))
            mri_img += img_arr
            filename = file

        mri_img = mri_img.astype('float32')
        max = np.max(mri_img)
        where_are_nan = np.isnan(mri_img)  # 不去除nan，则在取np.max时会取到nan
        mri_img[where_are_nan] = 0         # don't have to do this, img has been normalized by cat12
        img_name = filename
        full_save_path = os.path.join(mci_train_save_path, img_name)
        np.save(full_save_path, mri_img)


    for dirs in os.listdir(MCI_validation_path):
        validation_path = os.path.join(MCI_validation_path, dirs)
        validation_path = os.path.join(validation_path, 'mri')
        mri_img = 0
        for file in os.listdir(validation_path):
            validation_file_path = os.path.join(validation_path, file)
            img = nib.load(validation_file_path)
            img_arr = img.get_data()
            img_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))
            mri_img += img_arr
            filename = file

        mri_img = mri_img.astype('float32')
        max = np.max(mri_img)
        where_are_nan = np.isnan(mri_img)
        mri_img[where_are_nan] = 0
        img_name = filename
        full_save_path = os.path.join(mci_validation_save_path, img_name)
        np.save(full_save_path, mri_img)
'''



    for dirs in os.listdir(AD_train_path):
        train_path = os.path.join(AD_train_path, dirs)
        train_path = os.path.join(train_path, 'mri')
        mri_img = 0
        for file in os.listdir(train_path):
            train_file_path = os.path.join(train_path, file)
            img = nib.load(train_file_path)
            img_arr = img.get_data()
            img_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))
            mri_img += img_arr
            filename = file

        mri_img = mri_img.astype('float32')
        max = np.max(mri_img)
        where_are_nan = np.isnan(mri_img)  # 不去除nan，则在取np.max时会取到nan
        mri_img[where_are_nan] = 0         # don't have to do this, img has been normalized by cat12
        img_name = filename
        full_save_path = os.path.join(ad_train_save_path, img_name)
        np.save(full_save_path, mri_img)


    for dirs in os.listdir(AD_validation_path):
        validation_path = os.path.join(AD_validation_path, dirs)
        validation_path = os.path.join(validation_path, 'mri')
        mri_img = 0
        for file in os.listdir(validation_path):
            validation_file_path = os.path.join(validation_path, file)
            img = nib.load(validation_file_path)
            img_arr = img.get_data()
            img_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))
            mri_img += img_arr
            filename = file

        mri_img = mri_img.astype('float32')
        max = np.max(mri_img)
        where_are_nan = np.isnan(mri_img)
        mri_img[where_are_nan] = 0
        img_name = filename
        full_save_path = os.path.join(ad_validation_save_path, img_name)
        np.save(full_save_path, mri_img)



    for dirs in os.listdir(CN_train_path):
        train_path = os.path.join(CN_train_path, dirs)
        train_path = os.path.join(train_path, 'mri')
        mri_img = 0
        for file in os.listdir(train_path):
            train_file_path = os.path.join(train_path, file)
            img = nib.load(train_file_path)
            img_arr = img.get_data()
            img_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))
            mri_img += img_arr
            filename = file

        mri_img = mri_img.astype('float32')
        max = np.max(mri_img)
        where_are_nan = np.isnan(mri_img)  # 不去除nan，则在取np.max时会取到nan
        mri_img[where_are_nan] = 0         # don't have to do this, img has been normalized by cat12
        img_name = filename
        full_save_path = os.path.join(cn_train_save_path, img_name)
        np.save(full_save_path, mri_img)

    for dirs in os.listdir(CN_validation_path):
        validation_path = os.path.join(CN_validation_path, dirs)
        validation_path = os.path.join(validation_path, 'mri')
        mri_img = 0
        for file in os.listdir(validation_path):
            validation_file_path = os.path.join(validation_path, file)
            img = nib.load(validation_file_path)
            img_arr = img.get_data()
            img_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))
            mri_img += img_arr
            filename = file

        mri_img = mri_img.astype('float32')
        max = np.max(mri_img)
        where_are_nan = np.isnan(mri_img)
        mri_img[where_are_nan] = 0
        img_name = filename
        full_save_path = os.path.join(cn_validation_save_path, img_name)
        np.save(full_save_path, mri_img)

'''






