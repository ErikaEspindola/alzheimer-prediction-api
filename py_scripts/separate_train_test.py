# Separa os dados para treinamento e teste
import shutil
import glob
import os

os.makedirs('/home/erika/dataset/train/CN')
os.makedirs('/home/erika/dataset/train/AD')
os.makedirs('/home/erika/dataset/train/MCI')

os.makedirs('/home/erika/dataset/test/CN')
os.makedirs('/home/erika/dataset/test/AD')
os.makedirs('/home/erika/dataset/test/MCI')

def count_files(path):
    i = 0
    for file in glob.iglob(path, recursive=False):
        i += 1
    return i

def separate_train_test(count, path, class_name):
    train = count * 0.8

    for filename in glob.iglob(path, recursive=False):
        nii_name = filename.split('/')[len(filename.split('/')) - 1]
        if(count_files('/home/erika/dataset/train/' + class_name + '/**') <= train):
            shutil.copy2(filename, '/home/erika/dataset/train/' + class_name + '/' + nii_name)
        else:
            shutil.copy2(filename, '/home/erika/dataset/test/' + class_name + '/' + nii_name)



count_cn  = count_files('/home/erika/dataset/CN/**')
count_mci = count_files('/home/erika/dataset/MCI/**')
count_ad  = count_files('/home/erika/dataset/AD/**')

separate_train_test(count_cn , '/home/erika/dataset/CN/**' , 'CN')
separate_train_test(count_mci, '/home/erika/dataset/MCI/**', 'MCI')
separate_train_test(count_ad , '/home/erika/dataset/AD/**' , 'AD')