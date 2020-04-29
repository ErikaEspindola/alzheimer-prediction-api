import os
import math
from cv2 import cv2
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

data_dir = '/home/erika/New_ADNI2/'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('labels.csv', index_col=0)

IMG_PX_SIZE = 80
HM_SLICES = 100

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]

def mean(l):
    return sum(l)/len(l)

def apply_contrast_and_histogram(img_n):
    img_n = np.array(img_n)
    newImg = []

    for i in img_n:
        img = cv2.normalize(src=i, dst=None, alpha=0, beta=80, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)       

        alpha = 2.8 # Contrast control (1.0-3.0)
        beta = 0 # Brightness control (0-100)
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        equ = cv2.equalizeHist(adjusted)    
        # res = np.hstack((img, adjusted, equ))

        newImg.append(equ.tolist())
    
    return newImg

def process_data(patient, labels_df, img_px_size=100, hm_slices=100, visualize=False):
    label = labels_df.get_value(patient, 'label')
    path = data_dir + patient
    img = nib.load(path + '/' + os.listdir(path)[0])
    slices = img.get_fdata()

    new_slices = []

    slices = [cv2.resize(np.array(each_slice), (IMG_PX_SIZE, IMG_PX_SIZE)) for each_slice in slices]

    chunk_sizes = math.ceil(len(slices) / HM_SLICES)

    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    # Imagens com 30 fatias
    new_slices = new_slices[30:70]

    # new_slices2 = apply_contrast_and_histogram(new_slices)

    if visualize:
        fig = plt.figure()
        for num, each_slice in enumerate(new_slices):
            y = fig.add_subplot(6,8,num+1)
            y.imshow(each_slice)
        plt.show()

    if label == 'CN': label = np.array([0,0,1])
    elif label == 'MCI': label = np.array([0,1,0])
    elif label == 'AD': label = np.array([1,0,0])

    return np.array(new_slices), label

much_data = []

for num, patient in enumerate(patients):
    if num%100 == 0:
        print(num)

    try:
        img_data, label = process_data(patient, labels_df, img_px_size=IMG_PX_SIZE, hm_slices=HM_SLICES)
        much_data.append([img_data, label])
    except KeyError as e:
        print('Dado sem classificação')

np.save('../cnn/dataset-{}-{}-{}.npy'.format(IMG_PX_SIZE, IMG_PX_SIZE, 64), much_data)

