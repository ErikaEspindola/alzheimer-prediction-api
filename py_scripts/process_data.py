import os
import pandas as pd
import nibabel as nib
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

data_dir = '/home/erika/New_ADNI2/'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('labels.csv', index_col=0)

IMG_PX_SIZE = 50
HM_SLICES = 20 # quantidade de fatias para todas as imagens .nii

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]

def mean(l):
    return sum(l)/len(l)

def process_data(patient, labels_df, img_px_size=50, hm_slices=20, visualize=False):
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

    # print(len(new_slices))

    if len(new_slices) == HM_SLICES - 1:
        new_slices.append(new_slices[-1])

    if len(new_slices) == HM_SLICES - 2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])

    if len(new_slices) == HM_SLICES + 2:
        new_val = list(map(mean, zip(*new_slices[HM_SLICES-1], new_slices[HM_SLICES])))
        del new_slices[HM_SLICES]
        new_slices[HM_SLICES-1] = new_val

    if len(new_slices) == HM_SLICES + 1:
        new_val = list(map(mean, zip(*new_slices[HM_SLICES-1], new_slices[HM_SLICES])))
        del new_slices[HM_SLICES]
        new_slices[HM_SLICES-1] = new_val

    if visualize:
        fig = plt.figure()
        for num, each_slice in enumerate(new_slices):
            y = fig.add_subplot(5,4,num+1)
            y.imshow(each_slice)
        plt.show()

# VER ISSO AQUI DIREITO
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

np.save('muchdata-{}-{}-{}.npy'.format(IMG_PX_SIZE, IMG_PX_SIZE, HM_SLICES), much_data)

