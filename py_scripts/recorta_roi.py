import os
import numpy as np
import nibabel as nib
from nibabel.testing import data_path
import random as rd
import Fsl_helper as fslh

def cutBrainMask(fileIn, fileOut):    

    fileNii = nib.load(fileIn)
    mask    = nib.load(fslh.get_fsl_path() + '/' + 'data/atlases/Juelich/Juelich-maxprob-thr25-1mm.nii.gz')

    newFile = nib.Nifti1Image(fileNii.get_fdata(), fileNii.affine)    
    arrFile = newFile.get_fdata()
    arrMask = mask.get_fdata() 

    shape = arrFile.shape 

    for i in range (shape[0]-1):
        for j in range (shape[1]-1):
            for k in range (shape[2]-1):

                if(isROI(arrMask[i][j][k]) == False):
                    arrFile[i][j][k] = 0

    nib.save(newFile, fileOut)

# Aqui que ele vê o numero das labels na mascara pra pegar, o resto ele seta 0
# https://neurovault.org/images/1392/ -> label de cada região dessa máscara

def isROI(region):
    return region in [180, 109, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

cutBrainMask(
    '/home/edson/alzheimer-prediction-api/py_scripts/6663GQ_teste.nii',
    '/home/edson/alzheimer-prediction-api/py_scripts/saida.nii'
)