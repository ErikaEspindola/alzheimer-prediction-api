import os
import numpy as np
import nibabel as nib
from nibabel.testing import data_path
import random as rd
import fsl.Fsl_helper as fslh

class Recorta_ROI:

    @staticmethod
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
    return region in [180, 109, 97, 96, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 7, 8, 9, 10, 11, 12]
