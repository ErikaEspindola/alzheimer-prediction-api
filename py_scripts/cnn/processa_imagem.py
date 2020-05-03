import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2

def save3dBrain(fl, flOut):
    curDir = '/home/erika/NormalizacaoCerebro/I36464/'
    fileIn = curDir + fl
    fileNii = nib.load(fileIn)

    
    img_n = fileNii.get_fdata()
    newImg = []

    for i in img_n:

        img = cv2.normalize(src=i, dst=None, alpha=0, beta=80, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)       
        
        equ = cv2.equalizeHist(img)    
        #res = np.hstack((img, adjusted, equ)) 

        newImg.append(equ.tolist())
    

    newFile = nib.Nifti1Image(np.array(newImg), fileNii.affine)    
    nib.save(newFile, '/home/erika/Desktop/Resultados tcc/' + flOut)

save3dBrain('I36464.nii', 'semcontraste.nii')
# save3dBrain('cn.nii', 'cnProcessada.nii')

