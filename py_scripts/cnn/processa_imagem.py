import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2

def save3dBrain(fl, flOut):
    curDir = 'C:\\Users\\DELL\\Desktop\\ProjetosDeTeste\\tcc\\alzheimer-prediction-api\\py_scripts\\cnn\\'
    fileIn = curDir + fl
    fileNii = nib.load(fileIn)

    
    img_n = fileNii.get_fdata()
    newImg = []

    for i in img_n:

        img = cv2.normalize(src=i, dst=None, alpha=0, beta=80, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)       

        alpha = 2.8 # Contrast control (1.0-3.0)
        beta = 0 # Brightness control (0-100)
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        equ = cv2.equalizeHist(adjusted)    
        #res = np.hstack((img, adjusted, equ)) 

        newImg.append(equ.tolist())
    

        
    newImg = np.array(newImg)
    newFile = nib.Nifti1Image(newImg, fileNii.affine)    
    nib.save(newFile, curDir + flOut)

save3dBrain('ad.nii', 'adProcessada.nii')
save3dBrain('cn.nii', 'cnProcessada.nii')

