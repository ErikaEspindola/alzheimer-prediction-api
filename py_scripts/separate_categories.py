# Cria pastas separadas com as fotos de AD, CN ou MCI

import xml.etree.ElementTree as ET
import shutil
import glob
import os

os.makedirs('/home/erika/dataset/CN')
os.makedirs('/home/erika/dataset/AD')
os.makedirs('/home/erika/dataset/MCI')

for filename in glob.iglob('/home/erika/New_ADNI/**/*.xml', recursive=False):
    tree = ET.parse(filename)
    root = tree.getroot()

    for xml in root.findall('project/subject/researchGroup'):
        nii_file = filename.split('.')[0] + '.nii'
        nii_name = filename.split('/')[4:5][0] + '.nii'

        shutil.copy2(nii_file, '/home/erika/dataset/' + xml.text + '/' + nii_name)