# Organiza as pastas com os arquivos .nii e .xml
import os
import glob
import shutil
import collections

xml_list = []
nifti_list = []
folder_list = []

class File(object):
    def __init__(self, file_name, file_id):
        self.file_name = file_name
        self.file_id = file_id

class Folder(object):
    def __init__(self, nii_name, xml_name, id):
        self.nii_name = nii_name
        self.xml_name = xml_name
        self.id = id

def split_filename(filename):
    return filename.split('_')[len(filename.split('_')) - 1].split('.')[0]

def set_list(path):
    lst = []
    for filename in glob.iglob(path, recursive=False):
        lst.append(File(filename, split_filename(filename)))
    return lst

def join_xml_nii():
    for nii in nifti_list:
        for xml in xml_list:
            if nii.file_id == xml.file_id:
                folder_list.append(Folder(nii.file_name, xml.file_name, nii.file_id))
                break

def create_folders():
    for folder in folder_list:
        os.makedirs('/home/erika/New_ADNI/' + folder.id)
        shutil.copy2(folder.nii_name, '/home/erika/New_ADNI/' + folder.id)
        shutil.copy2(folder.xml_name, '/home/erika/New_ADNI/' + folder.id)

nifti_list = set_list('/home/erika/ADNI/**/**/**/**/*.nii')

xml_list = set_list('/home/erika/ADNI_Complete/ADNI1_Complete_1Yr_1.5T_metadata/ADNI/*.xml')

join_xml_nii()

create_folders()