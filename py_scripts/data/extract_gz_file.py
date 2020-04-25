import os
import gzip
import glob
import shutil

def extract():
    for filename in glob.iglob('/home/erika/NormalizacaoCerebro/**/*.nii.gz', recursive=False):
        with gzip.open(filename, 'rb') as f_in:
            with open(filename + '_new', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

def remove_file():
    for filename in glob.iglob('/home/erika/NormalizacaoCerebro/**/*.nii.gz', recursive=False):
        os.remove(filename)

def rename_file():
    for filename in glob.iglob('/home/erika/NormalizacaoCerebro/**/**', recursive=False):
        name = filename.split('.gz_new')[0]
        os.rename(filename, name)

rename_file()