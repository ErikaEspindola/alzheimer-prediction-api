import gzip
import glob
import shutil

def extract(path):
    name = ''
    for filename in glob.iglob(path + '.gz', recursive=False):
        with gzip.open(filename, 'rb') as f_in:
            name = filename.split('.')[0] + '.nii'
            with open(name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    return name