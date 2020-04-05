import xml.etree.ElementTree as ET
import pandas as pd
import shutil
import glob
import csv
import os

csv_array = []

for filename in glob.iglob('/home/erika/New_ADNI/**/*.xml', recursive=False):
    tree = ET.parse(filename)
    root = tree.getroot()
    id_text = str(filename.split('/')[4:5][0])

    for xml in root.findall('project/subject/researchGroup'):
        csv_array.append({'id': id_text, 'label': xml.text})

df = pd.DataFrame(csv_array)
df.to_csv('labels.csv', index=False)
