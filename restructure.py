import pandas as pd
import os

path_to_dir='/workspace/DATA/C. Diabetic Retinopathy Grading'

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('{}a. Training Set'.format(path_to_dir)) if isfile(join('{}a. Training Set'.format(path_to_dir), f))]

df=pd.read_csv('{}a. DRAC2022_ Diabetic Retinopathy Grading_Training Labels.csv'.format(path_to_dir))

for i in df['DR grade'].unique():
    directory = str(i)

    # Parent Directory path
#     parent_dir = "/home/ammar/Desktop/LMU/ADL/data/DRAC_data/"

    # Path
    path = os.path.join(path_to_dir, directory)

    os.mkdir(path)

for i,v in df.groupby(['DR grade']):
    print(i)
    for k,val in v.iterrows():
        os.replace("{}{}/{}".format(path_to_dir,'a. Training Set',val['image name']), "{}{}/{}".format(path_to_dir,i,val['image name']))