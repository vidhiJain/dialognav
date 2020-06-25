import csv
import os, glob
import random
import numpy as np
import pdb

os.chdir("/home/akshay/Downloads/visdial_data/extracted_frames")
csv_dir = "../csv_files"
if not(os.path.exists(csv_dir)):
    os.mkdir(csv_dir)
files = np.array(glob.glob("*.png"))
# writer.writerow(field_names)
np.random.shuffle(files)

batch_size = 100
batch_counter = 0

for i in range(0,files.shape[0], batch_size):
    file_list = files[i:i+batch_size].tolist()
    with open("../csv_files/file_list_{}.csv".format(batch_counter+1), 'w', newline='') as csv_file:
        wr = csv.writer(csv_file, lineterminator='\n')
        wr.writerow(['image_url'])
        for f in file_list:
            wr.writerow([f])

    batch_counter += 1


