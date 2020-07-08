import numpy as np
import csv
import os
import pdb
import sys
import argparse


## generic functions
def write_to_csv(text,file_name):
    with open(file_name, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, lineterminator='\n')
        writer.writerow(text)


def make_dir(dir_name):
    if(not os.path.exists(dir_name)):
        os.mkdir(dir_name)


## malmo specific
def get_image_frame(video_frame, height, width, depth=False):                                                   
    pixels = video_frame.pixels
    img = Image.frombytes('RGB', (height, width), bytes(pixels))
    frame = np.array(img.getdata()).reshape(height, width,-1)
    return frame


def getMissionXML(mission_file):
    with open(mission_file, 'r') as f:
        print("Loading mission from %s" % mission_file)
        mission_xml = f.read()
    return mission_xml
