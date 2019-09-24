import re
import os
import gc
import numpy as np
from tqdm import tqdm

def extract_targets(name):
    '''
    extract target values
    
    str -> list 
    '''
    f = open(name, "r")
    l_targets = [float(i) for i in f.readlines()[2:][0].split('\t')]
    return l_targets

def extract_data(name):
    '''
    extract measurements data with shape (55, 300)

    str -> numpy array
    '''
    f = open(name, "r")
    all_data = np.zeros((55,300))
    for num, i_list in enumerate(f.readlines()[6:]):
        all_data[num,:] = [float(i) for i in i_list.split('\t')]
    return all_data

def extract_data_heads(name):
    '''
    extract planet and solar data, 6 values. 
    
    str -> list 
    '''
    f = open(name, "r")
    return re.findall("\d+\.\d+",  "".join(f.readlines()[:6]))


def data_extractor(size, FOLDER):
    '''
    data main loop
    
    size = number of samples in a data set  
    
    int, str -> numpy array, numpy array
    '''
    all_data = np.zeros((size, 55, 300))
    all_data_head = np.zeros((size, 6))
    for subdir, dirs, files in os.walk(FOLDER):
        for i, file in enumerate(tqdm(sorted(files))):
            all_data[i] = extract_data(FOLDER + file)
            all_data_head[i] = extract_data_heads(FOLDER + file)
    gc.collect()
    return all_data, all_data_head

def targets_extractor(FOLDER):
    '''
    target main loop 
    
    str -> numpy array 
    '''
    all_targets = []
    for subdir, dirs, files in os.walk(FOLDER):
        for file in tqdm(sorted(files)):
            target = extract_targets(FOLDER + file)
            all_targets.append(target)
    return np.array(all_targets)
