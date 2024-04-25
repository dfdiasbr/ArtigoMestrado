# Tools
import os
import shutil
import numpy as np
from collections import OrderedDict

def del_file(filepath):
    """
    clear all items within a folder
    :param filepath: folder path
    :return:
    """
    
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)