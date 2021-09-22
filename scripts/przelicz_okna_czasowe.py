# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 00:23:10 2021

@author: d07321ow
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

module_path = r'C:\Users\d07321ow\Google Drive\GUT\PM_v1\smieciarki'
if module_path not in sys.path:
    sys.path.append(module_path)
    
import funkcje as f
import importlib
importlib.reload(f)


from os import listdir
from os.path import isfile, join

vehicle_id = '20269'

path_to_vehicle_data =r'C:\Users\d07321ow\Google Drive\GUT\Zoeller\pomiary_na_smieciarce\data_smieciarki' + '\\' + vehicle_id
path_to_save_okna = r'C:\Users\d07321ow\Google Drive\GUT\PM_v1\smieciarki\data\{}\podsumowanie_dnia'.format(vehicle_id)

# tbc