# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 23:32:10 2017

@author: Aditi
"""

from preprocess import Preprocess
import os

process = Preprocess() 
script_dir = os.path.dirname("") #<-- absolute dir the script is in
process.xls_to_txt('training-Obama-Romney-tweets.xlsx','Obama_data.txt','Romney_data.txt')
print("Text file saved")           
process.clean_sets('Romney_data_cleaned.txt','Romney_data.txt','Obama_data_cleaned.txt','Obama_data.txt')
