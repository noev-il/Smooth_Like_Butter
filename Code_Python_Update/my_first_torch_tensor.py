#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 22:06:30 2023

@author: aux
"""

import json
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
import torch

file_new = [r'/Users/aux/Downloads/JSON Packages/A1_Silk_Sonic_Intro_Analysis.json', 
            r'/Users/aux/Downloads/JSON Packages/A2_After_The_Storm_Analysis.json',
            r'/Users/aux/Downloads/JSON Packages/E1_1997_Brock_Hampton.json',
            r'/Users/aux/Downloads/JSON Packages/E2_Bean_Kobe_Uzi.json',
            r'/Users/aux/Downloads/JSON Packages/F1_Grace_Lil_Baby.json',
            r'/Users/aux/Downloads/JSON Packages/F2_Location_Playboi_Carti.json']

with open(file_new[0], 'r') as f:
    A1 = json.loads(f.read())
    
section_A1 = (A1['sections'])
A1_Data = pd.DataFrame(section_A1)
A1_section_cleaned = pd.json_normalize(A1_Data['duration'])
print(A1_Data)

##################play new juice world aaaaaaaaaa

##prepare data # then move # to # next # stage_>_>_>_>__>_>_<_
DATA_FRAME = []
tensors = []
Dictionary = {'start': [], 'duration': [], 'confidence': [], 'loudness': [], 'tempo': [], 'tempo_confidence': [], 'key': [], 'key_confidence': [], 'mode': [], 'mode_confidence': [], 'time_signature': [], 'time_signature_confidence': []}
for x in range(len(file_new)):
    with open(file_new[x], 'r') as f:
        sections = json.loads(f.read())
    sections_ = sections['sections']
    
    DATA_FRAME += [pd.DataFrame(sections_)]
    DATA_FRAME[x].columns = ['start', 'duration', 'confidence', 'loudness', 'tempo', 'tempo_confidence', 'key', 'key_confidence', 'mode', 'mode_confidence', 'time_signature', 'time_signature_confidence']
    
    ##Now Tensor Each Single Column
    Dictionary['start'] += [torch.tensor(DATA_FRAME[x]['start'].values)]
    Dictionary['duration'] += [torch.tensor(DATA_FRAME[x]['duration'].values)]
    Dictionary['confidence'] += [torch.tensor(DATA_FRAME[x]['confidence'].values)]
    Dictionary['loudness'] += [torch.tensor(DATA_FRAME[x]['loudness'].values)]
    Dictionary['tempo'] += [torch.tensor(DATA_FRAME[x]['tempo'].values)]
    Dictionary['tempo_confidence'] += [torch.tensor(DATA_FRAME[x]['tempo_confidence'].values)]
    Dictionary['key'] += [torch.tensor(DATA_FRAME[x]['key'].values)]
    Dictionary['key_confidence'] += [torch.tensor(DATA_FRAME[x]['key_confidence'].values)]
    Dictionary['mode'] += [torch.tensor(DATA_FRAME[x]['mode'].values)]
    Dictionary['mode_confidence'] += [torch.tensor(DATA_FRAME[x]['mode_confidence'].values)]
    Dictionary['time_signature'] += [torch.tensor(DATA_FRAME[x]['time_signature'].values)]
    Dictionary['time_signature_confidence'] += [torch.tensor(DATA_FRAME[x]['time_signature_confidence'].values)]
    
###coding master mind boii
###################tensor turning into pairs

    
        
