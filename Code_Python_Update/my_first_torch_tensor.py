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

file_new = [r'/Users/aux/Downloads/Mac_Storage/JSON Packages/A1_Silk_Sonic_Intro_Analysis.json', 
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/A2_After_The_Storm_Analysis.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/B1_Woods_Mac_Miller_Analysis.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/B2_Alotta_Cake_Gunna.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/C1_November_Tyler_The_Creator.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/C2_03_Sainte.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/D1_Thru_My_Hair_Teo.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/D2_Screwed_Up_Teeze.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/E1_1997_Brock_Hampton.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/E2_Bean_Kobe_Uzi.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/F1_Grace_Lil_Baby.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/F2_Location_Playboi_Carti.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/G1_Girl_With_Tattoo_Miguel.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/G2_Break_From_Toronot_PartyNextDoor.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/H1_Only_One_Travis_Scott.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/H2_Low_Down_Lil_Baby.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/I1_The_New_Workout_Plan_Kanye.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/I2_Work_Out_Jcole.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/J1_Highest_In_The_Room_Travis_Scott.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/J2_Tell_Em_Cochise.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/K1_Are_We_Still_Friends_Tyler_The_Creator.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/K2_Hurricane_Kanye.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/L1_A_Boy_Is_A_Gun_Tyler_The_Creator.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/L2_Loose_Change_Brent_Faiyaz.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/M1_Wolvez_Kanye.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/M2_No_Idea_Don_Toliver.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/N1_Girls_Want_Girls_Drake.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/N2_Life_Is_Good_Drake.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/O1_Real_Kendrick.json',
            r'/Users/aux/Downloads/Mac_Storage/JSON Packages/O2_Patience_Lil_Uzi_Vert.json']


with open(file_new[0], 'r') as f:
    A1 = json.loads(f.read())
    
section_A1 = (A1['sections'])
A1_Data = pd.DataFrame(section_A1)
A1_section_cleaned = pd.json_normalize(A1_Data['loudness'])
print(A1_Data)
##################play new juice world aaaaaaaaaa

##prepare data # then move # to # next # stage_>_>_>_>__>_>_<_
DATA_FRAME = []
tensors = []
Dictionary = {'start': [], 'duration': [], 'confidence': [], 'loudness': [], 'tempo': [], 'tempo_confidence': [], 'key': [], 'key_confidence': [], 'mode': [], 'mode_confidence': [], 'time_signature': [], 'time_signature_confidence': []}
new_tensor = None
Song_Tensor = {}
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
    for y in range(len(DATA_FRAME[x])):
        new_tensor = (DATA_FRAME[x]['loudness'][y], DATA_FRAME[x]['tempo'][y], DATA_FRAME[x]['key'][y], DATA_FRAME[x]['mode'][y])
        new_tensor_real = torch.tensor(new_tensor)
        Song_Tensor[f'{y}' + str(file_new[x][47:-5])] = new_tensor_real
        
###coding master mind boii
###################tensor turning into pairs
##print(Dictionary['loudness'][0])
###print(new_tensor_real)

#### 5 Dimensiona Tensors Established
#### Now to Calculate Euclidean Distance of Two Tensors
distanceA1_A2 = []
distanceA1_A2 += [torch.sqrt(torch.sum(torch.pow(torch.subtract(Song_Tensor['3A1_Silk_Sonic_Intro_Analysis'], Song_Tensor['0A2_After_The_Storm_Analysis']), 2), dim=0))]
distanceA1_A2 += [torch.sqrt(torch.sum(torch.pow(torch.subtract(Song_Tensor['3A1_Silk_Sonic_Intro_Analysis'], Song_Tensor['1A2_After_The_Storm_Analysis']), 2), dim=0))]

distanceB1_B2 = [torch.sqrt(torch.sum(torch.pow(torch.subtract(Song_Tensor['11B1_Woods_Mac_Miller_Analysis'], Song_Tensor['0B1_Woods_Mac_Miller_Analysis']), 2), dim=0))]

distanceC1_C2 = [torch.sqrt(torch.sum(torch.pow(torch.subtract(Song_Tensor['9C1_November_Tyler_The_Creator'], Song_Tensor['0C2_03_Sainte']), 2), dim=0))]

distanceD1_D2 = [torch.sqrt(torch.sum(torch.pow(torch.subtract(Song_Tensor['3D1_Thru_My_Hair_Teo'], Song_Tensor['0D2_Screwed_Up_Teeze']), 2), dim=0))]

distanceE1_E2 = [torch.sqrt(torch.sum(torch.pow(torch.subtract(Song_Tensor['7E1_1997_Brock_Hampton'], Song_Tensor['0E2_Bean_Kobe_Uzi']), 2), dim=0))]

distanceF1_F2 = [torch.sqrt(torch.sum(torch.pow(torch.subtract(Song_Tensor['9F2_Location_Playboi_Carti'], Song_Tensor['0F2_Location_Playboi_Carti']), 2), dim=0))]

distanceG1_G2 = [torch.sqrt(torch.sum(torch.pow(torch.subtract(Song_Tensor['5G1_Girl_With_Tattoo_Miguel'], Song_Tensor['0G2_Break_From_Toronot_PartyNextDoor']), 2), dim=0))]

distanceH1_H2 = [torch.sqrt(torch.sum(torch.pow(torch.subtract(Song_Tensor['1H1_Only_One_Travis_Scott'], Song_Tensor['0H2_Low_Down_Lil_Baby']), 2), dim=0))]

distanceI1_I2 = [torch.sqrt(torch.sum(torch.pow(torch.subtract(Song_Tensor['11I1_The_New_Workout_Plan_Kanye'], Song_Tensor['0I2_Work_Out_Jcole']), 2), dim=0))]

distanceJ1_J2 = [torch.sqrt(torch.sum(torch.pow(torch.subtract(Song_Tensor['7J1_Highest_In_The_Room_Travis_Scott'], Song_Tensor['0J2_Tell_Em_Cochise']), 2), dim=0))]

distanceK1_K2 = [torch.sqrt(torch.sum(torch.pow(torch.subtract(Song_Tensor['8K1_Are_We_Still_Friends_Tyler_The_Creator'], Song_Tensor['0K2_Hurricane_Kanye']), 2), dim=0))]

distanceL1_L2 = [torch.sqrt(torch.sum(torch.pow(torch.subtract(Song_Tensor['9L1_A_Boy_Is_A_Gun_Tyler_The_Creator'], Song_Tensor['0L2_Loose_Change_Brent_Faiyaz']), 2), dim=0))]

distanceM1_M2 = [torch.sqrt(torch.sum(torch.pow(torch.subtract(Song_Tensor['9M1_Wolvez_Kanye'], Song_Tensor['0M2_No_Idea_Don_Toliver']), 2), dim=0))]

distanceN1_N2 = [torch.sqrt(torch.sum(torch.pow(torch.subtract(Song_Tensor['9N1_Girls_Want_Girls_Drake'], Song_Tensor['0N2_Life_Is_Good_Drake']), 2), dim=0))]

distanceO1_O2 = [torch.sqrt(torch.sum(torch.pow(torch.subtract(Song_Tensor['19O1_Real_Kendrick'], Song_Tensor['0O2_Patience_Lil_Uzi_Vert']), 2), dim=0))]


new_list = [min(distanceA1_A2), distanceB1_B2, distanceC1_C2, distanceD1_D2, distanceE1_E2, distanceF1_F2, distanceG1_G2, distanceH1_H2, distanceI1_I2, distanceJ1_J2, distanceK1_K2, distanceL1_L2, distanceM1_M2, distanceN1_N2, distanceO1_O2]
print(len(new_list))
        
