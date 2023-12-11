import json

file_new = [r'/Users/aux/Downloads/JSON Packages/A1_Silk_Sonic_Intro_Analysis.json', 
            r'/Users/aux/Downloads/JSON Packages/A2_After_The_Storm_Analysis.json',
            r'/Users/aux/Downloads/JSON Packages/E1_1997_Brock_Hampton.json',
            r'/Users/aux/Downloads/JSON Packages/E2_Bean_Kobe_Uzi.json',
            r'/Users/aux/Downloads/JSON Packages/F1_Grace_Lil_Baby.json',
            r'/Users/aux/Downloads/JSON Packages/F2_Location_Playboi_Carti.json']

A1 = None
A2 = None
E1 = None
E2 = None
F1 = None
F2 = None
tempo_A1 = []
key_A1 = []
mode_A1 = []
tempo_A2 = []
key_A2 = []
mode_A2 = []

with open(file_new[0], 'r') as f:
    A1 = json.loads(f.read())
    ###remember to differentiate 1 vs. 2
    section_A1 = (A1['sections'])
    ###remember to differentiate 1 vs. 2 If ?1 or if ?2:
    for y in range(len(section_A1)):
        if section_A1[y]['start'] <= 6:
            tempo_A1 += [section_A1[y]['tempo']]
            key_A1 += [section_A1[y]['key']]
            mode_A1 += [section_A1[y]['mode']]

with open(file_new[1], 'r') as f:
    A2 = json.loads(f.read())
    ###remember to differentiate 1 vs. 2
    section_A2 = (A2['sections'])
    ###remember to differentiate 1 vs. 2 If ?1 or if ?2:
    for y in range(len(section_A2)):
        length_2 = float(A2['track']['duration']) - 10
        if section_A2[-1]['duration'] > 20:
            tempo_A2 += [section_A2[-1]['tempo']]
            key_A2 += [section_A2[-1]['key']]
            mode_A2 += [section_A2[-1]['mode']]
        else:
            if section_A2[y]['start'] >= length_2:
                tempo_A2 += [section_A2[y]['tempo']]
                key_A2 += [section_A2[y]['key']]
                mode_A2 += [section_A2[y]['mode']]
            
            

if len(tempo_A1) > len(tempo_A2):
    run_length = len(tempo_a1) - len(tempo_A2) ### if a1 has more sections than
    for j in range(run_length):
        tempo_A2 += [0] * run_length
elif len(tempo_A2) > len(tempo_A1):
    run_length = len(tempo_A2) - len(tempo_A1)


if len(key_a1) > len(key_a2):
    ###insert
tempo_A12_range = abs(tempo_A1 - tempo_A2)
key_A12_range = abs(key_A1 - key_A2)
if mode_A1 == -1 or mode_A2 == -1:
    mode_A12_range = 0
else:
    mode_A12_range = abs(mode_A1 - mode_A2)
    
print