
import base64
import requests

import json
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
import torch

class tensor_creator:
    def __init__(self, url_1, url_2):
        self.client_id = "76915a0c79a34e108e08ab4ca0e2605f"
        self.client_secret = "f30900615bad41d0ae950d4e2ad72668"
        self.access_token = None
        api_token = "https://accounts.spotify.com/api/token"
        auth_string = f'{self.client_id}:{self.client_secret}'
        auth_base64 = base64.b64encode(auth_string.encode()).decode('ascii')
        token_data = {
            "grant_type": "client_credentials"
        }
        
        token_headers = {
            "Authorization": f"Basic {auth_base64}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        req = requests.post(api_token, data=token_data, headers=token_headers)
        token_response_data = req.json()
        self.access_token = token_response_data.get('access_token')
        self.url_1 = url_1[31:]
        self.url_2 = url_2[31:]
        
        
    
    def get_track_analysis(self, song_id):
        token_live = self.access_token
        if token_live:
            api_token = f"https://api.spotify.com/v1/audio-analysis/{song_id}"
            header_1 = {
                "Authorization": f"Bearer {token_live}"
            }
            try:
                response = requests.get(api_token, headers=header_1)
                response.raise_for_status()  # Raise an exception for HTTP errors
                json_file = response.json()
                return (json_file)
            except requests.exceptions.RequestException as e:
                print("Request failed:", e)
                print("Response content:", response.content)  # Print response content for debugging
        else:
            print("Unable to retrieve access token.")
    
    def make_tensor(self):
        self.analysis_1 = self.get_track_analysis(self.url_1)
        self.analysis_2 = self.get_track_analysis(self.url_2)
        track_1 = self.analysis_1
        track_2 = self.analysis_2
        DATA_FRAME = []
        tensors = []
        Dictionary = {'start': [], 'duration': [], 'confidence': [], 'loudness': [], 'tempo': [], 'tempo_confidence': [], 'key': [], 'key_confidence': [], 'mode': [], 'mode_confidence': [], 'time_signature': [], 'time_signature_confidence': []}
        new_tensor = None
        Song_Tensor = []
        Song_Tensor_1 = []
        sections_1 = track_1['sections']
        sections_2 = track_2['sections']
        DATA_FRAME += [pd.DataFrame(sections_1)]
        DATA_FRAME += [pd.DataFrame(sections_2)]
        DATA_FRAME[0].columns = ['start', 'duration', 'confidence', 'loudness', 'tempo', 'tempo_confidence', 'key', 'key_confidence', 'mode', 'mode_confidence', 'time_signature', 'time_signature_confidence']
        Dictionary['start'] += [torch.tensor(DATA_FRAME[0]['start'].values)]
        Dictionary['duration'] += [torch.tensor(DATA_FRAME[0]['duration'].values)]
        Dictionary['confidence'] += [torch.tensor(DATA_FRAME[0]['confidence'].values)]
        Dictionary['loudness'] += [torch.tensor(DATA_FRAME[0]['loudness'].values)]
        Dictionary['tempo'] += [torch.tensor(DATA_FRAME[0]['tempo'].values)]
        Dictionary['tempo_confidence'] += [torch.tensor(DATA_FRAME[0]['tempo_confidence'].values)]
        Dictionary['key'] += [torch.tensor(DATA_FRAME[0]['key'].values)]
        Dictionary['key_confidence'] += [torch.tensor(DATA_FRAME[0]['key_confidence'].values)]
        Dictionary['mode'] += [torch.tensor(DATA_FRAME[0]['mode'].values)]
        Dictionary['mode_confidence'] += [torch.tensor(DATA_FRAME[0]['mode_confidence'].values)]
        Dictionary['time_signature'] += [torch.tensor(DATA_FRAME[0]['time_signature'].values)]
        Dictionary['time_signature_confidence'] += [torch.tensor(DATA_FRAME[0]['time_signature_confidence'].values)]
        for y in range(len(DATA_FRAME[0])):
            new_tensor = (DATA_FRAME[0]['loudness'][y], DATA_FRAME[0]['tempo'][y], DATA_FRAME[0]['key'][y], DATA_FRAME[0]['mode'][y])
            new_tensor_real = torch.tensor(new_tensor)
            Song_Tensor += [new_tensor_real]
        DATA_FRAME[1].columns = ['start', 'duration', 'confidence', 'loudness', 'tempo', 'tempo_confidence', 'key', 'key_confidence', 'mode', 'mode_confidence', 'time_signature', 'time_signature_confidence']
        Dictionary['start'] += [torch.tensor(DATA_FRAME[1]['start'].values)]
        Dictionary['duration'] += [torch.tensor(DATA_FRAME[1]['duration'].values)]
        Dictionary['confidence'] += [torch.tensor(DATA_FRAME[1]['confidence'].values)]
        Dictionary['loudness'] += [torch.tensor(DATA_FRAME[1]['loudness'].values)]
        Dictionary['tempo'] += [torch.tensor(DATA_FRAME[1]['tempo'].values)]
        Dictionary['tempo_confidence'] += [torch.tensor(DATA_FRAME[1]['tempo_confidence'].values)]
        Dictionary['key'] += [torch.tensor(DATA_FRAME[1]['key'].values)]
        Dictionary['key_confidence'] += [torch.tensor(DATA_FRAME[1]['key_confidence'].values)]
        Dictionary['mode'] += [torch.tensor(DATA_FRAME[1]['mode'].values)]
        Dictionary['mode_confidence'] += [torch.tensor(DATA_FRAME[1]['mode_confidence'].values)]
        Dictionary['time_signature'] += [torch.tensor(DATA_FRAME[1]['time_signature'].values)]
        Dictionary['time_signature_confidence'] += [torch.tensor(DATA_FRAME[0]['time_signature_confidence'].values)]
        for z in range(len(DATA_FRAME[1])):
            new_tensor = (DATA_FRAME[1]['loudness'][z], DATA_FRAME[1]['tempo'][z], DATA_FRAME[1]['key'][z], DATA_FRAME[1]['mode'][z])
            new_tensor_real = torch.tensor(new_tensor)
            Song_Tensor_1 += [new_tensor_real]
        distanceA1_A2 = torch.sqrt(torch.sum(torch.pow(torch.subtract(Song_Tensor[-1], Song_Tensor_1[0]), 2), dim=0))
        self.distanceA1_A2 = (distanceA1_A2)
    
    def segment_analyzer(self):
        return_value = {"Song_1": None, "Song_2": None}
        segment_1 = self.analysis_1['segments']
        segment_2 = self.analysis_2['segments']
        
        DATA_FRAME_1 = [pd.DataFrame(segment_1[:15])]
        pitches_1 = []
        timbre_1 = []
        pitches_2 = []
        timbre_2 = []
        DATA_FRAME_2 = [pd.DataFrame(segment_2[-15:])]
        DATA_FRAME_1[0].columns = ['start', 'duration', 'confidence', "loudness_start", "loudness_max_time", "loudness_max", "loudness_end", "pitches", "timbre"]
        DATA_FRAME_2[0].columns = ['start', 'duration', 'confidence', "loudness_start", "loudness_max_time", "loudness_max", "loudness_end", "pitches", "timbre"]
        ###return DATA_FRAME_1[0]['pitches'][-1]
        for p in range(12):
            pitches_1 += [torch.tensor(DATA_FRAME_1[0]['pitches'][p])] 
            pitches_2 += [torch.tensor(DATA_FRAME_2[0]['pitches'][p])] 
        for t in range(12):
            timbre_1 += [torch.tensor(DATA_FRAME_1[0]['timbre'][t])] 
            timbre_2 += [torch.tensor(DATA_FRAME_2[0]['timbre'][t])] 
        pitches_dist = []
        timbres_dist =  []
        for g in range(12):
            pitches_dist += [torch.sqrt(torch.sum(torch.pow(torch.subtract(pitches_1[g], pitches_2[g]), 2), dim=0))]
            timbres_dist += [torch.sqrt(torch.sum(torch.pow(torch.subtract(timbre_1[g], timbre_2[g]), 2), dim=0))]
        tensor_values_p = [tensor.item() for tensor in pitches_dist]
        tensor_values_t = [tensor.item() for tensor in timbres_dist]
        average_p = torch.tensor(tensor_values_p).mean()
        average_t = torch.tensor(tensor_values_t).mean()
        return self.distanceA1_A2.item(), average_p.item(), average_t.item()
tc_1 = tensor_creator('https://open.spotify.com/track/4K09vJ27xCOreumtSuU6Ao?si=91c7ad440b0349c9', 'https://open.spotify.com/track/1otG6j1WHNvl9WgXLWkHTo?si=c13240ae95c34160')
tc_2 = tensor_creator('https://open.spotify.com/track/3Qa944OTMZkg8DHjET8JQv?si=55a3c9a0cb664687', 'https://open.spotify.com/track/19QKaApDINxlRSKX3w1xSB?si=a36bfc08f5284519')
tc_3 = tensor_creator('https://open.spotify.com/track/4XDpeWqPADoWRKcUY3dC84?si=f4252385239d4ac6', 'https://open.spotify.com/track/47gzGfR4JfKC6aT5lM0wpn?si=8099e1f4cb2f4293')


{##tc_4 = tensor_creator('', '') 
##tc_5 = tensor_creator('', '') 
####tc_6 = tensor_creator('', '') 
##tc_7 = tensor_creator('', '') 
##tc_8 = tensor_creator('', '') 
##tc_9 = tensor_creator('', '') 
##tc_10 = tensor_creator('', '') 
##tc_11 = tensor_creator('', '') 
##tc_12 = tensor_creator('', '') 
##tc_13 = tensor_creator('', '') 
##tc_14 = tensor_creator('', '') 
###print(tc_1.make_tensor(), tc_2.make_tensor(), tc_3.make_tensor())
}
##tc_1.make_tensor()
##check = tc_1.segment_analyzer()
##print(check)

good_transitions = [
    "https://open.spotify.com/track/4K09vJ27xCOreumtSuU6Ao?si=7bf160671b854579",
    "https://open.spotify.com/track/1otG6j1WHNvl9WgXLWkHTo?si=a3575a900f3a4c6d",
    "https://open.spotify.com/track/3Qa944OTMZkg8DHjET8JQv?si=696ddb4a9daf4a67",
    "https://open.spotify.com/track/19QKaApDINxlRSKX3w1xSB?si=c9229546c6334acb",
    "https://open.spotify.com/track/4XDpeWqPADoWRKcUY3dC84?si=04e5e2519d52466c",
    "https://open.spotify.com/track/47gzGfR4JfKC6aT5lM0wpn?si=0e1b348a726140d4",
    'https://open.spotify.com/track/6BbAFjOCHA1AknMtIu3VjZ?si=8aea2fbe31c84d5e',
                 "https://open.spotify.com/track/3d65swPOxko76ZQL5WEQfH?si=37614ad6886a4e54",
                 "https://open.spotify.com/track/55jQMevNp7aWtiW5LPlPoa?si=1f0e24819e5e48b9",
                 "https://open.spotify.com/track/0IpnZchq8ek2A6pGEP2Qb1?si=678492cafa964419",
                 "https://open.spotify.com/track/01JMnRUs2YOK6DDpdQASGY?si=fb31acb7005141ec",
                 "https://open.spotify.com/track/3yk7PJnryiJ8mAPqsrujzf?si=c2e7c1714e0048be",
                 "https://open.spotify.com/track/1eUGmzzvahJjOSWgDHuRlv?si=6acaaa380516451f",
                 "https://open.spotify.com/track/5fEB6ZmVkg63GZg9qO86jh?si=d702f883e6a74661",
                 "https://open.spotify.com/track/4cEqoGTqPRZy76Yl3ymj3V?si=607505ce0fd74ab9",
                 "https://open.spotify.com/track/5m0yZ33oOy0yYBtdTXuxQe?si=8b052f9f38154c31",
                 "https://open.spotify.com/track/1Vp4St7JcXaUoJcIahtf3L?si=d98cb80a21c5487e",
                 "https://open.spotify.com/track/2wAJTrFhCnQyNSD3oUgTZO?si=a3ab2a4160324e16",
                 "https://open.spotify.com/track/3eekarcy7kvN4yt5ZFzltW?si=54306b352bad438c",
                 "https://open.spotify.com/track/7nc7mlSdWYeFom84zZ8Wr8?si=ec12e794185a4c20",
                 "https://open.spotify.com/track/5TxRUOsGeWeRl3xOML59Ai?si=bddb01a23e8d46aa",
                 "https://open.spotify.com/track/44I7sqKYCAa7bQdVywkShO?si=24e88ed468cd44e1",
                 "https://open.spotify.com/track/1nXZnTALNXiPlvXotqHm66?si=2a4f84e61d0c4b1b",
                 "https://open.spotify.com/track/00imgaPlYRrMGn9o83hfmk?si=620b5c7eb048468d",
                 "https://open.spotify.com/track/432hUIl3ISDeytYW5XBQ5h?si=eb0bcaba494a4a13",
                 "https://open.spotify.com/track/7AzlLxHn24DxjgQX73F9fU?si=861375864d784288",
                 "https://open.spotify.com/track/37Nqx7iavZpotJSDXZWbJ3?si=8ad80be899b34af0",
                 "https://open.spotify.com/track/5yY9lUy8nbvjM1Uyo1Uqoc?si=728160360270434a",
                 "https://open.spotify.com/track/5d8yMIlqJH78lwOUP7T3oF?si=8fedd201833b46d5",
                 "https://open.spotify.com/track/05grSYrVwYw58YMOdJceyz?si=c715abc806c84c5b"
                 ]
names = [
    "A1_Silk_Sonic_Intro_Analysis",
    "A2_After_The_Storm_Analysis",
    "B1_Woods_Mac_Miller_Analysis",
    "B2_Alotta_Cake_Gunna",
    "C1_November_Tyler_The_Creator",
    "C2_03_Sainte",
    "D1_Thru_My_Hair_Teo",
    "D2_Screwed_Up_Teeze",
    "E1_1997_Brock_Hampton",
    "E2_Bean_Kobe_Uzi",
    "F1_Grace_Lil_Baby",
    "F2_Location_Playboi_Carti",
    "G1_Girl_With_Tattoo_Miguel",
    "G2_Break_From_Toronot_PartyNextDoor",
    "H1_Only_One_Travis_Scott",
    "H2_Low_Down_Lil_Baby",
    "I1_The_New_Workout_Plan_Kanye",
    "I2_Work_Out_Jcole",
    "J1_Highest_In_The_Room_Travis_Scot", 
    "J2_Tell_Em_Cochise",
    "K1_Are_We_Still_Friends_Tyler_The_Creator",
    "K2_Hurricane_Kanye",
    "L1_A_Boy_Is_A_Gun_Tyler_The_Creator",
    "L2_Loose_Change_Brent_Faiyaz",
    "M1_Wolvez_Kanye",
    "M2_No_Idea_Don_Toliver",
    "N1_Girls_Want_Girls_Drake",
    "N2_Life_Is_Good_Drake",
    "O1_Real_Kendrick",
    "O2_Patience_Lil_Uzi_Vert"]
Dict_Names = {
"A1_Silk_Sonic_Intro_Analysis": None,
"A2_After_The_Storm_Analysis": None,
"B1_Woods_Mac_Miller_Analysis": None,
"B2_Alotta_Cake_Gunna": None,
"C1_November_Tyler_The_Creator": None,
"C2_03_Sainte": None,
"D1_Thru_My_Hair_Teo": None,
"D2_Screwed_Up_Teeze": None,
"E1_1997_Brock_Hampton": None,
"E2_Bean_Kobe_Uzi": None,
"F1_Grace_Lil_Baby": None,
"F2_Location_Playboi_Carti": None,
"G1_Girl_With_Tattoo_Miguel": None,
"G2_Break_From_Toronot_PartyNextDoor": None,
"H1_Only_One_Travis_Scott": None,
"H2_Low_Down_Lil_Baby": None,
"I1_The_New_Workout_Plan_Kanye": None,
"I2_Work_Out_Jcole": None,
"J1_Highest_In_The_Room_Travis_Scot": None, 
"J2_Tell_Em_Cochise": None,
"K1_Are_We_Still_Friends_Tyler_The_Creator": None,
"K2_Hurricane_Kanye": None,
"L1_A_Boy_Is_A_Gun_Tyler_The_Creator": None,
"L2_Loose_Change_Brent_Faiyaz": None,
"M1_Wolvez_Kanye": None,
"M2_No_Idea_Don_Toliver": None,
"N1_Girls_Want_Girls_Drake": None,
"N2_Life_Is_Good_Drake": None,
"O1_Real_Kendrick": None,
"O2_Patience_Lil_Uzi_Vert": None}
store_open = []
for h in range(29):
    if h % 2 == 0:
        store_new = tensor_creator(good_transitions[h], good_transitions[h+1])
        store_new.make_tensor()
        store_check = store_new.segment_analyzer()
        Dict_Names[names[h]] = store_check
        if h == 28:
            Dict_Names[names[h+1]] = Dict_Names[names[h]]
    else:
        Dict_Names[names[h]] = Dict_Names[names[h-1]]
running_db = pd.DataFrame(Dict_Names)

        
    
    
    
    
    
