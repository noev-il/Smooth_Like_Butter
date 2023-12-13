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
        distanceA1_A2 = []
        distanceA1_A2 += [torch.sqrt(torch.sum(torch.pow(torch.subtract(Song_Tensor[-1], Song_Tensor_1[0]), 2), dim=0))]
        return distanceA1_A2
tc_1 = tensor_creator('https://open.spotify.com/track/01JMnRUs2YOK6DDpdQASGY?si=3895052dec1244e8', 'https://open.spotify.com/track/01JMnRUs2YOK6DDpdQASGY?si=3895052dec1244e8')
tc_2 = tensor_creator('https://open.spotify.com/track/2U5cq89GCnsR1yixKkC8d5?si=2b0ec96cd8104c7d', 'https://open.spotify.com/track/7IpshETtrXEUGD4z2485R9?si=5de6f693b4824638')
tc_3 = tensor_creator('https://open.spotify.com/track/2HHLfzE7PkljuqyYU4vwmh?si=e5e1db82489b4b62', 'https://open.spotify.com/track/6gxKUmycQX7uyMwJcweFjp?si=b0838403b1844686')

##tc_4 = tensor_creator('', '') 
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
print(tc_1.make_tensor(), tc_2.make_tensor(), tc_3.make_tensor())
            
            
            
            