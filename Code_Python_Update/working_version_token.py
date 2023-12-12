import base64
import requests

import json
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
import torch

client_id = "76915a0c79a34e108e08ab4ca0e2605f"
client_secret = "f30900615bad41d0ae950d4e2ad72668"

def get_token():
    api_token = "https://accounts.spotify.com/api/token"
    auth_string = f'{client_id}:{client_secret}'
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
    return token_response_data.get('access_token')

def get_track_analysis(song_id):
    token_live = get_token()
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
get_song_id_1 = str(input("Enter Song URL #1: "))
get_song_id_2 = str(input("Enter Song URL #2: "))
analysis_1 = get_track_analysis(get_song_id_1[31:])
analysis_2 = get_track_analysis(get_song_id_2[31:])
####print(analysis_1['sections'])


##new_start##

DATA_FRAME = []
tensors = []
Dictionary = {'start': [], 'duration': [], 'confidence': [], 'loudness': [], 'tempo': [], 'tempo_confidence': [], 'key': [], 'key_confidence': [], 'mode': [], 'mode_confidence': [], 'time_signature': [], 'time_signature_confidence': []}
new_tensor = None
Song_Tensor = []
Song_Tensor_1 = []
sections_1 = analysis_1['sections']
sections_2 = analysis_2['sections']
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
print(distanceA1_A2)