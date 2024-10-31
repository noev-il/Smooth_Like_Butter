import numpy as np
from sklearn.ensemble import RandomForestClassifier
import base64
import requests
import pandas as pd
import torch
import os

class tensor_creator:
    def __init__(self, url_1, url_2):
        self.client_id = os.environ.get('CLIENT_ID')
        self.client_secret = os.environ.get('CLIENT_SECRET')
        self.access_token = None

        if not self.client_id or not self.client_secret:
            raise ValueError("CLIENT_ID and CLIENT_SECRET must be set in environment variables.")
        
        
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
        self.create_and_analyze()

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
                return {"Request failed": e, "Response content": response.content}  # Print response content for debugging
        else:
            pass

    def _make_tensor(self):
        self.analysis_1 = self.get_track_analysis(self.url_1)
        self.analysis_2 = self.get_track_analysis(self.url_2)
        track_1 = self.analysis_1
        track_2 = self.analysis_2
        DATA_FRAME = []
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

    def _segment_analyzer(self):
        segment_1 = self.analysis_1['segments']
        segment_2 = self.analysis_2['segments']

        DATA_FRAME_1 = [pd.DataFrame(segment_1[:15])]
        pitches_1 = []
        timbre_1 = []
        pitches_2 = []
        timbre_2 = []
        DATA_FRAME_2 = [pd.DataFrame(segment_2[-15:])]
        DATA_FRAME_1[0].columns = ['start', 'duration', 'confidence',
                                   "loudness_start", "loudness_max_time",
                                   "loudness_max", "loudness_end", "pitches",
                                   "timbre"]
        DATA_FRAME_2[0].columns = ['start', 'duration', 'confidence',
                                   "loudness_start", "loudness_max_time",
                                   "loudness_max", "loudness_end",
                                   "pitches", "timbre"]
        # return DATA_FRAME_1[0]['pitches'][-1]
        for p in range(12):
            pitches_1 += [torch.tensor(DATA_FRAME_1[0]['pitches'][p])] 
            pitches_2 += [torch.tensor(DATA_FRAME_2[0]['pitches'][p])] 
        for t in range(12):
            timbre_1 += [torch.tensor(DATA_FRAME_1[0]['timbre'][t])] 
            timbre_2 += [torch.tensor(DATA_FRAME_2[0]['timbre'][t])] 
        pitches_dist = []
        timbres_dist = []
        for g in range(12):
            pitches_dist += [torch.sqrt(torch.sum(torch.pow(torch.subtract(
                pitches_1[g], pitches_2[g]), 2), dim=0))]
            timbres_dist += [torch.sqrt(torch.sum(torch.pow(torch.subtract(
                timbre_1[g], timbre_2[g]), 2), dim=0))]
        tensor_values_p = [tensor.item() for tensor in pitches_dist]
        tensor_values_t = [tensor.item() for tensor in timbres_dist]
        average_p = torch.tensor(tensor_values_p).mean()
        average_t = torch.tensor(tensor_values_t).mean()
        return np.array([self.distanceA1_A2.item(), average_p.item(), average_t.item()])

    def create_and_analyze(self):
        self._make_tensor()
        distance_features = self._segment_analyzer()
        # weight = sum(distance_features) / len(distance_features)  # Simple average for weight
        # return weight
        return distance_features

good_transitions = [
    "https://open.spotify.com/track/"
    "4K09vJ27xCOreumtSuU6Ao?si=7bf160671b854579",
    "https://open.spotify.com/track/"
    "1otG6j1WHNvl9WgXLWkHTo?si=a3575a900f3a4c6d",
    "https://open.spotify.com/track/"
    "3Qa944OTMZkg8DHjET8JQv?si=696ddb4a9daf4a67",
    "https://open.spotify.com/track/"
    "19QKaApDINxlRSKX3w1xSB?si=c9229546c6334acb",
    "https://open.spotify.com/track/"
    "4XDpeWqPADoWRKcUY3dC84?si=04e5e2519d52466c",
    "https://open.spotify.com/track/"
    "47gzGfR4JfKC6aT5lM0wpn?si=0e1b348a726140d4",
    "https://open.spotify.com/track/"
    "6BbAFjOCHA1AknMtIu3VjZ?si=8aea2fbe31c84d5e",
    "https://open.spotify.com/track/"
    "3d65swPOxko76ZQL5WEQfH?si=37614ad6886a4e54",
    "https://open.spotify.com/track/"
    "55jQMevNp7aWtiW5LPlPoa?si=1f0e24819e5e48b9",
    "https://open.spotify.com/track/"
    "0IpnZchq8ek2A6pGEP2Qb1?si=678492cafa964419",
    "https://open.spotify.com/track/"
    "01JMnRUs2YOK6DDpdQASGY?si=fb31acb7005141ec",
    "https://open.spotify.com/track/"
    "3yk7PJnryiJ8mAPqsrujzf?si=c2e7c1714e0048be",
    "https://open.spotify.com/track/"
    "1eUGmzzvahJjOSWgDHuRlv?si=6acaaa380516451f",
    "https://open.spotify.com/track/"
    "5fEB6ZmVkg63GZg9qO86jh?si=d702f883e6a74661",
    "https://open.spotify.com/track/"
    "4cEqoGTqPRZy76Yl3ymj3V?si=607505ce0fd74ab9",
    "https://open.spotify.com/track/"
    "5m0yZ33oOy0yYBtdTXuxQe?si=8b052f9f38154c31",
    "https://open.spotify.com/track/"
    "1Vp4St7JcXaUoJcIahtf3L?si=d98cb80a21c5487e",
    "https://open.spotify.com/track/"
    "2wAJTrFhCnQyNSD3oUgTZO?si=a3ab2a4160324e16",
    "https://open.spotify.com/track/"
    "3eekarcy7kvN4yt5ZFzltW?si=54306b352bad438c",
    "https://open.spotify.com/track/"
    "7nc7mlSdWYeFom84zZ8Wr8?si=ec12e794185a4c20",
    "https://open.spotify.com/track/"
    "5TxRUOsGeWeRl3xOML59Ai?si=bddb01a23e8d46aa",
    "https://open.spotify.com/track/"
    "44I7sqKYCAa7bQdVywkShO?si=24e88ed468cd44e1",
    "https://open.spotify.com/track/"
    "1nXZnTALNXiPlvXotqHm66?si=2a4f84e61d0c4b1b",
    "https://open.spotify.com/track/"
    "00imgaPlYRrMGn9o83hfmk?si=620b5c7eb048468d",
    "https://open.spotify.com/track/"
    "432hUIl3ISDeytYW5XBQ5h?si=eb0bcaba494a4a13",
    "https://open.spotify.com/track/"
    "7AzlLxHn24DxjgQX73F9fU?si=861375864d784288",
    "https://open.spotify.com/track/"
    "37Nqx7iavZpotJSDXZWbJ3?si=8ad80be899b34af0",
    "https://open.spotify.com/track/"
    "5yY9lUy8nbvjM1Uyo1Uqoc?si=728160360270434a",
    "https://open.spotify.com/track/"
    "5d8yMIlqJH78lwOUP7T3oF?si=8fedd201833b46d5",
    "https://open.spotify.com/track/"
    "05grSYrVwYw58YMOdJceyz?si=c715abc806c84c5b"
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
    "O2_Patience_Lil_Uzi_Vert": None
}
store_open = []
for h in range(29):
    if h % 2 == 0:
        store_new = tensor_creator(good_transitions[h], good_transitions[h+1])
        store_check = store_new.create_and_analyze()
        Dict_Names[names[h]] = store_check
        if h == 28:
            Dict_Names[names[h+1]] = Dict_Names[names[h]]
    else:
        Dict_Names[names[h]] = Dict_Names[names[h-1]]
running_db = pd.DataFrame(Dict_Names)

numpy_arrays = []

for k in range(len(names)):
    if k % 2 == 0:
        numpy_arrays += [np.array(Dict_Names[f'{names[k]}'])]
    else:
        None

bad_1 = tensor_creator('https://open.spotify.com/track/2FDTHlrBguDzQkp7PVj16Q?si=3ad17825b4774244', 'https://open.spotify.com/track/09FcXaLu1BdrRNgxyBi6p5?si=4b41dfb40c174e38')
numpy_arrays.append(bad_1.create_and_analyze())
bad_2 = tensor_creator('https://open.spotify.com/track/0XqCWpRB3DLSy0l9bFQ15A?si=4ebc0d0bb4b3446c', 'https://open.spotify.com/track/0WCbhE2evMrIwRM0DlMy9k?si=389f63f89d194583')
numpy_arrays.append(bad_2.create_and_analyze())
bad_3 = tensor_creator('https://open.spotify.com/track/71SbmXsy5H0bqxJAVBcfsG?si=7692e0d63eea456e', 'https://open.spotify.com/track/7uHF03xE84sQ5PicRNH3yu?si=d8903751e6be4e22')
numpy_arrays.append(bad_3.create_and_analyze())
bad_4 = tensor_creator('https://open.spotify.com/track/7KVPsVMOK3NL7subwJ0dZj?si=46893e8879074917', 'https://open.spotify.com/track/1chxfk33LoVOznJiJ0WWPD?si=da93686c079e4d0c')
numpy_arrays.append(bad_4.create_and_analyze())
bad_5 = tensor_creator('https://open.spotify.com/track/421r1p6Uzy72gSOyWHpmdA?si=af4d8c4f6077495e', 'https://open.spotify.com/track/2OaKHGvIxoOzIYjyMsxcT8?si=1b22b0e0a99a4f14')
numpy_arrays.append(bad_5.create_and_analyze())
y_train = ['smooth'] * 15
y_train += ['bad'] * 5

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(numpy_arrays, y_train)

test = tensor_creator('https://open.spotify.com/track/6BquHLBPxtDrkqfP3GhLT5?si=f2c295a246d149ac', 'https://open.spotify.com/track/5RlDYfphEpQaft5JDQfQko?si=679e133939cf422b')
test = [test.create_and_analyze()]
print(clf.predict(test))

