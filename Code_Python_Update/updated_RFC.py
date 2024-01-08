import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


import base64
import requests

import json

from sklearn.datasets import load_iris
import pandas as pd



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
        for y in range(len(DATA_FRAME[0])):
            crucial_metadata = (DATA_FRAME[0]['loudness'][y], DATA_FRAME[0]['tempo'][y], DATA_FRAME[0]['key'][y], DATA_FRAME[0]['mode'][y])
            new_tensor = np.array(crucial_metadata)
            Song_Tensor += [new_tensor]
        DATA_FRAME[1].columns = ['start', 'duration', 'confidence', 'loudness', 'tempo', 'tempo_confidence', 'key', 'key_confidence', 'mode', 'mode_confidence', 'time_signature', 'time_signature_confidence']
        crucial_metadata_1 = (DATA_FRAME[1]['loudness'][0], DATA_FRAME[1]['tempo'][0], DATA_FRAME[1]['key'][0], DATA_FRAME[1]['mode'][0])
        new_tensor = np.array(crucial_metadata_1)
        Song_Tensor_1 += [new_tensor]
        initial_subtraction = Song_Tensor[-1] - Song_Tensor_1
        distanceA1_A2 = np.linalg.norm(initial_subtraction)
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
            pitches_1 += [np.array(DATA_FRAME_1[0]['pitches'][p])] 
            pitches_2 += [np.array(DATA_FRAME_2[0]['pitches'][p])] 
        for t in range(12):
            timbre_1 += [np.array(DATA_FRAME_1[0]['timbre'][t])] 
            timbre_2 += [np.array(DATA_FRAME_2[0]['timbre'][t])] 
        pitches_dist = []
        timbres_dist =  []
        for g in range(12):
            pitches_subtraction = pitches_1[g] - pitches_2[g]
            pitches_dist += [np.linalg.norm(pitches_subtraction)]
            timbre_subtraction = timbre_1[g] - timbre_2[g]
            timbres_dist += [np.linalg.norm(timbre_subtraction)]
        array_pitches = np.array(pitches_dist)
        pitches_mean = np.mean(array_pitches)
        array_timbre = np.array(timbres_dist)
        timbres_mean = np.mean(array_timbre)
        return self.distanceA1_A2.item(), pitches_mean, timbres_mean

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
    
    

numpy_arrays = []

##print(len(names))
for k in range(len(names)):
    if k % 2 == 0:
        numpy_arrays += [np.array(Dict_Names[f'{names[k]}'])]
    else:
        None
##print(len(numpy_arrays))    
bad_1 = tensor_creator('https://open.spotify.com/track/2FDTHlrBguDzQkp7PVj16Q?si=3ad17825b4774244', 'https://open.spotify.com/track/09FcXaLu1BdrRNgxyBi6p5?si=4b41dfb40c174e38')
bad_1.make_tensor()
numpy_arrays += [np.array(bad_1.segment_analyzer())]
bad_2 = tensor_creator('https://open.spotify.com/track/0XqCWpRB3DLSy0l9bFQ15A?si=4ebc0d0bb4b3446c', 'https://open.spotify.com/track/0WCbhE2evMrIwRM0DlMy9k?si=389f63f89d194583')
bad_2.make_tensor()
numpy_arrays += [np.array(bad_2.segment_analyzer())]
bad_3 = tensor_creator('https://open.spotify.com/track/71SbmXsy5H0bqxJAVBcfsG?si=7692e0d63eea456e', 'https://open.spotify.com/track/7uHF03xE84sQ5PicRNH3yu?si=d8903751e6be4e22')
bad_3.make_tensor()
numpy_arrays += [np.array(bad_3.segment_analyzer())]
bad_4 = tensor_creator('https://open.spotify.com/track/7KVPsVMOK3NL7subwJ0dZj?si=46893e8879074917', 'https://open.spotify.com/track/1chxfk33LoVOznJiJ0WWPD?si=da93686c079e4d0c')
bad_4.make_tensor()
numpy_arrays += [np.array(bad_4.segment_analyzer())]
bad_5 = tensor_creator('https://open.spotify.com/track/421r1p6Uzy72gSOyWHpmdA?si=af4d8c4f6077495e', 'https://open.spotify.com/track/2OaKHGvIxoOzIYjyMsxcT8?si=1b22b0e0a99a4f14')
bad_5.make_tensor()
numpy_arrays += [np.array(bad_5.segment_analyzer())]
y_train = ['smooth'] * 15
y_train += ['bad'] * 5

#{#
X_test = []
test_1 = tensor_creator('https://open.spotify.com/track/26hOm7dTtBi0TdpDGl141t?si=c16c108e8410477c', 'https://open.spotify.com/track/46NzAxDzsE443IsyZndZfP?si=acaed19953e04c25')
test_1.make_tensor()
X_test += [np.array(test_1.segment_analyzer())]
test_2 = tensor_creator('https://open.spotify.com/track/7viEq8U0GgZf3v5m4BON3c?si=5c2e592fb9b142f4', 'https://open.spotify.com/track/7EcE5yCPVZaZut1JqowbcI?si=cd49328b067748ab')
test_2.make_tensor()
X_test += [np.array(test_2.segment_analyzer())]
test_3 = tensor_creator('https://open.spotify.com/track/7fEoXCZTZFosUFvFQg1BmW?si=9fb6dbbdb06e48f5', 'https://open.spotify.com/track/3Iy4j2lCqW8BXGkFk21U6F?si=20634ba0b4cf4479')
test_3.make_tensor()
X_test += [np.array(test_3.segment_analyzer())]
#}#
y_test = ['smooth', 'bad', 'bad']
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(numpy_arrays, y_train)
predictions = clf.predict(X_test) 
print(predictions)
##URLS for Test
URL_TEST = ["https://open.spotify.com/track/0M9ydKzuF3oZTfYYPfaGX1?si=d6ea82e91efc44d1",
"https://open.spotify.com/track/4UKCKdYiLN6IMA5ZESUTL7?si=bbcf8b1225b947c7",
"https://open.spotify.com/track/4UKCKdYiLN6IMA5ZESUTL7?si=c78bf98910c74897",
"https://open.spotify.com/track/6fujklziTHa8uoM5OQSfIo?si=6a19c019bbe64e28",
"https://open.spotify.com/track/6fujklziTHa8uoM5OQSfIo?si=6a19c019bbe64e28",
"https://open.spotify.com/track/2EEeOnHehOozLq4aS0n6SL?si=4e26105c4258431c",
"https://open.spotify.com/track/2EEeOnHehOozLq4aS0n6SL?si=4e26105c4258431c",
"https://open.spotify.com/track/79s5XnCN4TJKTVMSmOx8Ep?si=7c53078fd3814d1b",
"https://open.spotify.com/track/79s5XnCN4TJKTVMSmOx8Ep?si=7c53078fd3814d1b",
"https://open.spotify.com/track/343YBumqHu19cGoGARUTsd?si=a65a91c1c5114b7d",
"https://open.spotify.com/track/343YBumqHu19cGoGARUTsd?si=a65a91c1c5114b7d",
"https://open.spotify.com/track/51Fjme0JiitpyXKuyQiCDo?si=4a593871b1ac4888",
"https://open.spotify.com/track/51Fjme0JiitpyXKuyQiCDo?si=4a593871b1ac4888",
"https://open.spotify.com/track/5yuShbu70mtHXY0yLzCQLQ?si=5add196616c245b0",
"https://open.spotify.com/track/5yuShbu70mtHXY0yLzCQLQ?si=5add196616c245b0",
"https://open.spotify.com/track/7eBqSVxrzQZtK2mmgRG6lC?si=892c4005e3f9440c",
"https://open.spotify.com/track/7eBqSVxrzQZtK2mmgRG6lC?si=892c4005e3f9440c",
"https://open.spotify.com/track/1PS1QMdUqOal0ai3Gt7sDQ?si=88451415c60048d6",
"https://open.spotify.com/track/1PS1QMdUqOal0ai3Gt7sDQ?si=88451415c60048d6",
"https://open.spotify.com/track/7165YDcOpr9yEypbdpU6fa?si=6778a325070e4719",
"https://open.spotify.com/track/7165YDcOpr9yEypbdpU6fa?si=6778a325070e4719",
"https://open.spotify.com/track/0F7FA14euOIX8KcbEturGH?si=5b67a56ea9de4043",
"https://open.spotify.com/track/0F7FA14euOIX8KcbEturGH?si=5b67a56ea9de4043",
"https://open.spotify.com/track/6vN77lE9LK6HP2DewaN6HZ?si=7adc4e50366243b2",
"https://open.spotify.com/track/6vN77lE9LK6HP2DewaN6HZ?si=7adc4e50366243b2",
"https://open.spotify.com/track/1xzBco0xcoJEDXktl7Jxrr?si=92a25887fff44b94",
"https://open.spotify.com/track/1xzBco0xcoJEDXktl7Jxrr?si=92a25887fff44b94",
"https://open.spotify.com/track/32lItqlMi4LBhb4k0BaSaC?si=a5c45dcbf2e8462c",
"https://open.spotify.com/track/32lItqlMi4LBhb4k0BaSaC?si=a5c45dcbf2e8462c",
"https://open.spotify.com/track/0FZ4Dmg8jJJAPJnvBIzD9z?si=1a0ff0c9e5d34d41",
"https://open.spotify.com/track/0FZ4Dmg8jJJAPJnvBIzD9z?si=1a0ff0c9e5d34d41",
"https://open.spotify.com/track/6or1bKJiZ06IlK0vFvY75k?si=8f57e65e46e44da4",
"https://open.spotify.com/track/6or1bKJiZ06IlK0vFvY75k?si=8f57e65e46e44da4",
"https://open.spotify.com/track/1tkg4EHVoqnhR6iFEXb60y?si=f5b4ae8deb0347c6",
"https://open.spotify.com/track/1tkg4EHVoqnhR6iFEXb60y?si=f5b4ae8deb0347c6",
"https://open.spotify.com/track/127QTOFJsJQp5LbJbu3A1y?si=4135faf5328d457a",
"https://open.spotify.com/track/127QTOFJsJQp5LbJbu3A1y?si=4135faf5328d457a",
"https://open.spotify.com/track/0TlLq3lA83rQOYtrqBqSct?si=0c6d419196494e9c",
"https://open.spotify.com/track/0TlLq3lA83rQOYtrqBqSct?si=0c6d419196494e9c",
"https://open.spotify.com/track/4jvjzW7Hm0yK4LvvE0Paz9?si=9473fa1aef184376",
"https://open.spotify.com/track/4jvjzW7Hm0yK4LvvE0Paz9?si=9473fa1aef184376",
"https://open.spotify.com/track/6HZILIRieu8S0iqY8kIKhj?si=58115a08418f4b78",
"https://open.spotify.com/track/6HZILIRieu8S0iqY8kIKhj?si=58115a08418f4b78",
"https://open.spotify.com/track/0PvFJmanyNQMseIFrU708S?si=7fd89364e6a343b4",
"https://open.spotify.com/track/0PvFJmanyNQMseIFrU708S?si=7fd89364e6a343b4",
"https://open.spotify.com/track/7ycWLEP1GsNjVvcjawXz3z?si=83db96551e644c83",
"https://open.spotify.com/track/7ycWLEP1GsNjVvcjawXz3z?si=83db96551e644c83",
"https://open.spotify.com/track/7bYZBVrnRfqeaPbhRyEvK3?si=5e4878c4d33846ca",
"https://open.spotify.com/track/7bYZBVrnRfqeaPbhRyEvK3?si=5e4878c4d33846ca",
"https://open.spotify.com/track/6eT7xZZlB2mwyzJ2sUKG6w?si=b83d5eec8e1a4d3f",
"https://open.spotify.com/track/6eT7xZZlB2mwyzJ2sUKG6w?si=b83d5eec8e1a4d3f",
"https://open.spotify.com/track/7KA4W4McWYRpgf0fWsJZWB?si=503139682f6a4cde",
"https://open.spotify.com/track/7KA4W4McWYRpgf0fWsJZWB?si=503139682f6a4cde",
"https://open.spotify.com/track/4dASQiO1Eoo3RJvt74FtXB?si=c49426f549e84c3b",
"https://open.spotify.com/track/4dASQiO1Eoo3RJvt74FtXB?si=c49426f549e84c3b",
"https://open.spotify.com/track/5eqK0tbzUPo2SoeZsov04s?si=36eb74a2abe24593",
"https://open.spotify.com/track/5eqK0tbzUPo2SoeZsov04s?si=36eb74a2abe24593",
"https://open.spotify.com/track/2d8JP84HNLKhmd6IYOoupQ?si=a0c14d4ca6904a39",
"https://open.spotify.com/track/2d8JP84HNLKhmd6IYOoupQ?si=a0c14d4ca6904a39",
"https://open.spotify.com/track/2IRZnDFmlqMuOrYOLnZZyc?si=db72ad8a6d884449",
"https://open.spotify.com/track/2IRZnDFmlqMuOrYOLnZZyc?si=db72ad8a6d884449",
"https://open.spotify.com/track/6PGoSes0D9eUDeeAafB2As?si=52473437ed1d416a",
"https://open.spotify.com/track/6PGoSes0D9eUDeeAafB2As?si=52473437ed1d416a",
"https://open.spotify.com/track/74tLlkN3rgVzRqQJgPfink?si=97f11bab43094893",
"https://open.spotify.com/track/74tLlkN3rgVzRqQJgPfink?si=97f11bab43094893",
"https://open.spotify.com/track/2JvzF1RMd7lE3KmFlsyZD8?si=dd1d2b269c8d4e21",
"https://open.spotify.com/track/2JvzF1RMd7lE3KmFlsyZD8?si=dd1d2b269c8d4e21",
"https://open.spotify.com/track/2PpruBYCo4H7WOBJ7Q2EwM?si=8fd32a25d946454d",
"https://open.spotify.com/track/2PpruBYCo4H7WOBJ7Q2EwM?si=8fd32a25d946454d",
"https://open.spotify.com/track/5yY9lUy8nbvjM1Uyo1Uqoc?si=9247d27c31c64971",
"https://open.spotify.com/track/5yY9lUy8nbvjM1Uyo1Uqoc?si=9247d27c31c64971",
"https://open.spotify.com/track/275a9yzwGB6ncAW4SxY7q3?si=46f59f4757e34276",
"https://open.spotify.com/track/275a9yzwGB6ncAW4SxY7q3?si=46f59f4757e34276",
"https://open.spotify.com/track/0t3ZvGKlmYmVsDzBJAXK8C?si=b7b49a74c3b548a3",
"https://open.spotify.com/track/0t3ZvGKlmYmVsDzBJAXK8C?si=b7b49a74c3b548a3",
"https://open.spotify.com/track/2cYqizR4lgvp4Qu6IQ3qGN?si=dd30326f0a9e426a",
"https://open.spotify.com/track/2cYqizR4lgvp4Qu6IQ3qGN?si=dd30326f0a9e426a",
"https://open.spotify.com/track/3CA9pLiwRIGtUBiMjbZmRw?si=6db20eaa34ed4b4e",
"https://open.spotify.com/track/3CA9pLiwRIGtUBiMjbZmRw?si=6db20eaa34ed4b4e",
"https://open.spotify.com/track/1lOe9qE0vR9zwWQAOk6CoO?si=64113fea95644f6c",
"https://open.spotify.com/track/1lOe9qE0vR9zwWQAOk6CoO?si=64113fea95644f6c",
"https://open.spotify.com/track/6iaSML1PIYq936g62BDtBq?si=341cdd5d621b4492",
"https://open.spotify.com/track/6iaSML1PIYq936g62BDtBq?si=341cdd5d621b4492",
"https://open.spotify.com/track/7FIWs0pqAYbP91WWM0vlTQ?si=8bccb1c936d54722",
"https://open.spotify.com/track/7FIWs0pqAYbP91WWM0vlTQ?si=8bccb1c936d54722",
"https://open.spotify.com/track/6wJYhPfqk3KGhHRG76WzOh?si=9a9636f83bd64501",
"https://open.spotify.com/track/6wJYhPfqk3KGhHRG76WzOh?si=9a9636f83bd64501",
"https://open.spotify.com/track/3DXncPQOG4VBw3QHh3S817?si=c8e46d945c1e42cb",
"https://open.spotify.com/track/3DXncPQOG4VBw3QHh3S817?si=c8e46d945c1e42cb",
"https://open.spotify.com/track/4Li2WHPkuyCdtmokzW2007?si=e2fde7728449440d",
"https://open.spotify.com/track/4Li2WHPkuyCdtmokzW2007?si=e2fde7728449440d",
"https://open.spotify.com/track/7iL6o9tox1zgHpKUfh9vuC?si=f087edde1eed46ef",
"https://open.spotify.com/track/7iL6o9tox1zgHpKUfh9vuC?si=f087edde1eed46ef",
"https://open.spotify.com/track/4Oun2ylbjFKMPTiaSbbCih?si=2fbea8c1ce8740aa",
"https://open.spotify.com/track/4Oun2ylbjFKMPTiaSbbCih?si=2fbea8c1ce8740aa",
"https://open.spotify.com/track/02kDW379Yfd5PzW5A6vuGt?si=f9947e0e92e548da",
"https://open.spotify.com/track/02kDW379Yfd5PzW5A6vuGt?si=f9947e0e92e548da",
"https://open.spotify.com/track/503OTo2dSqe7qk76rgsbep?si=3d88f4e284664bd3",
"https://open.spotify.com/track/503OTo2dSqe7qk76rgsbep?si=3d88f4e284664bd3",
"https://open.spotify.com/track/503OTo2dSqe7qk76rgsbep?si=84df979e73674b35",
"https://open.spotify.com/track/503OTo2dSqe7qk76rgsbep?si=84df979e73674b35",
"https://open.spotify.com/track/0j2T0R9dR9qdJYsB7ciXhf?si=8048df73ded54a33",
"https://open.spotify.com/track/0j2T0R9dR9qdJYsB7ciXhf?si=8048df73ded54a33",
"https://open.spotify.com/track/0wwPcA6wtMf6HUMpIRdeP7?si=6b4bf4d902e7426f",
"https://open.spotify.com/track/0wwPcA6wtMf6HUMpIRdeP7?si=6b4bf4d902e7426f",
"https://open.spotify.com/track/15JINEqzVMv3SvJTAXAKED?si=1fc78675381d4f92",
"https://open.spotify.com/track/15JINEqzVMv3SvJTAXAKED?si=1fc78675381d4f92",
"https://open.spotify.com/track/5HQVUIKwCEXpe7JIHyY734?si=10660412635d4a8e",
"https://open.spotify.com/track/5HQVUIKwCEXpe7JIHyY734?si=10660412635d4a8e",
"https://open.spotify.com/track/2toVe5hfuIi97ytDPDbQFt?si=19a6fe44378d4b49",
"https://open.spotify.com/track/2toVe5hfuIi97ytDPDbQFt?si=19a6fe44378d4b49",
"https://open.spotify.com/track/561jH07mF1jHuk7KlaeF0s?si=a7efaab3a2d241c3",
"https://open.spotify.com/track/561jH07mF1jHuk7KlaeF0s?si=a7efaab3a2d241c3",
"https://open.spotify.com/track/7sO5G9EABYOXQKNPNiE9NR?si=5e9f503888a244a1",
"https://open.spotify.com/track/7sO5G9EABYOXQKNPNiE9NR?si=5e9f503888a244a1",
"https://open.spotify.com/track/78QR3Wp35dqAhFEc2qAGjE?si=4b766ac9788a4163",
"https://open.spotify.com/track/78QR3Wp35dqAhFEc2qAGjE?si=4b766ac9788a4163",
"https://open.spotify.com/track/7floNISpH8VF4z4459Qo18?si=a4e8bfdaaebb4383",
"https://open.spotify.com/track/7floNISpH8VF4z4459Qo18?si=a4e8bfdaaebb4383",
"https://open.spotify.com/track/7AFASza1mXqntmGtbxXprO?si=d467a8917402480f",
"https://open.spotify.com/track/7AFASza1mXqntmGtbxXprO?si=d467a8917402480f",
"https://open.spotify.com/track/3GCdLUSnKSMJhs4Tj6CV3s?si=b8b10cfe1ffb45ee",
"https://open.spotify.com/track/3GCdLUSnKSMJhs4Tj6CV3s?si=b8b10cfe1ffb45ee",
"https://open.spotify.com/track/2G7V7zsVDxg1yRsu7Ew9RJ?si=fcd754fa99174bfd",
"https://open.spotify.com/track/2G7V7zsVDxg1yRsu7Ew9RJ?si=fcd754fa99174bfd",
"https://open.spotify.com/track/52okn5MNA47tk87PeZJLEL?si=24d9abae83204b9a",
"https://open.spotify.com/track/52okn5MNA47tk87PeZJLEL?si=24d9abae83204b9a",
"https://open.spotify.com/track/4VXIryQMWpIdGgYR4TrjT1?si=0a97fb7163a84010",
"https://open.spotify.com/track/4VXIryQMWpIdGgYR4TrjT1?si=0a97fb7163a84010",
"https://open.spotify.com/track/1DIXPcTDzTj8ZMHt3PDt8p?si=62362f24f37940db",
"https://open.spotify.com/track/1DIXPcTDzTj8ZMHt3PDt8p?si=62362f24f37940db",
"https://open.spotify.com/track/7AQim7LbvFVZJE3O8TYgf2?si=d2c6755fad524caf",
"https://open.spotify.com/track/7AQim7LbvFVZJE3O8TYgf2?si=d2c6755fad524caf",
"https://open.spotify.com/track/3GVkPk8mqxz0itaAriG1L7?si=e375414d98c44408",
"https://open.spotify.com/track/3GVkPk8mqxz0itaAriG1L7?si=e375414d98c44408",
"https://open.spotify.com/track/696DnlkuDOXcMAnKlTgXXK?si=056e1024895a4f9f",
"https://open.spotify.com/track/696DnlkuDOXcMAnKlTgXXK?si=056e1024895a4f9f",
"https://open.spotify.com/track/3eekarcy7kvN4yt5ZFzltW?si=08bad87dad8d4062",
"https://open.spotify.com/track/3eekarcy7kvN4yt5ZFzltW?si=08bad87dad8d4062",
"https://open.spotify.com/track/0VgkVdmE4gld66l8iyGjgx?si=5a8bd559ee1d4cee",
"https://open.spotify.com/track/0VgkVdmE4gld66l8iyGjgx?si=5a8bd559ee1d4cee",
"https://open.spotify.com/track/3swc6WTsr7rl9DqQKQA55C?si=74783d75fd4e4602",
"https://open.spotify.com/track/3swc6WTsr7rl9DqQKQA55C?si=6017448db45b423c",
"https://open.spotify.com/track/3yfqSUWxFvZELEM4PmlwIR?si=634337129606406e",
"https://open.spotify.com/track/3yfqSUWxFvZELEM4PmlwIR?si=634337129606406e",
"https://open.spotify.com/track/3yfqSUWxFvZELEM4PmlwIR?si=226d28fb8de54aeb",
"https://open.spotify.com/track/3yfqSUWxFvZELEM4PmlwIR?si=226d28fb8de54aeb",
"https://open.spotify.com/track/2YpeDb67231RjR0MgVLzsG?si=74e54648941c4d47",
"https://open.spotify.com/track/2YpeDb67231RjR0MgVLzsG?si=74e54648941c4d47",
"https://open.spotify.com/track/7xQAfvXzm3AkraOtGPWIZg?si=ef603ad741c34c9c",
"https://open.spotify.com/track/7xQAfvXzm3AkraOtGPWIZg?si=ef603ad741c34c9c",
"https://open.spotify.com/track/58q2HKrzhC3ozto2nDdN4z?si=e8e0d2a21ce24136",
"https://open.spotify.com/track/58q2HKrzhC3ozto2nDdN4z?si=e8e0d2a21ce24136",
"https://open.spotify.com/track/7ytR5pFWmSjzHJIeQkgog4?si=cec6858419884815",
"https://open.spotify.com/track/7ytR5pFWmSjzHJIeQkgog4?si=cec6858419884815",
"https://open.spotify.com/track/2JzZzZUQj3Qff7wapcbKjc?si=9740e758674e40d1",
"https://open.spotify.com/track/2JzZzZUQj3Qff7wapcbKjc?si=9740e758674e40d1",
"https://open.spotify.com/track/0nbXyq5TXYPCO7pr3N8S4I?si=57453ce8bdd342d7",
"https://open.spotify.com/track/0nbXyq5TXYPCO7pr3N8S4I?si=57453ce8bdd342d7",
"https://open.spotify.com/track/0JP9xo3adEtGSdUEISiszL?si=8c8cd8707cfd4a59",
"https://open.spotify.com/track/0JP9xo3adEtGSdUEISiszL?si=8c8cd8707cfd4a59",
"https://open.spotify.com/track/7lQ8MOhq6IN2w8EYcFNSUk?si=13cf0920abc342e9",
"https://open.spotify.com/track/7lQ8MOhq6IN2w8EYcFNSUk?si=13cf0920abc342e9",
"https://open.spotify.com/track/4xkOaSrkexMciUUogZKVTS?si=4788e6bcc88c4f42",
"https://open.spotify.com/track/4xkOaSrkexMciUUogZKVTS?si=4788e6bcc88c4f42",
"https://open.spotify.com/track/4jPy3l0RUwlUI9T5XHBW2m?si=b7f29aa57bb0465e",
"https://open.spotify.com/track/4jPy3l0RUwlUI9T5XHBW2m?si=b7f29aa57bb0465e",
"https://open.spotify.com/track/3B54sVLJ402zGa6Xm4YGNe?si=1857cbcac4114f63",
"https://open.spotify.com/track/3B54sVLJ402zGa6Xm4YGNe?si=1857cbcac4114f63",
"https://open.spotify.com/track/67BtfxlNbhBmCDR2L2l8qd?si=1c2d3709e2634134",
"https://open.spotify.com/track/67BtfxlNbhBmCDR2L2l8qd?si=1c2d3709e2634134",
"https://open.spotify.com/track/3a1lNhkSLSkpJE4MSHpDu9?si=a7afa8fd529046fc",
"https://open.spotify.com/track/3a1lNhkSLSkpJE4MSHpDu9?si=a7afa8fd529046fc",
"https://open.spotify.com/track/7dt6x5M1jzdTEt8oCbisTK?si=1814441549a14e9a",
"https://open.spotify.com/track/7dt6x5M1jzdTEt8oCbisTK?si=1814441549a14e9a",
"https://open.spotify.com/track/68Dni7IE4VyPkTOH9mRWHr?si=9899932927034a28",
"https://open.spotify.com/track/68Dni7IE4VyPkTOH9mRWHr?si=9899932927034a28",
"https://open.spotify.com/track/27NovPIUIRrOZoCHxABJwK?si=6a05441862524d21",
"https://open.spotify.com/track/27NovPIUIRrOZoCHxABJwK?si=6a05441862524d21",
"https://open.spotify.com/track/5Z01UMMf7V1o0MzF86s6WJ?si=2d3044fa76704f7c",
"https://open.spotify.com/track/5Z01UMMf7V1o0MzF86s6WJ?si=2d3044fa76704f7c",
"https://open.spotify.com/track/2xLMifQCjDGFmkHkpNLD9h?si=75478c9e18e74499",
"https://open.spotify.com/track/2xLMifQCjDGFmkHkpNLD9h?si=75478c9e18e74499",
"https://open.spotify.com/track/7GX5flRQZVHRAGd6B4TmDO?si=77e023f056fe4516",
"https://open.spotify.com/track/7GX5flRQZVHRAGd6B4TmDO?si=77e023f056fe4516",
"https://open.spotify.com/track/7KXjTSCq5nL1LoYtL7XAwS?si=e46862a5c5974ed8",
"https://open.spotify.com/track/7KXjTSCq5nL1LoYtL7XAwS?si=e46862a5c5974ed8",
"https://open.spotify.com/track/3bidbhpOYeV4knp8AIu8Xn?si=b64cf46f9d334e9f",
"https://open.spotify.com/track/3bidbhpOYeV4knp8AIu8Xn?si=b64cf46f9d334e9f",
"https://open.spotify.com/track/7m9OqQk4RVRkw9JJdeAw96?si=bfd6fe89bece4ab5",
"https://open.spotify.com/track/7m9OqQk4RVRkw9JJdeAw96?si=bfd6fe89bece4ab5",
"https://open.spotify.com/track/6gBFPUFcJLzWGx4lenP6h2?si=d4fb296513394195",
"https://open.spotify.com/track/6gBFPUFcJLzWGx4lenP6h2?si=d4fb296513394195",
"https://open.spotify.com/track/3ee8Jmje8o58CHK66QrVC2?si=ba0e9c33a34948a9",
"https://open.spotify.com/track/3ee8Jmje8o58CHK66QrVC2?si=ba0e9c33a34948a9",
"https://open.spotify.com/track/6DCZcSspjsKoFjzjrWoCdn?si=9b398c7cef764be9",
"https://open.spotify.com/track/6DCZcSspjsKoFjzjrWoCdn?si=9b398c7cef764be9",
"https://open.spotify.com/track/285pBltuF7vW8TeWk8hdRR?si=2a1da95f540b463f",
"https://open.spotify.com/track/285pBltuF7vW8TeWk8hdRR?si=2a1da95f540b463f",
"https://open.spotify.com/track/0e7ipj03S05BNilyu5bRzt?si=3cc3dfa2f17d4216",
"https://open.spotify.com/track/0e7ipj03S05BNilyu5bRzt?si=3cc3dfa2f17d4216",
"https://open.spotify.com/track/0RiRZpuVRbi7oqRdSMwhQY?si=cc539e000fb74b45",
    ]
Massive_Test = []
##for item in range(len(URL_TEST) - 1):
    ##fortune = tensor_creator(URL_TEST[item], URL_TEST[item + 1])
    ##fortune.make_tensor()
    ##Massive_Test += [np.array(fortune.segment_analyzer())]

##sci-kit learn machine learning Random Forest for Classification
##clf = RandomForestClassifier(n_estimators=100, random_state=42)
##clf.fit(numpy_arrays, y_train)
##ONLY DO THIS PART IF YOU HAVE THE TIME AND PC CAPABILITIES FOR THIS
##predictions = clf.predict(Massive_Test) 
##print(predictions) 


    
    
    
    
    


