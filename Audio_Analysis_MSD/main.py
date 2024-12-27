import h5py
import numpy as np
import pandas as pd
import os
import json

# Open the HDF5 file
h5_file = h5py.File('/Users/aux/Downloads/Mac_Storage/Smooth_Like_Butter/Audio_Analysis_MSD/MillionSongSubset/A/A/A/TRAAAAW128F429D538.h5', 'r')
with h5_file as f: #loudness, tempo, key, mode, pitch, timbre
    analysis = f['analysis']
    metadata = f['metadata']

    # Song-level details
    song_details = {
        "title": metadata['songs'][0][b'title'.decode('utf-8')].decode('utf-8'),
        "loudness": analysis['songs'][0][b'loudness'.decode('utf-8')][()].item(),
        "tempo": analysis['songs'][0][b'tempo'.decode('utf-8')][()].item(),
        "key": analysis['songs'][0][b'key'.decode('utf-8')][()].item(),
        "mode": analysis['songs'][0][b'mode'.decode('utf-8')][()].item(),
    }

    # Extract segment start times
    segment_starts = analysis['segments_start'][:]
    pitches = analysis['segments_pitches'][:]
    timbres = analysis['segments_timbre'][:]

    # Find segments in the last 4 seconds
    duration = analysis['songs'][0][b'duration'.decode('utf-8')].item()  # Total song duration
    last_4_seconds = duration - 4

    # Filter segments starting in the last 4 seconds
    last_segments = [i for i, t in enumerate(segment_starts) if t >= last_4_seconds]

    # Extract features for the last 4 seconds
    last_4_features = []
    for idx in last_segments:
        last_4_features.append({
            "start": segment_starts[idx],
            "pitch": list(pitches[idx]),
            "timbre": list(timbres[idx])
        })

    # Combine all data
    result = {
        "song_details": song_details,
        "last_4_seconds": last_4_features
    }

    print(result)
    musicbrainz = f['musicbrainz']
    print(f"Musicbrainz option: {list(musicbrainz.keys())}")

    print("==================================================================================================================================================================")

def extract_info(file_path):
    with h5py.File(file_path, 'r') as f:
        # analysis = f['analysis']
        # metadata = f['metadata']
        # musicbrainz = f['musicbrainz']
        song_title = f['metadata']['songs'][0][b'title'.decode('utf-8')].decode('utf-8')
        song_id = f['metadata']['songs'][0][b'song_id'.decode('utf-8')].decode('utf-8')
        return song_title, song_id


def process_folder(folder_path):
    #/Users/aux/Downloads/Mac_Storage/Smooth_Like_Butter/Audio_Analysis_MSD/MillionSongSubset/A
    #/Users/aux/Downloads/Mac_Storage/Smooth_Like_Butter/Audio_Analysis_MSD/MillionSongSubset/B

    song_dict = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.h5'):
                full_path = os.path.join(root, file)
                song_name, song_id = extract_info(full_path)
                song_dict[song_name] = song_id

    output_file = '/Users/aux/Downloads/Mac_Storage/Smooth_Like_Butter/Audio_Analysis_MSD/song_names.json'

    with open(output_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(song_dict, jsonfile, indent=4, ensure_ascii=False)

MSD_path = '/Users/aux/Downloads/Mac_Storage/Smooth_Like_Butter/Audio_Analysis_MSD/MillionSongSubset'
#intialization = process_folder(MSD_path)




 

        

