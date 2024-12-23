import h5py
import numpy as np
import pandas as pd
import os
import json
# Open the HDF5 file
h5_file = h5py.File('/Users/aux/Downloads/Mac_Storage/Smooth_Like_Butter/Audio_Analysis_MSD/MillionSongSubset/A/A/A/TRAAAAW128F429D538.h5', 'r')
with h5_file as f:
    print("==================================================================================================================================================================")
    print(f"High level options: {list(f.keys())}")

    analysis = f['analysis']
    print(f"Analysis option: {list(analysis.keys())}", f"{analysis['bars_confidence']}")

    metadata = f['metadata']
    print(f"Metadata option: {list(metadata.keys())}")

    song_title = f['metadata']['songs'][0][b'title'.decode('utf-8')].decode('utf-8')
    print("Song Title:", song_title)


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
intialization = process_folder(MSD_path)




 

        

