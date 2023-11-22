import requests
import json

ACCESS_TOKEN = "BQB8Dx50M7HtsZ5zXXXnDPtl85DygO4vKOHtLUBVztRkjzVk6GhgACC42fCkKiqldNLO7l8UrNRK5kcU-4WaUC0VnNSNZKE7Ua7Um_OLErKc7tcdF4w"
RAW_URL = input("Enter Song URL Here: ")
SONG_ID = RAW_URL[31:53]
SONG_ANALYSIS_URL = f"https://api.spotify.com/v1/audio-analysis/{SONG_ID}"

def get_audio_analysis():
    response = requests.get(SONG_ANALYSIS_URL,
        headers={"Authorization": f"Bearer {ACCESS_TOKEN}"
        }
        )
    json_resp = response.json()
    return json_resp

def save_json_to_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def main():
    analysis = get_audio_analysis()
    file_name_input = input("Enter Desired File Name: ")
    file_path = fr"C:\Users\pc46\Spotify_API_Project\Code_Spyder\Perfect_Transitions\{file_name_input}.json"
    save_json_to_file(analysis, file_path)
    print(f"Analysis: {analysis}")
    
if __name__ == '__main__':
    main()


