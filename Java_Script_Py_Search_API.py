import requests
import json

ACCESS_TOKEN = "BQCJ6OWaB-PaRp5LqLMgUnxTEQG6uorof1u1lNLvr7prQEkMlxjWgWaQF5MJO09KhBdnH8JViyXgVgHaBGKUuyGMlcPDcj15g_A3U0TlhlEUQ1OoEA0"
SONG_ANALYSIS_URL = "https://api.spotify.com/v1/audio-analysis/1otG6j1WHNvl9WgXLWkHTo"

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
    file_path = r"C:\Users\pc46\Spotify_API_Project\Code_Spyder\Perfect_Transitions\analysis.json"
    save_json_to_file(analysis, file_path)
    print(f"Analysis: {analysis}")
    
if __name__ == '__main__':
    main()
        