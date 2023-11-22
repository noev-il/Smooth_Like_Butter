import json

file_name = input("Enter File Name: ")
r_literal = fr'C:\Users\pc46\Spotify_API_Project\Code_Spyder\Perfect_Transitions\{file_name}.json'
with open(r_literal, 'r') as f:
    json_object = json.loads(f.read())
    
track_duration = float(json_object['track']["duration"])
new_dict = {}
first_section = json_object['sections'][0]['start']

for y in range(len(json_object['sections'])):
    start_section = json_object['sections'][y]['start']
    if start_section <= 6:
        section_content = json_object['sections'][y]
        new_dict[y] = section_content

count = 0
for a in range(len(json_object['segments'])):
    start_segment = json_object['segments'][a]
    if start_segment['start'] <= 6.5:
        count += 1
print(count)