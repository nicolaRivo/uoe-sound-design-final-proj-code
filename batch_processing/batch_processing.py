  ###########################
 ### batch_processing.py ###
###########################

import os
from modules.functions_b import  *

# Set global plot size
plot_width, plot_height = 50, 20

working_directory = "/Users/nicola/Documents/MSc SOund Design 20-24/Final Project/"
audio_files_subdir = "/Sound/Audio_Files/"
graphs_location = "/Graphs/Batch All_B/"
os.chdir(working_directory) 
json_file = "/Graphs/Batch All_B/tracking_file.json"
sr = 44100
store_cache = True

def main():
    #i'm in
    print("hwrd")
    #file_names=['Olson - Boards of Canada', "Don't Leave Me This Way - Harold Melvin & The Blue Notes"]
    json_file_path = os.getcwd() + json_file 
    json_file_path ='/Volumes/Nicola Projects SSD1TB/graphs/tracking_file.json'
    working_directory = os.getcwd()
    print (working_directory)
    audio_files_dir = working_directory + audio_files_subdir
    graphs_dir='/Volumes/Nicola Projects SSD1TB/graphs/'
    cache_dir = '/Volumes/Nicola Projects SSD1TB/graphs/feature_cache'
    print(audio_files_dir)

    # Define the paths to your audio file and JSON tracking file


    #load the files
    loaded_audio_files = initialize_environment(working_directory, audio_files_dir, json_file_path, sr = sr)

    json_file_path = os.getcwd() + json_file 
    json_file_path = '/Volumes/Nicola Projects SSD1TB/graphs/tracking_file.json'
    working_directory = os.getcwd()
    audio_files_dir = working_directory + audio_files_subdir
    graphs_dir = '/Volumes/Nicola Projects SSD1TB/graphs/'
    cache_dir = '/Volumes/Nicola Projects SSD1TB/graphs/feature_cache'

    # Load the JSON tracking file
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)

    # Load the audio files
    loaded_audio_files = initialize_environment(working_directory, audio_files_dir, json_file_path, sr=sr)

    for audio_file in loaded_audio_files:
        # Get the name of the song from the audio path
        song_name = os.path.splitext(os.path.basename(audio_file[2]))[0]
                          
        # Create the song's directory if it doesn't exist                 
        song_dir = os.path.join(os.path.dirname(graphs_dir), song_name)
        os.makedirs(song_dir, exist_ok=True)

        # Check if processing is needed
        if is_processing_needed(song_name, json_data):
            process_all(audio_file, json_file_path, song_dir, cache_dir=cache_dir)
        else:
            print(f"Skipping {song_name}, all graphs are already processed.")

        # Clean cache if necessary
        if not store_cache:
            delete_cached_features(audio_file, cache_dir="feature_cache")


if __name__ == "__main__":
    main()