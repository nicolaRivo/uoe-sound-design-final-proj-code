import os
import json
from modules.functions_b import *

# Set global plot size
plot_width, plot_height = 50, 20

working_directory = "/Users/nicola/Documents/MSc SOund Design 20-24/Final Project/"
audio_files_subdir = "/Sound/Audio_Files/"
graphs_location = "/Graphs/Batch All_1.0/"
os.chdir(working_directory)
json_file = graphs_location + "/tracking_file.json"
sr = 44100
store_cache = False
surprise_mode = True
def main():
    print("Starting batch processing...")

    json_file_path = os.getcwd() + json_file 
    json_file_path = '/Volumes/Nicola Projects SSD1TB/graphs/tracking_file.json'
    working_directory = os.getcwd()
    audio_files_dir = working_directory + audio_files_subdir
    audio_files_dir = '/Volumes/Nicola Projects SSD1TB/analysis'
    graphs_dir = '/Volumes/Nicola Projects SSD1TB/graphs/'
    cache_dir = '/Volumes/Nicola Projects SSD1TB/graphs/feature_cache'

    # Load the JSON tracking file
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)

    start_index = 0
    chunk_size = 20

    while start_index is not None:
        # Load a chunk of audio files, with shuffling enabled
        loaded_audio_files, next_start_index = initialize_environment(
            working_directory,
            audio_files_dir,
            json_file_path,
            sr=sr,
            load_all_tracks=True,
            start_index=start_index,
            chunk_size=chunk_size,
            shuffle=True,  # Enable shuffling
            surprise_mode=surprise_mode
        )

        # Process each audio file in the loaded chunk
        for audio_file in loaded_audio_files:
            # Get the name of the song from the audio path
            song_name = audio_file[4]
                              
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
                delete_cached_features(audio_file, cache_dir=cache_dir)

        # Update the start index for the next chunk
        start_index = next_start_index

    print("Batch processing complete.")

if __name__ == "__main__":
    main()