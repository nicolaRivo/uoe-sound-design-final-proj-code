import os
import json
from modules.utility_functions import *
from modules.stem_split import make_drumTrack_and_drumLessTrack

# Set global plot size
plot_width, plot_height = 50, 20

# Define working directories and paths
working_directory = "/Users/nicola/Documents/MSc SOund Design 20-24/Final Project"
# audio_files_root_directory = '/Volumes/Nicola Projects SSD1TB/'
# audio_files_subdir = "/Sound/Audio_Files"

audio_files_root_directory = '/Volumes/Arancione'
audio_files_subdir = "Musica Flac"
graphs_location = "myFavouritesWithStems"
graphs_root_dir = '/Volumes/Nicola Projects SSD1TB/graphs'
graphs_dir = os.path.join(graphs_root_dir, graphs_location)
cache_dir = os.path.join(graphs_dir, '.feature_cache')
stems_dir = os.path.join(graphs_dir, '.stems')
json_file_path = os.path.join(graphs_dir, '.tracking_file.json')
audio_files_dir = os.path.join(audio_files_root_directory, audio_files_subdir)

# Check and create directories as needed
directories_to_check = [working_directory, audio_files_dir, graphs_dir, cache_dir, stems_dir]
for directory in directories_to_check:
    ensure_directory_exists(directory)

# Audio processing settings
sr = 44100
store_cache = True
use_artist_name = False

surprise_mode = True

if surprise_mode:
    store_cache=False
    graphs_dir = os.path.join(graphs_root_dir, 'surprise_mode')
    use_artist_name = True


# Configuration for stem processing
config_stems_drums = {
    "process_stft_and_save": True,
    "process_SSM_and_chr_and_save": False,
    "process_mel_spectrogram_and_save": True,
    "process_harmonic_cqt_and_percussive_sfft_and_save": False,
    "process_harmonic_cqt_and_harmonic_mel_and_save": False
}

config_stems_drumless = {
    "process_stft_and_save": True,
    "process_SSM_and_chr_and_save": True,
    "process_mel_spectrogram_and_save": True,
    "process_harmonic_cqt_and_percussive_sfft_and_save": False,
    "process_harmonic_cqt_and_harmonic_mel_and_save": True
}

def main():
    print("Starting batch processing...")

    ensure_json_file_exists(json_file_path)
    
    # Load the JSON tracking file
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)

    start_index = 0
    chunk_size = 20

    while start_index is not None:
        # Load a chunk of audio files, with shuffling enabled
        loaded_audio_files, next_start_index = load_tracks(
            working_directory,
            audio_files_dir,
            json_file_path,
            sr=sr,
            load_all_tracks=True,
            start_index=start_index,
            chunk_size=chunk_size,
            shuffle=True,  # Enable shuffling
            surprise_mode=surprise_mode,
            use_artist_name=use_artist_name,
        
        )

        # Process each audio file in the loaded chunk
        for audio_file in loaded_audio_files:
            # Get the name of the song from the audio path
            song_name = audio_file[4]

            # Create the song's directory if it doesn't exist                 
            song_dir = os.path.join(graphs_dir, song_name)
            os.makedirs(song_dir, exist_ok=True)

            # Separate the stems and return a drum track and a drumless track
            if is_processing_needed(song_name, json_data):
                stem_split = make_drumTrack_and_drumLessTrack(audio_file, parent_dir=stems_dir)
               

                process_all(stem_split['drumless'], json_file_path, song_dir, cache_dir=cache_dir, config=config_stems_drumless)
                process_all(stem_split['drums'], json_file_path, song_dir, cache_dir=cache_dir, config=config_stems_drums)
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
