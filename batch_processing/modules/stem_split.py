import os
import numpy as np
from spleeter.separator import Separator
from pydub import AudioSegment
import random
import librosa

def load_tracks(working_directory, audio_files_dir='', json_file_path='', sr=44100, debug=0, playback=0, file_names='', load_all_tracks=False, start_index=0, chunk_size=20, shuffle=False, use_artist_name=True, surprise_mode=False):
    """
    Initializes the working environment by setting paths, loading audio files,
    and printing relevant information in chunks. Handles surprise mode by selecting
    a single track, encrypting its name, and saving related information.
    """
    os.chdir(working_directory)
    print("Current Working Directory:", os.getcwd())

    if load_all_tracks and not audio_files_dir:
        audio_files_dir = input("Please provide the directory path containing the audio files: ")

    if not os.path.exists(audio_files_dir):
        print(f"Directory {audio_files_dir} does not exist.")
        return [], None


    if surprise_mode:
        # Seed with the current time to ensure different results each time
        random.seed(time.time())

        # Explore all files and select one randomly
        audio_files_paths = explore_audio_files(audio_files_dir)
        if not audio_files_paths:
            print("No audio files found.")
            return [], None
        
        # Select a random track
        selected_file = random.choice(audio_files_paths)
        
        # Encrypt the track name
        track_name = os.path.splitext(os.path.basename(selected_file))[0]
        key = 'hiddenTrack'
        encrypted_name = encrypt(selected_file, key)
        encrypted_name_hash = string_to_5_letter_hash(encrypted_name)

        
        # Create /surprise_file/ directory if it doesn't exist
        surprise_dir = os.path.join(working_directory, "surprise_file")
        os.makedirs(surprise_dir, exist_ok=True)
        
        # Save the encrypted name and decryption key
        encrypted_file_path = os.path.join(surprise_dir, ".encrypted-name.txt")
        with open(encrypted_file_path, 'w') as f:
            f.write(f"Encrypted HASH: {encrypted_name_hash}\n")
            f.write(f"Encrypted Name: {encrypted_name}\n")
            f.write(f"Decryption Key: {key}\n")
        
        # Substitute file name with 'secret - {first 5 digits of the encrypted name}'
        secret_name = f"secret - {encrypted_name_hash}"
        secret_file_path = os.path.join(os.path.dirname(selected_file), f"{track_name}{os.path.splitext(selected_file)[1]}")
        os.rename(selected_file, secret_file_path)
        print(f"Surprise mode activated. Track: {secret_name}")
        
        # Only this track will be processed
        audio_files_paths = [secret_file_path]
        save_dir = surprise_dir  # All outputs will be saved here

    else:
        # Regular mode, loading all tracks or specific files
        if load_all_tracks:
            print("Exploring all audio files in the directory and subdirectories...")
            audio_files_paths = explore_audio_files(audio_files_dir)
        else:
            audio_files_paths = []
            not_found = 0
            print("Starting file search...")

            if file_names == '':
                file_names = [
                    # Default file names list as provided earlier
                ]

            for file_name in file_names:
                found_something = False

                for root, _, files in os.walk(audio_files_dir):
                    file_paths = {
                        "flac": os.path.join(root, f"{file_name}.flac"),
                        "wav": os.path.join(root, f"{file_name}.wav"),
                        "mp3": os.path.join(root, f"{file_name}.mp3"),
                        "aiff": os.path.join(root, f"{file_name}.aiff")
                    }

                    for ext, path in file_paths.items():
                        if os.path.exists(path):
                            audio_files_paths.append(path)
                            print(f"{file_name}.{ext} found at {path}!\n")
                            found_something = True
                            break

                    if found_something:
                        break

                if not found_something:
                    not_found += 1
                    print(f"{file_name} wasn't found in any supported format.\n")

            if not_found == 0:
                print("All Tracks Found \n\n(: \n")
            else:
                print(f"{not_found} tracks were not found!\n\n): \n")
        
        # Shuffle the list if requested
        if shuffle:
            random.shuffle(audio_files_paths)

        save_dir = os.path.join(working_directory, "output_files")

    # Process files in chunks
    total_files = len(audio_files_paths)
    end_index = min(start_index + chunk_size, total_files)

    if start_index >= total_files:
        print("All files have been processed.")
        return [], None

    chunk_files_paths = audio_files_paths[start_index:end_index]
    loaded_audio_files = []

    print(f"Processing files from {start_index + 1} to {end_index} of {total_files}...")

    for path in chunk_files_paths:
        try:
            y, sr = librosa.load(path, sr=sr)
            duration = librosa.get_duration(y=y, sr=sr)
            # Extract artist name from the parent folder if use_artist_name is True
            if surprise_mode:
                song_name = encrypted_name_hash
            elif use_artist_name:
                artist_name = os.path.basename(os.path.dirname(path))
                song_name = f"{artist_name} - {os.path.splitext(os.path.basename(path))[0]}"
            else:
                song_name = os.path.splitext(os.path.basename(path))[0]
                
            loaded_audio_files.append((y, sr, path, duration, song_name))
            

            if surprise_mode:
                print(f"Audio file correctly loaded: \nsr = {sr} \n duration = {duration} seconds \n\n")           
            else:
                print(f"Audio file correctly loaded from {path}: \nsr = {sr} \n duration = {duration} seconds \n\n")
        except Exception as e:
            print(f"Failed to load audio file from {path}: {e}")

    print("================================")
    print(f"Processed {len(loaded_audio_files)} files in this chunk.\n")

    loaded_audio_files.sort(key=lambda x: x[3])

    print("Tracks sorted by duration (shortest to longest):")
    for i, (y, sr, path, duration, song_name) in enumerate(loaded_audio_files, start_index + 1):
        if surprise_mode:
            print(f"{i}. {encrypted_name_hash} - {duration:.2f} seconds")
        else:
            print(f"{i}. {song_name} - {duration:.2f} seconds")
    print("================================\n")

    if playback == 1 and loaded_audio_files:
        for _, _, path, _, _ in loaded_audio_files:
            print(f"Listen to {path}")

    # Determine if there are more files to process
    next_start_index = end_index if end_index < total_files else None

    return loaded_audio_files, next_start_index

import os
from pydub import AudioSegment
from spleeter.separator import Separator

def make_drumTrack_and_drumLessTrack(input_audio, parent_dir=None, separator=None):
    """
    Isolates the drums from an audio track using Spleeter and creates a drumless track by phase inversion.
    Saves the drumless track and drum track.

    Parameters:
    input_audio (tuple): Tuple containing the loaded audio file info (y, sr, path, duration, song_name).
    parent_dir (str): Directory where the output folder will be saved.
    separator (Separator): Optional. A pre-initialized Spleeter Separator object.

    Returns:
    dict: A dictionary with keys 'drumless' and 'drums', each containing an array with 
          [audio data (y), sample rate (sr), file path, duration (0), song_name].
    """
    # Ensure the parent directory is provided
    if parent_dir is None:
        raise ValueError("Parent directory must be provided.")

    song_name = input_audio[4]
    output_folder = os.path.join(parent_dir, song_name)
    ensure_directory_exists(output_folder)

    # Initialize Spleeter separator if not provided
    if separator is None:
        separator = Separator('spleeter:4stems')

    # Separate the audio into stems
    #separator.separate_to_file(input_audio[2], parent_dir)
    separator.separate_to_file(input_audio[2], output_folder, filename_format = "{instrument}.{codec}")

    # Process the drumless track
    drumless_path = mix_down_without_drums(input_audio, parent_dir, output_folder)
    drumless_audio = AudioSegment.from_file(drumless_path)
    drumless_export_path = os.path.join(output_folder, song_name + '_drumless.wav')
    drumless_audio.export(drumless_export_path, format="wav")

    # Load the drumless track
    y_drumless, sr_drumless = librosa.load(drumless_export_path, sr=None)
    drumless_data = [y_drumless, sr_drumless, drumless_export_path, 0, song_name + '_drumless']

    # Load the drum track (assuming it's saved as 'drums.wav' in the output folder)
    drums_path = os.path.join(output_folder, 'drums.wav')
    y_drums, sr_drums = librosa.load(drums_path, sr=None)
    drum_data = [y_drums, sr_drums, drums_path, 0, song_name + '_drums']

    # Create the stem_split dictionary
    stem_split = {
        'drumless': drumless_data,
        'drums': drum_data
    }

    return stem_split


def ensure_directory_exists(directory):
    """
    Ensures the specified directory exists by creating it if necessary.
    
    Parameters:
    directory (str): The directory path to check or create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def separate_audio_stems(input_audio_path, output_folder, separator):
    """
    Uses Spleeter to separate audio into 4 stems (vocals, bass, drums, others).

    Parameters:
    input_audio_path (str): Path to the input audio file.
    output_folder (str): Directory where the separated stems will be saved.
    separator (Separator): A Spleeter Separator object.
    """
    separator.separate_to_file(input_audio_path, output_folder)

def mix_down_without_drums(input_audio, parent_dir, output_folder):
    """
    Mixes down the three stems (vocals, bass, and other) excluding the drums
    and saves the result as a new audio file.

    Parameters:
    input_audio (tuple): Tuple containing the loaded audio file info (y, sr, path, duration, song_name).
    output_folder (str): Directory to save the mixed-down track.

    Returns:
    str: Path to the mixed-down audio file without drums.
    """
    # separator = Separator('spleeter:4stems')
    # separator.separate_to_file(input_audio[2], output_folder)

    # Paths to the separated stems
    vocals_track_path = os.path.join(output_folder, 'vocals.wav')
    bass_track_path = os.path.join(output_folder, 'bass.wav')
    other_track_path = os.path.join(output_folder, 'other.wav')
    
    # Ensure all necessary stems exist
    if not all(os.path.exists(path) for path in [vocals_track_path, bass_track_path, other_track_path]):
        raise FileNotFoundError("One or more stems (vocals, bass, other) not found after separation.")

    # Load the stems as AudioSegment objects
    vocals_audio = AudioSegment.from_file(vocals_track_path)
    bass_audio = AudioSegment.from_file(bass_track_path)
    other_audio = AudioSegment.from_file(other_track_path)

    # Mix down the stems (excluding drums)
    mixed_audio = vocals_audio.overlay(bass_audio).overlay(other_audio)

    # Export the mixed-down track
    mixed_down_path = os.path.join(output_folder, input_audio[4] + '_mixed_down_no_drums.wav')
    mixed_audio.export(mixed_down_path, format="wav")

    return mixed_down_path
