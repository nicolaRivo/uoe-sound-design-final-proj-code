  ###########################
 ### functions.py ###
###########################

import os
import math
import json
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
from pathlib import Path

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def log_elapsed_time(process_name_getter):
    """
    A decorator to log the elapsed time of a process.

    Args:
    process_name (str): Name of the process to be logged.

    Returns:
    A wrapper function that logs the time taken for the process.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):

            process_name = process_name_getter(*args, **kwargs)

            start_time = time.time()
            logging.info(f"--------Starting process: {process_name}")
            
            # Execute the function
            result = func(*args, **kwargs)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            logging.info(f"--------Finished process: {process_name}")
            logging.info(f"--------Elapsed time for {process_name}: {elapsed_time:.2f} seconds\n")
            
            return result
        return wrapper
    return decorator

##########################################################################################
##########################################################################################
##########################################################################################


def ensure_json_file_exists(json_path):
    """
    Checks if the JSON file exists at the specified path.
    If it doesn't exist, an empty JSON file with a valid JSON object is created.

    Parameters:
    json_path (str): The path where the JSON file should be checked or created.
    """
    # Check if the file exists
    if not os.path.exists(json_path):
        try:
            # If the file doesn't exist, create an empty JSON file with a valid JSON object
            with open(json_path, 'w') as json_file:
                json.dump({}, json_file)  # Create an empty JSON object
            print(f"JSON file created at: {json_path}")
        except Exception as e:
            print(f"Error creating JSON file at {json_path}: {e}")
    else:
        try:
            # Verify that the existing file is a valid JSON file
            with open(json_path, 'r') as json_file:
                json.load(json_file)
            print(f"JSON file already exists and is valid at: {json_path}")
        except json.JSONDecodeError:
            # If the file is not a valid JSON, overwrite it with an empty JSON object
            with open(json_path, 'w') as json_file:
                json.dump({}, json_file)  # Create an empty JSON object
            print(f"Invalid JSON file at {json_path} was overwritten with a valid empty JSON object.")
        except Exception as e:
            print(f"Error verifying JSON file at {json_path}: {e}")

##########################################################################################
##########################################################################################
##########################################################################################


def initialize_environment(working_directory, audio_files_dir, json_file_path, sr = 44100, debug=0, playback=0, file_names='', load_all_tracks=False):
    """
    Initializes the working environment by setting paths, loading audio files,
    and printing relevant information.

    Parameters:
    working_directory (str): The path to the working directory.
    audio_files_dir (str): The path to the directory containing audio files.
    json_file_path (str): The path to the JSON tracker.
    sr (int): sample rate. Default is 44100.
    debug (int): Flag to enable or disable debug mode. Default is 0 (disabled).
    playback (int): Flag to enable or disable audio playback. Default is 0 (disabled).
    file_names (arr): array with the names of the files to be searched. if left empty a default one will be assigned.
    load_all_tracks (bool): Flag to load all tracks in the folder. Default is False (load specified tracks only).
    
    Returns:
    list: A list of loaded audio files with their paths and durations.
    """

    # Change the current working directory
    os.chdir(working_directory)
    print("Current Working Directory:", os.getcwd())

    # Initialize arrays to store found audio file paths and loaded audio data
    audio_files_paths = []
    loaded_audio_files = []

    # Check if the directory exists
    if not os.path.exists(audio_files_dir):
        print(f"Directory {audio_files_dir} does not exist.")
        return []
    else:
        if debug == 1:
            # List all files in the directory
            all_files = os.listdir(audio_files_dir)
            print("Files in the directory:")
            for file in all_files:
                print(file, '\n')

    if load_all_tracks:
        # Load all audio files in the directory
        print("Loading all audio files in the directory...")
        for file in os.listdir(audio_files_dir):
            if file.endswith((".flac", ".wav", ".mp3", ".aiff")):
                audio_files_paths.append(os.path.join(audio_files_dir, file))
                print(f"Found file: {file}")
    else:
        not_found = 0
        print("Starting file search...")
    
        if file_names == '':
            file_names = [
                "chroma_test",
                "Music For Airports - Brian Eno",
                "Orphans - AFX",
                "Olson - Boards of Canada",
                "Don't Leave Me This Way - Harold Melvin & The Blue Notes",
                "Don't Stop Till You Get Enough - Michael Jackson",
                "Plantas Falsas - Bruxas",
                "Empire Ants - Gorillaz",
                "That's Us/Wild Combination - Arthur Russell",
                "This Is How We Walk on the Moon - Arthur Russell",
                "Gipsy Woman (She's Homeless) - Crystal Waters",
                "Warszava - David Bowie",
                "I Feel Love - 12\"Version - Donna Summer",
                "Workinonit - J Dilla",
                "I Swear, I Really Wanted to Make a 'Rap' Album but This Is Literally the Way the Wind Blew Me This Time - Andre 3000",
                "oo Licky - Matthew Herbert",
                "King Tubby Meets The Rockers Uptown - King Tubby",
                "Mood Swings - Little Simz"
            ]

        # Iterate over each file name
        for file_name in file_names:
            file_paths = {
                "flac": os.path.join(audio_files_dir, f"{file_name}.flac"),
                "wav": os.path.join(audio_files_dir, f"{file_name}.wav"),
                "mp3": os.path.join(audio_files_dir, f"{file_name}.mp3"),
                "aiff": os.path.join(audio_files_dir, f"{file_name}.aiff")
            }

            found_something = False
            for ext, path in file_paths.items():
                if os.path.exists(path):
                    audio_files_paths.append(path)
                    print(f"{file_name}.{ext} found at {path}!\n")
                    found_something = True
                    break  # Stop searching if one format is found

            if not found_something:
                not_found += 1
                print(f"{file_name} wasn't found in any supported format.\n")

        print("\n")
        if not_found == 0:
            print("All Tracks Found \n\n(: \n")
        else:
            print(f"{not_found} tracks were not found!\n\n): \n")

    print("Starting to load audio files...")
    print("================================\n")
   
    ensure_json_file_exists(json_file_path)

    # Loading the tracks into memory
    for path in audio_files_paths:
        try:
            y, sr = librosa.load(path, sr=sr)
            duration = librosa.get_duration(y=y, sr=sr)
            loaded_audio_files.append((y, sr, path, duration))
            print(f"Audio file correctly loaded from {path}: y = {y.shape} \n sr = {sr} \n duration = {duration} seconds \n\n")
        except Exception as e:
            print(f"Failed to load audio file from {path}: {e}")

    print("================================")
    print("Audio file loading complete.\n")

    # Sort the audio files by duration (shortest first)
    loaded_audio_files.sort(key=lambda x: x[3])

    print("Tracks sorted by duration (shortest to longest):")
    for i, (y, sr, path, duration) in enumerate(loaded_audio_files, 1):
        print(f"{i}. {os.path.basename(path)} - {duration:.2f} seconds")
    print("================================")
    print("\n")
    if playback == 1:
        # Playback each audio file found
        if loaded_audio_files:
            for _, _, path, _ in loaded_audio_files:
                print(f"Listen to {path}")
                # Implement playback function if needed
        else:
            print("No audio files were found to play.")

    return loaded_audio_files


##########################################################################################
##########################################################################################
##########################################################################################

@log_elapsed_time(lambda *args, **kwargs: f"STFT - {Path(args[0][2]).name}")
def process_stft_and_save(audio_file, json_path, save_path, plot_width = 50, plot_height = 20, jignore = False):
    """
    Processes the STFT of the given audio file and saves the spectrogram image.
    Updates the processing status in the JSON file.
    """
    
    # Load the tracking JSON file
    if os.path.exists(json_path) and jignore == False :
        with open(json_path, 'r') as f:
            tracking_data = json.load(f)
    else:
        tracking_data = {}
    #initialise variables
    y = audio_file[0]
    sr = audio_file[1]
    audio_path = audio_file[2]

    # Get the name of the song from the audio path
    song_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Check if this step has already been processed
    if tracking_data.get(song_name, {}).get("STFT_processed", False):
        print(f"STFT already processed for {song_name}. Skipping...")
        return
    

    # Set the STFT parameters
    n_fft = 22000 * 10
    hop_length = math.ceil(sr / 5)

    # Compute the STFT
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    D = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    # Create the song's directory if it doesn't exist
    song_dir = os.path.join(os.path.dirname(save_path), song_name)

    os.makedirs(song_dir, exist_ok=True)

    # Generate the spectrogram plot
    plt.figure(figsize=(plot_width, plot_height))
    librosa.display.specshow(D, sr=sr, x_axis='s', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Log-frequency Power Spectrogram - PARAMS: n_fft: {n_fft}, hop_length: {hop_length}')
    plt.ylim(20, 20000)  # Set y-axis limits to 20 Hz to 20,000 Hz

    # Customize x-axis to show time in minutes and seconds
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: '{:d}:{:02d}'.format(int(x // 60), int(x % 60))))

    plt.text(0.01, 0.95, f'Log-frequency Power Spectrogram - PARAMS: n_fft: {n_fft}, hop_length: {hop_length}',
             fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)

    # Save the plot
    output_file = os.path.join(song_dir, f"STFT_{song_name}.png")
    plt.savefig(output_file)
    plt.close()

    # Update the JSON tracking
    if song_name not in tracking_data:
        tracking_data[song_name] = {}
    tracking_data[song_name]["STFT_processed"] = True
    if jignore == False:
        with open(json_path, 'w') as f:
            json.dump(tracking_data, f, indent=4)
    
    print(f"STFT processing complete for {song_name}.\n Saved to {output_file}.")

##########################################################################################
##########################################################################################
##########################################################################################

@log_elapsed_time(lambda *args, **kwargs: f"Chromagram and CQT Spectrogram - {Path(args[0][2]).name}")
def process_chromagram_and_save(audio_file, json_path, save_path, plot_width=50, plot_height=20, jignore=False):
    """
    Processes the Chromagram and CQT spectrogram of the given audio file and saves the plots.
    Updates the processing status in the JSON file.
    
    Parameters:
    - audio_file: tuple containing (y, sr, path, duration)
    - json_path: str, path to the JSON tracking file
    - save_path: str, directory where plots will be saved
    - plot_width: int, width of the plot
    - plot_height: int, height of the plot
    - jignore: bool, whether to ignore updating the JSON file
    """
    
    # Load the tracking JSON file
    if os.path.exists(json_path) and not jignore:
        with open(json_path, 'r') as f:
            tracking_data = json.load(f)
    else:
        tracking_data = {}

    # Initialize variables
    y = audio_file[0]
    sr = audio_file[1]
    audio_path = audio_file[2]

    # Get the name of the song from the audio path
    song_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Check if this step has already been processed
    if tracking_data.get(song_name, {}).get("Chromagram_CQT_processed", False):
        print(f"Chromagram and CQT Spectrogram already processed for {song_name}. Skipping...")
        return

    # Set parameters for  and Chromagram
    hop_length = math.ceil(sr / 5)  # Adjust hop_length for time resolution
    bins_per_octave = 36  # Increase bins per octave for higher frequency resolution in CQT
    n_bins = 7 * bins_per_octave  # Total number of bins (7 octaves as an example)
    compression_ratio = 0.4  # Compression ratio for dynamic range compression

    # Harmonic-Percussive Source Separation (HPSS)
    y_harmonic, _ = librosa.effects.hpss(y)

    # Compute CQT spectrogram with increased resolution
    CQT = librosa.cqt(y=y_harmonic, sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave, n_bins=n_bins)
    CQT_db = librosa.amplitude_to_db(np.abs(CQT), ref=np.max)

    # Compute Chromagram directly from the harmonic component of the audio
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=hop_length, norm=None)

    # Perform global normalization on the chromagram
    chroma_max = chroma.max()
    if chroma_max > 0:
        chroma /= chroma_max

    # Apply dynamic range compression
    chroma = chroma ** compression_ratio

    # Create the song's directory if it doesn't exist
    song_dir = os.path.join(os.path.dirname(save_path), song_name)
    os.makedirs(song_dir, exist_ok=True)

    # Plotting CQT Spectrogram
    plt.figure(figsize=(plot_width, plot_height))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(CQT_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='cqt_note', bins_per_octave=bins_per_octave)
    plt.title(f'High-Resolution CQT Spectrogram for {song_name}')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()



    # Save the plot to a file
    output_file = os.path.join(song_dir, f"Chromagram_CQT_{song_name}.png")
    plt.savefig(output_file)
    plt.close()

    # Update the JSON tracking
    if song_name not in tracking_data:
        tracking_data[song_name] = {}
    tracking_data[song_name]["Chromagram_CQT_processed"] = True
    if not jignore:
        with open(json_path, 'w') as f:
            json.dump(tracking_data, f, indent=4)

    print(f"Chromagram and CQT Spectrogram processing complete for {song_name}.\n Saved to {output_file}.")

##########################################################################################
##########################################################################################
##########################################################################################

@log_elapsed_time(lambda *args, **kwargs: f"Self-Similarity Matrix and Chromagram - {Path(args[0][2]).name}")
def process_SSM_and_chr_and_save(audio_file, json_path, save_path, plot_width=12, plot_height=12, jignore=False):
    """
    Processes the self-similarity matrix (SSM) and Chromagram of the given audio file, applies diagonal enhancement, and saves the plots.
    Updates the processing status in the JSON file.
    
    Parameters:
    - audio_file: tuple containing (y, sr, path, duration)
    - json_path: str, path to the JSON tracking file
    - save_path: str, directory where plots will be saved
    - plot_width: int, width of the plot
    - plot_height: int, height of the plot
    - jignore: bool, whether to ignore updating the JSON file
    """
    
    def get_ssm(C):
        CNorm = np.sqrt(np.sum(C**2, axis=0))
        C = C / CNorm[None, :]
        return np.dot(C.T, C)

    def chunk_average(C, f):
        C2 = np.zeros((C.shape[0], C.shape[1] // f))
        for j in range(C2.shape[1]):
            C2[:, j] = np.mean(C[:, j * f:(j + 1) * f], axis=1)
        return C2

    def diagonally_enhance(D, K):
        M = D.shape[0] - K + 1
        S = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                avg = 0
                for k in range(K):
                    avg += D[i + k, j + k]
                S[i, j] = avg / K
        return S

    # Load the tracking JSON file
    if os.path.exists(json_path) and not jignore:
        with open(json_path, 'r') as f:
            tracking_data = json.load(f)
    else:
        tracking_data = {}

    # Initialize variables
    y = audio_file[0]
    sr = audio_file[1]
    audio_path = audio_file[2]

    # Set parameters for CQT and Chromagram
    hop_length = math.ceil(sr / 5)  # Adjust hop_length for time resolution
    bins_per_octave = 36  # Increase bins per octave for higher frequency resolution in CQT
    n_bins = 7 * bins_per_octave  # Total number of bins (7 octaves as an example)
    compression_ratio = 0.4  # Compression ratio for dynamic range compression
    
    # Get the name of the song from the audio path
    song_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Check if this step has already been processed
    if tracking_data.get(song_name, {}).get("SSM_processed", False):
        print(f"Self-Similarity Matrix already processed for {song_name}. Skipping...")
        return

    #---COMPUTING CHROMA---#

    # Harmonic-Percussive Source Separation (HPSS)
    y_harmonic, _ = librosa.effects.hpss(y)

    # Compute Chromagram directly from the harmonic component of the audio
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=hop_length, norm=None)

    # Perform global normalization on the chromagram
    chroma_max = chroma.max()
    if (chroma_max > 0):
        chroma /= chroma_max

    # Apply dynamic range compression
    chroma = chroma ** compression_ratio

    #---COMPUTING SSM---#

    # Compute Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_to_db = librosa.power_to_db(mel, ref=np.max)

    # Compute Self-Similarity Matrix
    D = get_ssm(chunk_average(mel_to_db, 43))

    # Diagonally enhance the matrix
    DDiag = diagonally_enhance(D, 4)

    # Plotting SSM and Chromagram
    plt.figure(figsize=(plot_width, plot_height))

    # Plot Self-Similarity Matrix
    plt.subplot(2, 1, 1)
    ax = plt.gca()
    im = ax.imshow(DDiag, cmap='magma')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.title(f'Self-Similarity Matrix: {song_name}')
    plt.colorbar(im, ax=ax)

    # Convert frames to seconds for x and y axis labels
    hop = 512
    frames_per_second = sr / hop
    num_frames = DDiag.shape[0]
    integer_ticks = np.arange(0, int(num_frames // frames_per_second) + 1)
    ax.set_xticks(integer_ticks * frames_per_second)
    ax.set_xticklabels(integer_ticks)
    ax.set_yticks(integer_ticks * frames_per_second)
    ax.set_yticklabels(integer_ticks)

    # Plotting Chromagram
    plt.subplot(2, 1, 2)
    librosa.display.specshow(chroma, sr=sr, hop_length=hop_length, x_axis='time', y_axis='chroma')
    plt.title(f'Chromagram for {song_name}')
    plt.colorbar()
    plt.tight_layout()

    plot_path = os.path.join(save_path, f"SSM_Chromagram_{song_name}.png")
    plt.savefig(plot_path)
    plt.close()

    # Update the JSON tracking
    if song_name not in tracking_data:
        tracking_data[song_name] = {}
    tracking_data[song_name]["SSM_processed"] = True
    if not jignore:
        with open(json_path, 'w') as f:
            json.dump(tracking_data, f, indent=4)

    print(f"Self-Similarity Matrix and Chromagram processing complete for {song_name}.\n Saved to {plot_path}.")

##########################################################################################
##########################################################################################
##########################################################################################

@log_elapsed_time(lambda *args, **kwargs: f"Mel Spectrogram - {Path(args[0][2]).name}")
def process_mel_spectrogram_and_save(audio_file, json_path, save_path, plot_width=12, plot_height=8, jignore=False):
    """
    Processes the Mel spectrogram of the given audio file, converts it to decibel scale, and saves the plot.
    Updates the processing status in the JSON file.
    
    Parameters:
    - audio_file: tuple containing (y, sr, path, duration)
    - json_path: str, path to the JSON tracking file
    - save_path: str, directory where plots will be saved
    - plot_width: int, width of the plot
    - plot_height: int, height of the plot
    - jignore: bool, whether to ignore updating the JSON file
    """
    
    # Load the tracking JSON file
    if os.path.exists(json_path) and not jignore:
        with open(json_path, 'r') as f:
            tracking_data = json.load(f)
    else:
        tracking_data = {}

    # Initialize variables
    y = audio_file[0]
    sr = audio_file[1]
    audio_path = audio_file[2]

    # Get the name of the song from the audio path
    song_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Check if this step has already been processed
    if tracking_data.get(song_name, {}).get("Mel_Spectrogram_processed", False):
        print(f"Mel Spectrogram already processed for {song_name}. Skipping...")
        return

    # Compute Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_to_db = librosa.power_to_db(mel, ref=np.max)

    # Plot and save Mel Spectrogram
    plt.figure(figsize=(plot_width, plot_height))
    librosa.display.specshow(mel_to_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    plt.title(f'Mel Spectrogram: {song_name}')
    plt.colorbar(format='%+2.0f dB')

    mel_spec_path = os.path.join(save_path, f"Mel_Spectrogram_{song_name}.png")
    plt.savefig(mel_spec_path)
    plt.close()

    # Update the JSON tracking
    if song_name not in tracking_data:
        tracking_data[song_name] = {}
    tracking_data[song_name]["Mel_Spectrogram_processed"] = True
    if not jignore:
        with open(json_path, 'w') as f:
            json.dump(tracking_data, f, indent=4)

    print(f"Mel Spectrogram processing complete for {song_name}.\n Saved to {mel_spec_path}.")


##########################################################################################
##########################################################################################
##########################################################################################



@log_elapsed_time(lambda *args, **kwargs: f"Harmonic CQT and Percussive SFFT - {Path(args[0][2]).name}")
def process_harmonic_cqt_and_percussive_sfft_and_save(audio_file, json_path, save_path, plot_width=12, plot_height=12, jignore=False):
    """
    Processes the Harmonic CQT and Percussive SFFT of the given audio file and saves the plots.
    Updates the processing status in the JSON file.
    
    Parameters:
    - audio_file: tuple containing (y, sr, path, duration)
    - json_path: str, path to the JSON tracking file
    - save_path: str, directory where plots will be saved
    - plot_width: int, width of the plot
    - plot_height: int, height of the plot
    - jignore: bool, whether to ignore updating the JSON file
    """
    
    # Load the tracking JSON file
    if os.path.exists(json_path) and not jignore:
        with open(json_path, 'r') as f:
            tracking_data = json.load(f)
    else:
        tracking_data = {}

    # Initialize variables
    y = audio_file[0]
    sr = audio_file[1]
    audio_path = audio_file[2]

    # Parameters for CQT
    octave_grain = 8
    bins_per_octave = 12 * octave_grain
    n_bins = bins_per_octave * 9

    # Get the name of the song from the audio path
    song_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Check if this step has already been processed
    if tracking_data.get(song_name, {}).get("Harmonic_CQT_Percussive_SFFT_processed", False):
        print(f"Harmonic CQT and Percussive SFFT already processed for {song_name}. Skipping...")
        return

    # Separate harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Compute the harmonic CQT
    CQT_harmonic = librosa.amplitude_to_db(np.abs(librosa.cqt(y_harmonic, sr=sr, bins_per_octave=bins_per_octave, n_bins=n_bins)), ref=np.max)

    # Compute the percussive SFFT
    D_percussive = librosa.amplitude_to_db(np.abs(librosa.stft(y_percussive)), ref=np.max)

    # Plot and save Harmonic CQT and Percussive SFFT
    plt.figure(figsize=(plot_width, plot_height))

    # Plot Harmonic CQT
    plt.subplot(2, 1, 1)
    librosa.display.specshow(CQT_harmonic, sr=sr, x_axis='time', y_axis='cqt_note', bins_per_octave=bins_per_octave)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{song_name} Harmonic CQT')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    # Plot Percussive SFFT
    plt.subplot(2, 1, 2)
    librosa.display.specshow(D_percussive, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{song_name} Percussive SFFT')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()

    plot_path = os.path.join(save_path, f"Harmonic_CQT_Percussive_SFFT_{song_name}.png")
    plt.savefig(plot_path)
    plt.close()

    # Update the JSON tracking
    if song_name not in tracking_data:
        tracking_data[song_name] = {}
    tracking_data[song_name]["Harmonic_CQT_Percussive_SFFT_processed"] = True
    if not jignore:
        with open(json_path, 'w') as f:
            json.dump(tracking_data, f, indent=4)

    print(f"Harmonic CQT and Percussive SFFT processing complete for {song_name}.\n Saved to {plot_path}.")

##########################################################################################
##########################################################################################
##########################################################################################


@log_elapsed_time(lambda *args, **kwargs: f"Harmonic CQT and Harmonic Mel - {Path(args[0][2]).name}")
def process_harmonic_cqt_and_harmonic_mel_and_save(audio_file, json_path, save_path, plot_width=12, plot_height=12, jignore=False):
    """
    Processes the Harmonic CQT and Harmonic Mel spectrogram of the given audio file and saves the plots.
    Updates the processing status in the JSON file.
    
    Parameters:
    - audio_file: tuple containing (y, sr, path, duration)
    - json_path: str, path to the JSON tracking file
    - save_path: str, directory where plots will be saved
    - plot_width: int, width of the plot
    - plot_height: int, height of the plot
    - jignore: bool, whether to ignore updating the JSON file
    """
    
    # Load the tracking JSON file
    if os.path.exists(json_path) and not jignore:
        with open(json_path, 'r') as f:
            tracking_data = json.load(f)
    else:
        tracking_data = {}

    # Initialize variables
    y = audio_file[0]
    sr = audio_file[1]
    audio_path = audio_file[2]

    # Parameters for Mel spectrogram
    n_fft = 2048 * 10
    hop_length = 44100 // 100
    n_mels = 256 * 10

    # Parameters for CQT
    octave_grain = 8
    bins_per_octave = 12 * octave_grain
    n_bins = bins_per_octave * 9

    # Get the name of the song from the audio path
    song_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Check if this step has already been processed
    if tracking_data.get(song_name, {}).get("Harmonic_CQT_Harmonic_Mel_processed", False):
        print(f"Harmonic CQT and Harmonic Mel spectrogram already processed for {song_name}. Skipping...")
        return

    # Separate harmonic and percussive components
    y_harmonic, _ = librosa.effects.hpss(y)

    # Compute the harmonic CQT
    CQT_harmonic = librosa.amplitude_to_db(np.abs(librosa.cqt(y_harmonic, sr=sr, bins_per_octave=bins_per_octave, n_bins=n_bins)), ref=np.max)

    # Compute the harmonic Mel spectrogram
    S_harmonic = librosa.feature.melspectrogram(y=y_harmonic, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    D_harmonic = librosa.amplitude_to_db(S_harmonic, ref=np.max)

    # Plot and save Harmonic CQT and Harmonic Mel spectrogram
    plt.figure(figsize=(plot_width, plot_height))

    # Plot Harmonic CQT
    plt.subplot(2, 1, 1)
    librosa.display.specshow(CQT_harmonic, sr=sr, x_axis='time', y_axis='cqt_note', bins_per_octave=bins_per_octave)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{song_name} Harmonic CQT')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    # Plot Harmonic Mel spectrogram
    plt.subplot(2, 1, 2)
    librosa.display.specshow(D_harmonic, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{song_name} Harmonic Mel Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()

    plot_path = os.path.join(save_path, f"Harmonic_CQT_Harmonic_Mel_{song_name}.png")
    plt.savefig(plot_path)
    plt.close()

   
    # Update the JSON tracking
    if song_name not in tracking_data:
        tracking_data[song_name] = {}
    tracking_data[song_name]["Harmonic_CQT_Harmonic_Mel_processed"] = True
    if not jignore:
        with open(json_path, 'w') as f:
            json.dump(tracking_data, f, indent=4)

    print(f"Harmonic CQT and Harmonic Mel processing complete for {song_name}.\n Saved to {plot_path}.")
