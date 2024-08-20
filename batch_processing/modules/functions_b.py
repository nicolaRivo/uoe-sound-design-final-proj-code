import os
import math
import json
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import pickle
import threading
from pathlib import Path

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s \n')


def log_elapsed_time(process_name_getter):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            stop_flag = threading.Event()  # Event to signal the thread to stop
            process_name = process_name_getter(*args, **kwargs)
            start_time = time.time()
            logging.info(f"***Starting process: {process_name}")

            # Start the time tracker in a separate thread
            def time_tracker():
                while not stop_flag.is_set():  # Run until stop_flag is set
                    elapsed_time = time.time() - start_time
                    print(f"{process_name} Elapsed time: {int(elapsed_time)} seconds", end='\r')
                    time.sleep(1)
            
            tracker_thread = threading.Thread(target=time_tracker)
            tracker_thread.daemon = True
            tracker_thread.start()

            result = func(*args, **kwargs)  # Run the actual function

            stop_flag.set()  # Signal the tracker thread to stop
            tracker_thread.join()  # Wait for the tracker thread to finish

            elapsed_time = time.time() - start_time
            print(f"\n{process_name_getter(*args, **kwargs)} completed in {elapsed_time:.2f} seconds")
            return result

        return wrapper
    return decorator

def ensure_json_file_exists(json_path):
    """
    Checks if the JSON file exists at the specified path.
    If it doesn't exist, an empty JSON file with a valid JSON object is created.
    """
    if not os.path.exists(json_path):
        try:
            with open(json_path, 'w') as json_file:
                json.dump({}, json_file)
            print(f"JSON file created at: {json_path}")
        except Exception as e:
            print(f"Error creating JSON file at {json_path}: {e}")
    else:
        try:
            with open(json_path, 'r') as json_file:
                json.load(json_file)
            print(f"JSON file already exists and is valid at: {json_path}")
        except json.JSONDecodeError:
            with open(json_path, 'w') as json_file:
                json.dump({}, json_file)
            print(f"Invalid JSON file at {json_path} was overwritten with a valid empty JSON object.")
        except Exception as e:
            print(f"Error verifying JSON file at {json_path}: {e}")

def initialize_environment(working_directory, audio_files_dir, json_file_path, sr=44100, debug=0, playback=0, file_names='', load_all_tracks=False):
    """
    Initializes the working environment by setting paths, loading audio files,
    and printing relevant information.
    """
    os.chdir(working_directory)
    print("Current Working Directory:", os.getcwd())

    audio_files_paths = []
    loaded_audio_files = []

    if not os.path.exists(audio_files_dir):
        print(f"Directory {audio_files_dir} does not exist.")
        return []
    else:
        if debug == 1:
            all_files = os.listdir(audio_files_dir)
            print("Files in the directory:")
            for file in all_files:
                print(file, '\n')

    if load_all_tracks:
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
                    break

            if not found_something:
                not_found += 1
                print(f"{file_name} wasn't found in any supported format.\n")

        if not_found == 0:
            print("All Tracks Found \n\n(: \n")
        else:
            print(f"{not_found} tracks were not found!\n\n): \n")

    print("Starting to load audio files...")
    print("================================\n")
   
    ensure_json_file_exists(json_file_path)

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

    loaded_audio_files.sort(key=lambda x: x[3])

    print("Tracks sorted by duration (shortest to longest):")
    for i, (y, sr, path, duration) in enumerate(loaded_audio_files, 1):
        print(f"{i}. {os.path.basename(path)} - {duration:.2f} seconds")
    print("================================\n")

    if playback == 1 and loaded_audio_files:
        for _, _, path, _ in loaded_audio_files:
            print(f"Listen to {path}")

    return loaded_audio_files

def is_processing_needed(track_name, json_data):
    """Check if any processing is needed for the track."""
    required_keys = [
        "STFT_processed",
        "Mel_Spectrogram_processed",
        "Harmonic_CQT_Percussive_SFFT_processed",
        "Harmonic_CQT_Harmonic_Mel_processed",
        "SSM_processed"
    ]
    if track_name in json_data:
        for key in required_keys:
            if not json_data[track_name].get(key, False):
                return True
        return False
    return True  # If track not found in JSON, assume processing is needed


def delete_cached_features(audio_file, cache_dir="feature_cache"):



    """
    Deletes the cached feature file for the given audio file.

    Parameters:
    - audio_file: tuple containing (y, sr, path, duration)
    - cache_dir: str, directory where cached feature files are stored

    Returns:
    - bool: True if the cache file was successfully deleted, False if the file was not found.
    """
    _, _, path, _ = audio_file
    song_name = os.path.splitext(os.path.basename(path))[0]
    feature_cache_path = os.path.join(cache_dir, f"{song_name}_features.pkl")

    if os.path.exists(feature_cache_path):
        os.remove(feature_cache_path)
        print(f"Cached features for {song_name} have been deleted from {feature_cache_path}")
        return True
    else:
        print(f"No cached features found for {song_name} at {feature_cache_path}")
        return False
    

@log_elapsed_time(lambda *args, **kwargs: f"Extract Audio Features - {Path(args[0][2]).name}")
def extract_audio_features(audio_file, cache_dir="feature_cache"):
    """
    Extract common features from the audio file to be reused across multiple functions.
    Features are cached to disk to avoid recomputation.
    """
    y, sr, path, duration = audio_file
    song_name = os.path.splitext(os.path.basename(path))[0]
    feature_cache_path = os.path.join(cache_dir, f"{song_name}_features.pkl")

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    # Check if features are already cached
    if os.path.exists(feature_cache_path):
        print(f"\n\nLoading cached features for {song_name} from {feature_cache_path}\n\n")
        try:
            with open(feature_cache_path, 'rb') as f:
                features = pickle.load(f)
            return features
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Warning: Could not load features from {feature_cache_path}. File may be corrupted or incomplete.")
            print(f"Error details: {e}")
            # Optionally, delete the corrupted cache file so it can be regenerated
            os.remove(feature_cache_path)
            print(f"Corrupted cache file {feature_cache_path} has been deleted.")
    
    # Compute features (perform only once)

    # Set the STFT parameters
    STFT_n_fft = 22000 * 10
    STFT_hop_length = math.ceil(sr / 5)

    # Parameters for Mel spectrogram
    mel_n_fft = 2048 * 10
    mel_hop_length = 44100 // 100
    n_mels = 256 * 10

    # Set parameters for Chromagram
    chr_hop_length = math.ceil(sr / 5)  # Adjust hop_length for time resolution

    # Set parameters for CQT
    cqt_hop_length = math.ceil(sr / 5)  # Adjust hop_length for time
    cqt_bins_per_octave = 36  # Increase bins per octave for higher frequency resolution in CQT
    cqt_n_bins = 7 * cqt_bins_per_octave  # Total number of bins (7 octaves as an example)
    
    # Feature extraction
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    S = librosa.stft(y, n_fft=STFT_n_fft, hop_length=STFT_hop_length)
    D = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=chr_hop_length, norm=None)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=mel_n_fft, hop_length=mel_hop_length, n_mels=n_mels)
    mel_to_db = librosa.power_to_db(mel, ref=np.max)
    CQT = librosa.cqt(y=y_harmonic, sr=sr, hop_length=cqt_hop_length, bins_per_octave=cqt_bins_per_octave, n_bins=cqt_n_bins)

    features = {
        'y': y,
        'sr': sr,
        'path': path,
        'y_harmonic': y_harmonic,
        'y_percussive': y_percussive,
        'S': S,
        'D': D,
        'chroma': chroma,
        'mel_to_db': mel_to_db,
        'CQT': CQT,
    }

    # Cache features to disk
    with open(feature_cache_path, 'wb') as f:
        pickle.dump(features, f)
    print(f"Features for {song_name} cached to {feature_cache_path}")

    return features

@log_elapsed_time(lambda *args, **kwargs: f"STFT - {Path(args[0]['path']).name}")
def process_stft_and_save(features, json_path, save_path, plot_width=50, plot_height=20, jignore=False):
    """
    Processes the STFT of the given audio file and saves the spectrogram image.
    Updates the processing status in the JSON file.
    """
    D = features['D']
    sr = features['sr']
    audio_path = features['path']
    song_name = os.path.splitext(os.path.basename(audio_path))[0]

    if os.path.exists(json_path) and not jignore:
        with open(json_path, 'r') as f:
            tracking_data = json.load(f)
    else:
        tracking_data = {}

    if tracking_data.get(song_name, {}).get("STFT_processed", False):
        print(f"STFT already processed for {song_name}. Skipping...")
        return
    
    song_dir = os.path.join(os.path.dirname(save_path), song_name)
    os.makedirs(song_dir, exist_ok=True)

    plt.figure(figsize=(plot_width, plot_height))
    librosa.display.specshow(D, sr=sr, x_axis='s', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Log-frequency Power Spectrogram - {song_name}')
    plt.ylim(20, 20000)
    plt.text(0.01, 0.95, f'Log-frequency Power Spectrogram - {song_name}',
             fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)

    output_file = os.path.join(song_dir, f"STFT_{song_name}.png")
    plt.savefig(output_file)
    plt.close()

    tracking_data.setdefault(song_name, {})["STFT_processed"] = True
    if not jignore:
        with open(json_path, 'w') as f:
            json.dump(tracking_data, f, indent=4)
    
    print(f"STFT processing complete for {song_name}.\n Saved to {output_file}.")
'''
@log_elapsed_time(lambda *args, **kwargs: f"Self-Similarity Matrix and Chromagram - {Path(args[0][2]).name}")
def process_SSM_and_chr_and_save(features, json_path, save_path, plot_width=12, plot_height=12, jignore=False):
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
    y = features['y']
    sr = features['sr']
    audio_path = features['path']


    # Set parameters for CQT and Chromagram
    hop_length = math.ceil(sr / 5)  # Adjust hop_length for time resolution
    compression_ratio = 0.4  # Compression ratio for dynamic range compression
    
    # Get the name of the song from the audio path
    song_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Check if this step has already been processed
    if tracking_data.get(song_name, {}).get("SSM_processed", False):
        print(f"Self-Similarity Matrix already processed for {song_name}. Skipping...")
        return

    #---COMPUTING CHROMA---#


    # Compute Chromagram directly from the harmonic component of the audio
    chroma = features['chroma'] 
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
'''
@log_elapsed_time(lambda *args, **kwargs: f"Self-Similarity Matrix and Chromagram - {Path(args[0]['path']).name}")
def process_SSM_and_chr_and_save(features, json_path, save_path, jignore=False):
    """
    Processes the self-similarity matrix (SSM) and Chromagram of the given audio file, applies diagonal enhancement, and saves the plots.
    Updates the processing status in the JSON file.
    
    Parameters:
    - audio_file: tuple containing (y, sr, path, duration)
    - json_path: str, path to the JSON tracking file
    - save_path: str, directory where plots will be saved
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
    y = features['y']
    sr = features['sr']
    audio_path = features['path']

    # Set parameters for CQT and Chromagram
    hop_length = math.ceil(sr / 5)  # Adjust hop_length for time resolution
    compression_ratio = 0.4  # Compression ratio for dynamic range compression
    
    # Get the name of the song from the audio path
    song_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Check if this step has already been processed
    if tracking_data.get(song_name, {}).get("SSM_processed", False):
        print(f"Self-Similarity Matrix already processed for {song_name}. Skipping...")
        return

    #---COMPUTING CHROMA---#

    # Compute Chromagram directly from the harmonic component of the audio
    chroma = features['chroma'] 
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

    # Determine the size of s
    s = 12  # You can adjust this value as needed for appropriate graph filling

    # Plotting SSM and Chromagram
    plt.figure(figsize=(s, s + s / 2))  # Height is s + s/2 to fit both plots

    # Plot Self-Similarity Matrix (Square)
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    im1 = ax1.imshow(DDiag, cmap='magma')
    ax1.set_aspect('equal')  # Ensure the plot is square
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')
    plt.title(f'Self-Similarity Matrix: {song_name}')
    #plt.colorbar(im1, ax=ax1)

    # Convert frames to seconds for x and y axis labels
    hop = 512
    frames_per_second = sr / hop
    num_frames = DDiag.shape[0]
    integer_ticks = np.arange(0, int(num_frames // frames_per_second) + 1)
    ax1.set_xticks(integer_ticks * frames_per_second)
    ax1.set_xticklabels(integer_ticks)
    ax1.set_yticks(integer_ticks * frames_per_second)
    ax1.set_yticklabels(integer_ticks)

    # Plotting Chromagram (Rectangle, width=s, height=s/2)
    ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
    chroma_img = librosa.display.specshow(chroma, sr=sr, hop_length=hop_length, x_axis='time', y_axis='chroma', ax=ax2)
    plt.title(f'Chromagram for {song_name}')
    #plt.colorbar(chroma_img, ax=ax2)

    # Set the aspect ratio and align the Chromagram to the top
    ax2.set_aspect(aspect=2)  # Aspect ratio 2:1 to make it half the height of SSM
    ax2.set_anchor('N')  # Align Chromagram plot to the top

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

@log_elapsed_time(lambda *args, **kwargs: f"Mel Spectrogram - {Path(args[0]['path']).name}")
def process_mel_spectrogram_and_save(features, json_path, save_path, plot_width=12, plot_height=8, jignore=False):
    """
    Processes the Mel spectrogram of the given audio file, converts it to decibel scale, and saves the plot.
    Updates the processing status in the JSON file.
    """
    mel_to_db = features['mel_to_db']
    sr = features['sr']
    audio_path = features['path']
    song_name = os.path.splitext(os.path.basename(audio_path))[0]

    if os.path.exists(json_path) and not jignore:
        with open(json_path, 'r') as f:
            tracking_data = json.load(f)
    else:
        tracking_data = {}

    if tracking_data.get(song_name, {}).get("Mel_Spectrogram_processed", False):
        print(f"Mel Spectrogram already processed for {song_name}. Skipping...")
        return

    plt.figure(figsize=(plot_width, plot_height))
    librosa.display.specshow(mel_to_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    plt.title(f'Mel Spectrogram: {song_name}')
    plt.colorbar(format='%+2.0f dB')

    mel_spec_path = os.path.join(save_path, f"Mel_Spectrogram_{song_name}.png")
    plt.savefig(mel_spec_path)
    plt.close()

    tracking_data.setdefault(song_name, {})["Mel_Spectrogram_processed"] = True
    if not jignore:
        with open(json_path, 'w') as f:
            json.dump(tracking_data, f, indent=4)

    print(f"Mel Spectrogram processing complete for {song_name}.\n Saved to {mel_spec_path}.")

@log_elapsed_time(lambda *args, **kwargs: f"Harmonic CQT and Percussive SFFT - {Path(args[0]['path']).name}")
def process_harmonic_cqt_and_percussive_sfft_and_save(features, json_path, save_path, plot_width=12, plot_height=12, jignore=False):
    """
    Processes the Harmonic CQT and Percussive SFFT of the given audio file and saves the plots.
    Updates the processing status in the JSON file.
    """
    y_harmonic = features['y_harmonic']
    y_percussive = features['y_percussive']
    sr = features['sr']
    audio_path = features['path']
    song_name = os.path.splitext(os.path.basename(audio_path))[0]

    bins_per_octave = 12 * 8
    n_bins = bins_per_octave * 9

    if os.path.exists(json_path) and not jignore:
        with open(json_path, 'r') as f:
            tracking_data = json.load(f)
    else:
        tracking_data = {}

    if tracking_data.get(song_name, {}).get("Harmonic_CQT_Percussive_SFFT_processed", False):
        print(f"Harmonic CQT and Percussive SFFT already processed for {song_name}. Skipping...")
        return

    CQT_harmonic = librosa.amplitude_to_db(np.abs(librosa.cqt(y_harmonic, sr=sr, bins_per_octave=bins_per_octave, n_bins=n_bins)), ref=np.max)
    D_percussive = librosa.amplitude_to_db(np.abs(librosa.stft(y_percussive)), ref=np.max)

    plt.figure(figsize=(plot_width, plot_height))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(CQT_harmonic, sr=sr, x_axis='time', y_axis='cqt_note', bins_per_octave=bins_per_octave)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{song_name} Harmonic CQT')

    plt.subplot(2, 1, 2)
    librosa.display.specshow(D_percussive, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{song_name} Percussive SFFT')
    plt.tight_layout()

    plot_path = os.path.join(save_path, f"Harmonic_CQT_Percussive_SFFT_{song_name}.png")
    plt.savefig(plot_path)
    plt.close()

    tracking_data.setdefault(song_name, {})["Harmonic_CQT_Percussive_SFFT_processed"] = True
    if not jignore:
        with open(json_path, 'w') as f:
            json.dump(tracking_data, f, indent=4)

    print(f"Harmonic CQT and Percussive SFFT processing complete for {song_name}.\n Saved to {plot_path}.")

@log_elapsed_time(lambda *args, **kwargs: f"Harmonic CQT and Harmonic Mel - {Path(args[0]['path']).name}")
def process_harmonic_cqt_and_harmonic_mel_and_save(features, json_path, save_path, plot_width=12, plot_height=12, jignore=False):
    """
    Processes the Harmonic CQT and Harmonic Mel spectrogram of the given audio file and saves the plots.
    Updates the processing status in the JSON file.
    """
    y_harmonic = features['y_harmonic']
    sr = features['sr']
    audio_path = features['path']
    song_name = os.path.splitext(os.path.basename(audio_path))[0]

    bins_per_octave = 12 * 8
    n_bins = bins_per_octave * 9
    n_fft = 2048 * 10
    hop_length = 44100 // 100
    n_mels = 256 * 10

    if os.path.exists(json_path) and not jignore:
        with open(json_path, 'r') as f:
            tracking_data = json.load(f)
    else:
        tracking_data = {}

    if tracking_data.get(song_name, {}).get("Harmonic_CQT_Harmonic_Mel_processed", False):
        print(f"Harmonic CQT and Harmonic Mel spectrogram already processed for {song_name}. Skipping...")
        return

    CQT_harmonic = librosa.amplitude_to_db(np.abs(librosa.cqt(y_harmonic, sr=sr, bins_per_octave=bins_per_octave, n_bins=n_bins)), ref=np.max)
    S_harmonic = librosa.feature.melspectrogram(y=y_harmonic, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    D_harmonic = librosa.amplitude_to_db(S_harmonic, ref=np.max)

    plt.figure(figsize=(plot_width, plot_height))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(CQT_harmonic, sr=sr, x_axis='time', y_axis='cqt_note', bins_per_octave=bins_per_octave)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{song_name} Harmonic CQT')

    plt.subplot(2, 1, 2)
    librosa.display.specshow(D_harmonic, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{song_name} Harmonic Mel Spectrogram')
    plt.tight_layout()

    plot_path = os.path.join(save_path, f"Harmonic_CQT_Harmonic_Mel_{song_name}.png")
    plt.savefig(plot_path)
    plt.close()

    tracking_data.setdefault(song_name, {})["Harmonic_CQT_Harmonic_Mel_processed"] = True
    if not jignore:
        with open(json_path, 'w') as f:
            json.dump(tracking_data, f, indent=4)

    print(f"Harmonic CQT and Harmonic Mel processing complete for {song_name}.\n Saved to {plot_path}.")

def process_all(audio_file, json_path, save_path, cache_dir="feature_cache"):
    """
    Unified function to process all the spectrograms and related features for an audio file.
    Uses caching to save and retrieve extracted features.
    """
    features = extract_audio_features(audio_file, cache_dir)
    process_stft_and_save(features, json_path, save_path, jignore = False) #YES!
    process_SSM_and_chr_and_save(features, json_path, save_path, jignore = False) #YES!
    process_mel_spectrogram_and_save(features, json_path, save_path, jignore = False) #YES!
    process_harmonic_cqt_and_percussive_sfft_and_save(features, json_path, save_path, jignore = False) #YES!
    process_harmonic_cqt_and_harmonic_mel_and_save(features, json_path, save_path, jignore = False) #NO!
    



'''
# @log_elapsed_time(lambda *args, **kwargs: f"Chromagram and CQT Spectrogram - {Path(args[0]['path']).name}")
# def process_chromagram_and_save(features, json_path, save_path, plot_width=50, plot_height=20, jignore=False):
#     """
#     Processes the Chromagram and CQT spectrogram of the given audio file and saves the plots.
#     Updates the processing status in the JSON file.
#     """
#     y_harmonic = features['y_harmonic']
#     sr = features['sr']
#     audio_path = features['path']
#     CQT = features['CQT']
#     chroma = features['chroma']
#     song_name = os.path.splitext(os.path.basename(audio_path))[0]

#     if os.path.exists(json_path) and not jignore:
#         with open(json_path, 'r') as f:
#             tracking_data = json.load(f)
#     else:
#         tracking_data = {}

#     if tracking_data.get(song_name, {}).get("Chromagram_CQT_processed", False):
#         print(f"Chromagram and CQT Spectrogram already processed for {song_name}. Skipping...")
#         return

#     song_dir = os.path.join(os.path.dirname(save_path), song_name)
#     os.makedirs(song_dir, exist_ok=True)

#     plt.figure(figsize=(plot_width, plot_height))
#     plt.subplot(2, 1, 1)
#     librosa.display.specshow(librosa.amplitude_to_db(np.abs(CQT), ref=np.max), sr=sr, x_axis='time', y_axis='cqt_note')
#     plt.title(f'High-Resolution CQT Spectrogram for {song_name}')
#     plt.colorbar(format='%+2.0f dB')
#     plt.tight_layout()

#     output_file = os.path.join(song_dir, f"Chromagram_CQT_{song_name}.png")
#     plt.savefig(output_file)
#     plt.close()

#     tracking_data.setdefault(song_name, {})["Chromagram_CQT_processed"] = True
#     if not jignore:
#         with open(json_path, 'w') as f:
#             json.dump(tracking_data, f, indent=4)

#     print(f"Chromagram and CQT Spectrogram processing complete for {song_name}.\n Saved to {output_file}.")
'''

'''
@log_elapsed_time(lambda *args, **kwargs: f"Self-Similarity Matrix and Chromagram - {Path(args[0]['path']).name}")
def process_SSM_and_chr_and_save(features, json_path, save_path, plot_width=30, plot_height=30, jignore=False):
    """
    Processes the self-similarity matrix (SSM) and Chromagram of the given audio file, applies diagonal enhancement, and saves the plots.
    Updates the processing status in the JSON file.
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

    y = features['y']
    sr = features['sr']
    chroma = features['chroma']
    audio_path = features['path']
    mel_to_db = ['mel_to_db']
    song_name = os.path.splitext(os.path.basename(audio_path))[0]

    if os.path.exists(json_path) and not jignore:
        with open(json_path, 'r') as f:
            tracking_data = json.load(f)
    else:
        tracking_data = {}

    if tracking_data.get(song_name, {}).get("SSM_processed", False):
        print(f"Self-Similarity Matrix already processed for {song_name}. Skipping...")
        return

    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_to_db = librosa.power_to_db(mel, ref=np.max)

    D = get_ssm(chunk_average(mel_to_db, 43))
    DDiag = diagonally_enhance(D, 4)

    plt.figure(figsize=(plot_width, plot_height))
    plt.subplot(2, 1, 1)
    ax = plt.gca()
    im = ax.imshow(DDiag, cmap='magma')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.title(f'Self-Similarity Matrix: {song_name}')
    plt.colorbar(im, ax=ax)

    plt.subplot(2, 1, 2)
    librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
    plt.title(f'Chromagram for {song_name}')
    plt.colorbar()
    plt.tight_layout()

    plot_path = os.path.join(save_path, f"SSM_Chromagram_{song_name}.png")
    plt.savefig(plot_path)
    plt.close()

    tracking_data.setdefault(song_name, {})["SSM_processed"] = True
    if not jignore:
        with open(json_path, 'w') as f:
            json.dump(tracking_data, f, indent=4)

    print(f"Self-Similarity Matrix and Chromagram processing complete for {song_name}.\n Saved to {plot_path}.")

@log_elapsed_time(lambda *args, **kwargs: f"Self-Similarity Matrix and Chromagram - {Path(args[0]['path']).name}")
def process_SSM_and_chr_and_save(features, json_path, save_path, plot_width=12, plot_height=12, jignore=False):
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
    y = features['y']
    sr = features['sr']
    audio_path = features['path']


    # Set parameters for CQT and Chromagram
    hop_length = math.ceil(sr / 5)  # Adjust hop_length for time resolution
    compression_ratio = 0.4  # Compression ratio for dynamic range compression
    
    # Get the name of the song from the audio path
    song_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Check if this step has already been processed
    if tracking_data.get(song_name, {}).get("SSM_processed", False):
        print(f"Self-Similarity Matrix already processed for {song_name}. Skipping...")
        return

    #---COMPUTING CHROMA---#


    # Compute Chromagram directly from the harmonic component of the audio
    chroma = features['chroma'] 
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

    # Set the aspect ratio to stretch the plot horizontally
    chroma_width = chroma.shape[1] * hop_length / sr
    num_frames = DDiag.shape[0]
    aspect_ratio = chroma_width / num_frames
    ax.set_aspect(aspect_ratio)
    
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.title(f'Self-Similarity Matrix: {song_name}')
    plt.colorbar(im, ax=ax)

    # Convert frames to seconds for x and y axis labels
    hop = 512
    frames_per_second = sr / hop
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

@log_elapsed_time(lambda *args, **kwargs: f"Self-Similarity Matrix and Chromagram - {Path(args[0]['path']).name}")
def process_SSM_and_chr_and_save(features, json_path, save_path, plot_width=12, plot_height=12, jignore=False):
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
    y = features['y']
    sr = features['sr']
    audio_path = features['path']

    # Set parameters for CQT and Chromagram
    hop_length = math.ceil(sr / 5)  # Adjust hop_length for time resolution
    compression_ratio = 0.4  # Compression ratio for dynamic range compression
    
    # Get the name of the song from the audio path
    song_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Check if this step has already been processed
    if tracking_data.get(song_name, {}).get("SSM_processed", False):
        print(f"Self-Similarity Matrix already processed for {song_name}. Skipping...")
        return

    #---COMPUTING CHROMA---#

    # Compute Chromagram directly from the harmonic component of the audio
    chroma = features['chroma'] 
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
    # Adjust the height to maintain the square aspect ratio for the SSM
    num_frames = DDiag.shape[0]
    chroma_width_seconds = chroma.shape[1] * hop_length / sr
    aspect_ratio = chroma_width_seconds / num_frames

    adjusted_height = plot_width * aspect_ratio  # Adjust the height to maintain the square aspect ratio

    plt.figure(figsize=(plot_width, adjusted_height + plot_height))  # Combine adjusted height with Chromagram plot height

    # Plot Self-Similarity Matrix
    plt.subplot(2, 1, 1)
    ax = plt.gca()
    im = ax.imshow(DDiag, cmap='magma')
    ax.set_aspect('equal')  # Force the aspect ratio to be 1:1 to make the plot square
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.title(f'Self-Similarity Matrix: {song_name}')
    plt.colorbar(im, ax=ax)

    # Convert frames to seconds for x and y axis labels
    hop = 512
    frames_per_second = sr / hop
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
'''

'''
def extract_audio_features(audio_file, cache_dir="feature_cache"):
    """
    Extract common features from the audio file to be reused across multiple functions.
    Features are cached to disk to avoid recomputation.
    """
    y, sr, path, duration = audio_file
    song_name = os.path.splitext(os.path.basename(path))[0]
    feature_cache_path = os.path.join(cache_dir, f"{song_name}_features.pkl")

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    # Check if features are already cached
    if os.path.exists(feature_cache_path):
        print(f"\n\nLoading cached features for {song_name} from {feature_cache_path}\n\n")
        with open(feature_cache_path, 'rb') as f:
            features = pickle.load(f)
        return features

    # Compute features (perform only once)

    # Set the STFT parameters
    STFT_n_fft = 22000 * 10
    STFT_hop_length = math.ceil(sr / 5)


    # Parameters for Mel spectrogram
    mel_n_fft = 2048 * 10
    mel_hop_length = 44100 // 100
    n_mels = 256 * 10

    # Set parameters for Chromagram
    chr_hop_length = math.ceil(sr / 5)  # Adjust hop_length for time resolution

    # Set parameters for CQT
    cqt_hop_length = math.ceil(sr / 5)  # Adjust hop_length for time
    cqt_bins_per_octave = 36  # Increase bins per octave for higher frequency resolution in CQT
    cqt_n_bins = 7 * cqt_bins_per_octave  # Total number of bins (7 octaves as an example)
    
    
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    S = librosa.stft(y, n_fft = STFT_n_fft, hop_length = STFT_hop_length)
    D = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=chr_hop_length, norm=None)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft = mel_n_fft, hop_length= mel_hop_length, n_mels = n_mels)
    mel_to_db = librosa.power_to_db(mel, ref=np.max)
    CQT = librosa.cqt(y=y_harmonic, sr=sr, hop_length=cqt_hop_length, bins_per_octave=cqt_bins_per_octave, n_bins=cqt_n_bins)

    features = {
        'y': y,
        'sr': sr,
        'path': path,
        'y_harmonic': y_harmonic,
        'y_percussive': y_percussive,
        'S': S,
        'D': D,
        'chroma': chroma,
        'mel_to_db': mel_to_db,
        'CQT': CQT,
    }

    # Cache features to disk
    with open(feature_cache_path, 'wb') as f:
        pickle.dump(features, f)
    print(f"Features for {song_name} cached to {feature_cache_path}")

    return features
'''

'''
def log_elapsed_time(process_name_getter):

    """
    A decorator to log the elapsed time of a process.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            process_name = process_name_getter(*args, **kwargs)
            start_time = time.time()
            logging.info(f"***Starting process: {process_name}")
            
            # Execute the function
            result = func(*args, **kwargs)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            logging.info(f"***Finished process: {process_name}")
            logging.info(f"***Elapsed time for {process_name}: {elapsed_time:.2f} seconds\n")
            
            return result
        return wrapper
    return decorator
'''


