import os
import math
import json
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .timing_utils import *
import pickle


@log_elapsed_time(lambda *args, **kwargs: f"Extract Audio Features - {Path(args[0][4])}")
def extract_audio_features(audio_file, cache_dir="feature_cache"):
    """
    Extract common features from the audio file to be reused across multiple functions.
    Features are cached to disk to avoid recomputation.
    """
    y, sr, path, duration, name = audio_file
    song_name = name
    feature_cache_path = os.path.join(cache_dir, f"{song_name}_features.pkl")

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    # Check if features are already cached
    if os.path.exists(feature_cache_path):
        print(f"\n\nLoading cached features for {song_name} from {feature_cache_path}\n\n")
        try:
            if os.path.getsize(feature_cache_path) > 0:  # Check if file is not empty
                with open(feature_cache_path, 'rb') as f:
                    features = pickle.load(f)
                return features
            else:
                print(f"Warning: Cache file {feature_cache_path} is empty.")
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Warning: Could not load features from {feature_cache_path}. File may be corrupted or incomplete.")
            print(f"Error details: {e}")
            # Optionally, delete the corrupted cache file so it can be regenerated
            os.remove(feature_cache_path)
            print(f"Corrupted cache file {feature_cache_path} has been deleted.")
    
    # Compute features (perform only once)
    # The rest of the feature extraction code remains the same

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
        'song_name' : name,
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



@log_elapsed_time(lambda *args, **kwargs: f"STFT - {Path(args[0]['song_name']).name}")
def process_stft_and_save(features, json_path, save_path, plot_width=50, plot_height=20, jignore=False):
    """
    Processes the STFT of the given audio file and saves the spectrogram image.
    Updates the processing status in the JSON file.
    """
    D = features['D']
    sr = features['sr']
    audio_path = features['path']
    song_name = features['song_name']

    if os.path.exists(json_path) and not jignore:
        with open(json_path, 'r') as f:
            tracking_data = json.load(f)
    else:
        tracking_data = {}

    if tracking_data.get(song_name, {}).get("STFT_processed", False):
        print(f"STFT already processed for {song_name}. Skipping...")
        return
    
    #song_dir = os.path.join(os.path.dirname(save_path), song_name)
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(plot_width, plot_height))
    librosa.display.specshow(D, sr=sr, x_axis='s', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Log-frequency Power Spectrogram - {song_name}')
    plt.ylim(20, 20000)
    plt.text(0.01, 0.95, f'Log-frequency Power Spectrogram - {song_name}',
             fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)


    #output_file = os.path.join(save_path, f"STFT.png")
    output_file = os.path.join(save_path, f"STFT - {song_name}.png")

    plt.savefig(output_file)
    plt.close()

    tracking_data.setdefault(song_name, {})["STFT_processed"] = True
    if not jignore:
        with open(json_path, 'w') as f:
            json.dump(tracking_data, f, indent=4)
    
    print(f"STFT processing complete for {song_name}.\n Saved to {output_file}.")

@log_elapsed_time(lambda *args, **kwargs: f"Self-Similarity Matrix and Chromagram - {Path(args[0]['song_name']).name}")
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
    song_name = features['song_name']

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
    librosa.display.specshow(chroma, sr=sr, hop_length=hop_length, x_axis='time', y_axis='chroma', ax=ax2)
    plt.title(f'Chromagram for {song_name}')
    #plt.colorbar(chroma_img, ax=ax2)

    # Set the aspect ratio and align the Chromagram to the top
    ax2.set_aspect(aspect=2)  # Aspect ratio 2:1 to make it half the height of SSM
    ax2.set_anchor('N')  # Align Chromagram plot to the top

    plt.tight_layout()

    plot_path = os.path.join(save_path, f"SSM_Chromagram_{song_name}.png")
    #plot_path = os.path.join(save_path, f"SSM_Chromagram.png")
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

@log_elapsed_time(lambda *args, **kwargs: f"Mel Spectrogram - {Path(args[0]['song_name']).name}")
def process_mel_spectrogram_and_save(features, json_path, save_path, plot_width=12, plot_height=8, jignore=False):
    """
    Processes the Mel spectrogram of the given audio file, converts it to decibel scale, and saves the plot.
    Updates the processing status in the JSON file.
    """
    mel_to_db = features['mel_to_db']
    sr = features['sr']
    audio_path = features['path']
    song_name = features['song_name']

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

    #mel_spec_path = os.path.join(save_path, f"Mel_Spectrogram_{song_name}.png")
    mel_spec_path = os.path.join(save_path, f"Mel_Spectrogram.png")
    plt.savefig(mel_spec_path)
    plt.close()

    tracking_data.setdefault(song_name, {})["Mel_Spectrogram_processed"] = True
    if not jignore:
        with open(json_path, 'w') as f:
            json.dump(tracking_data, f, indent=4)

    print(f"Mel Spectrogram processing complete for {song_name}.\n Saved to {mel_spec_path}.")

@log_elapsed_time(lambda *args, **kwargs: f"Harmonic CQT and Percussive SFFT - {Path(args[0]['song_name']).name}")
def process_harmonic_cqt_and_percussive_sfft_and_save(features, json_path, save_path, plot_width=12, plot_height=12, jignore=False):
    """
    Processes the Harmonic CQT and Percussive SFFT of the given audio file and saves the plots.
    Updates the processing status in the JSON file.
    """
    y_harmonic = features['y_harmonic']
    y_percussive = features['y_percussive']
    sr = features['sr']
    audio_path = features['path']
    song_name = features['song_name']

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

    #plot_path = os.path.join(save_path, f"Harmonic_CQT_Percussive_SFFT_{song_name}.png")
    plot_path = os.path.join(save_path, f"Harmonic_CQT_Percussive_SFFT.png")
    plt.savefig(plot_path)
    plt.close()

    tracking_data.setdefault(song_name, {})["Harmonic_CQT_Percussive_SFFT_processed"] = True
    if not jignore:
        with open(json_path, 'w') as f:
            json.dump(tracking_data, f, indent=4)

    print(f"Harmonic CQT and Percussive SFFT processing complete for {song_name}.\n Saved to {plot_path}.")

@log_elapsed_time(lambda *args, **kwargs: f"Harmonic CQT and Harmonic Mel - {Path(args[0]['song_name']).name}")
def process_harmonic_cqt_and_harmonic_mel_and_save(features, json_path, save_path, plot_width=12, plot_height=12, jignore=False):
    """
    Processes the Harmonic CQT and Harmonic Mel spectrogram of the given audio file and saves the plots.
    Updates the processing status in the JSON file.
    """
    y_harmonic = features['y_harmonic']
    sr = features['sr']
    audio_path = features['path']
    song_name = features['song_name']

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
    #plot_path = os.path.join(save_path, f"Harmonic_CQT_Harmonic_Mel.png")
    plt.savefig(plot_path)
    plt.close()

    tracking_data.setdefault(song_name, {})["Harmonic_CQT_Harmonic_Mel_processed"] = True
    if not jignore:
        with open(json_path, 'w') as f:
            json.dump(tracking_data, f, indent=4)

    print(f"Harmonic CQT and Harmonic Mel processing complete for {song_name}.\n Saved to {plot_path}.")
