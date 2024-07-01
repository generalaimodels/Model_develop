from typing import Tuple
import argparse
import os
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import librosa
from librosa.feature import mfcc
import plotly.graph_objs as go
import plotly.io as pio

TARGET_SAMPLE_RATE = 16000
NUM_MFCC = 13
PLOTS_DIR = 'plots'

def ensure_dir(directory: str) -> None:
    """Ensure that a directory exists, and create it if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_and_process_audio(file_path: str, target_sample_rate: int = TARGET_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load audio file and process it to fit specifications.
    
    Args:
        file_path (str): Path to the audio file.
        target_sample_rate (int, optional): Target sample rate for the audio.
    
    Returns:
        Tuple[np.ndarray, int]: Tuple containing processed audio data and the sample rate.
    """
    # Load the FLAC audio file
    audio, sample_rate = sf.read(file_path, always_2d=True)
    
    # Ensure mono by taking only the first channel
    audio = audio[:, 0]
    
    # Resample if necessary
    if sample_rate != target_sample_rate:
        audio = resample_poly(audio, target_sample_rate, sample_rate)
        sample_rate = target_sample_rate

    return audio, sample_rate

def extract_mfcc(audio: np.ndarray, sample_rate: int, num_mfcc: int = NUM_MFCC) -> np.ndarray:
    """Extract MFCC features from audio data.
    
    Args:
        audio (np.ndarray): Audio data array.
        sample_rate (int): Sample rate of the audio data.
        num_mfcc (int, optional): Number of MFCCs to return.
    
    Returns:
        np.ndarray: MFCC feature array.
    """
    mfcc_features = mfcc(y=audio, sr=sample_rate, n_mfcc=num_mfcc)
    return mfcc_features
def standardize_mfcc(mfcc: np.ndarray, target_length: int) -> np.ndarray:
    """
    Standardize the length of MFCC feature matrices by padding or truncating.

    :param mfcc: MFCC feature matrix
    :param target_length: Desired number of frames
    :return: Standardized MFCC feature matrix
    """
    if mfcc.shape[1] > target_length:
        return mfcc[:, :target_length]
    else:
        padding = target_length - mfcc.shape[1]
        return np.pad(mfcc, ((0, 0), (0, padding)), mode='constant')

def plot_audio_waveform(audio: np.ndarray, sample_rate: int, output_path: str) -> None:
    """Plot and save the audio waveform using Plotly.
    
    Args:
        audio (np.ndarray): Audio data array.
        sample_rate (int): Sample rate of the audio data.
        output_path (str): Path to save the plot.
    """
    time_axis = np.arange(audio.size) / sample_rate
    fig = go.Figure(data=go.Scatter(x=time_axis, y=audio, mode='lines', name='Waveform'))
    fig.update_layout(title='Audio Waveform', xaxis_title='Time (s)', yaxis_title='Amplitude')
    pio.write_html(fig, output_path)

def plot_mfcc_features(mfcc_features: np.ndarray, output_path: str) -> None:
    """Plot and save the MFCC features using Plotly.
    
    Args:
        mfcc_features (np.ndarray): MFCC feature array.
        output_path (str): Path to save the plot.
    """
    fig = go.Figure(data=go.Heatmap(z=mfcc_features, colorscale='Viridis'))
    fig.update_layout(title='MFCC Features', xaxis_title='Frame', yaxis_title='Coefficient')
    pio.write_html(fig, output_path)

def main():

    try:
        # Ensure the plots directory exists
        ensure_dir(PLOTS_DIR)

        # Load and process the audio file
        audio, sample_rate = load_and_process_audio(r"C:\Users\heman\Desktop\Coding\output1\audio\test_add_background_noise.flac")
        # Extract MFCC features
        mfcc_features = extract_mfcc(audio, sample_rate)
                      
        mfcc_features=standardize_mfcc(mfcc_features,target_length=30)
        # Save audio waveform plot
        waveform_plot_path = os.path.join(PLOTS_DIR, 'audio_waveform.html')
        plot_audio_waveform(audio, sample_rate, waveform_plot_path)

        # Save MFCC feature plot
        mfcc_plot_path = os.path.join(PLOTS_DIR, 'mfcc_features.html')
        plot_mfcc_features(mfcc_features, mfcc_plot_path)

        print(f'Audio waveform plot saved to: {waveform_plot_path}')
        print(f'MFCC features plot saved to: {mfcc_plot_path}')

    except Exception as e:
        print(f'Error processing audio file: {e}')

if __name__ == '__main__':
    main()