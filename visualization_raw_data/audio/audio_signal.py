import logging
import os
from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from scipy import fft, signal
from scipy.io import wavfile

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_audio(file_path: str) -> Tuple[int, np.ndarray]:
    """Load an audio file and return the sample rate and audio data."""
    try:
        sample_rate, audio_data = wavfile.read(file_path)
        logger.info(f"Audio file loaded successfully: {file_path}")
        return sample_rate, audio_data
    except FileNotFoundError:
        logger.error(f"Audio file not found: {file_path}")
        raise
    except ValueError:
        logger.error(f"Invalid or corrupted audio file: {file_path}")
        raise

def process_audio(audio_data: np.ndarray, sample_rate: int, max_duration: float = 300.0) -> np.ndarray:
    """Process the audio data, ensuring it doesn't exceed the maximum duration."""
    max_samples = int(max_duration * sample_rate)
    if len(audio_data) > max_samples:
        logger.warning(f"Audio duration exceeds {max_duration} seconds. Truncating.")
        return audio_data[:max_samples]
    return audio_data

def compute_transform(audio_data: np.ndarray, transform_func, *args, **kwargs) -> np.ndarray:
    """Compute a transform on the audio data."""
    return transform_func(audio_data, *args, **kwargs)

def plot_transform(x: np.ndarray, y: np.ndarray, title: str, x_label: str, y_label: str, output_path: str) -> None:
    """Plot a transform and save it as an HTML file."""
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines'))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    fig.write_html(output_path)
    logger.info(f"{title} plot saved: {output_path}")

def plot_2d_transform(x: np.ndarray, y: np.ndarray, z: np.ndarray, title: str, x_label: str, y_label: str, output_path: str) -> None:
    """Plot a 2D transform and save it as an HTML file."""
    fig = go.Figure(data=go.Heatmap(x=x, y=y, z=z, colorscale='Viridis'))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    fig.write_html(output_path)
    logger.info(f"{title} plot saved: {output_path}")

def main(audio_file: str, output_folder: str) -> None:
    """Main function to process the audio file and generate plots."""
    try:
        os.makedirs(output_folder, exist_ok=True)

        sample_rate, audio_data = load_audio(audio_file)
        audio_data = process_audio(audio_data, sample_rate)

        time = np.arange(len(audio_data)) / sample_rate

        # FFT
        fft_result = compute_transform(audio_data, fft.fft)
        frequencies = fft.fftfreq(len(audio_data), d=1/sample_rate)
        plot_transform(frequencies[:len(frequencies)//2], np.abs(fft_result[:len(fft_result)//2]), 
                       'FFT Magnitude Spectrum', 'Frequency (Hz)', 'Magnitude', 
                       os.path.join(output_folder, 'fft.html'))

        # IFFT
        ifft_result = compute_transform(fft_result, fft.ifft)
        plot_transform(time, np.real(ifft_result), 'Inverse FFT', 'Time (s)', 'Amplitude', 
                       os.path.join(output_folder, 'ifft.html'))

        # RFFT
        rfft_result = compute_transform(audio_data, fft.rfft)
        rfft_freqs = fft.rfftfreq(len(audio_data), d=1/sample_rate)
        plot_transform(rfft_freqs, np.abs(rfft_result), 'Real FFT Magnitude Spectrum', 
                       'Frequency (Hz)', 'Magnitude', os.path.join(output_folder, 'rfft.html'))

        # IRFFT
        irfft_result = compute_transform(rfft_result, fft.irfft, n=len(audio_data))
        plot_transform(time, irfft_result, 'Inverse Real FFT', 'Time (s)', 'Amplitude', 
                       os.path.join(output_folder, 'irfft.html'))

        # DCT
        dct_result = compute_transform(audio_data, fft.dct)
        plot_transform(np.arange(len(dct_result)), dct_result, 'Discrete Cosine Transform', 
                       'Frequency Bin', 'Magnitude', os.path.join(output_folder, 'dct.html'))

        # IDCT
        idct_result = compute_transform(dct_result, fft.idct)
        plot_transform(time, idct_result, 'Inverse Discrete Cosine Transform', 
                       'Time (s)', 'Amplitude', os.path.join(output_folder, 'idct.html'))

        # DST
        dst_result = compute_transform(audio_data, fft.dst)
        plot_transform(np.arange(len(dst_result)), dst_result, 'Discrete Sine Transform', 
                       'Frequency Bin', 'Magnitude', os.path.join(output_folder, 'dst.html'))

        # IDST
        idst_result = compute_transform(dst_result, fft.idst)
        plot_transform(time, idst_result, 'Inverse Discrete Sine Transform', 
                       'Time (s)', 'Amplitude', os.path.join(output_folder, 'idst.html'))

        # Spectrogram
        f, t, Sxx = signal.spectrogram(audio_data, fs=sample_rate)
        plot_2d_transform(t, f, 10 * np.log10(Sxx), 'Spectrogram', 'Time (s)', 'Frequency (Hz)', 
                          os.path.join(output_folder, 'spectrogram.html'))

        logger.info("Audio processing and plot generation completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")

main(
    audio_file=r"C:\Users\heman\Desktop\Coding\AugLy\augly\tests\assets\audio\inputs\vad-go-mono-32000.wav",
    output_folder=r"C:\Users\heman\Desktop\Coding\output1\audio"
)