import io
import librosa
import torchaudio
import soundfile as sf
import numpy as np

from typing import Union, Tuple, List

def load_and_resample_audio(
    audio_path: str,
    target_sample_rate: int,
) -> Tuple[np.ndarray, int]:
    """Loads an audio file, resamples it to a target sample rate, and ensures it's mono channel.

    This function uses librosa, torchaudio, and soundfile in a try-except block 
    to ensure compatibility with various audio file formats.

    Args:
        audio_path (str): Path to the audio file.
        target_sample_rate (int): Desired sample rate after resampling.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing the resampled audio as a NumPy array 
                               and the original sample rate.

    Raises:
        RuntimeError: If the audio file cannot be loaded by any of the supported libraries.
    Example:
         # test cases
        audio_file = r"C:\\Users\\heman\\Desktop\\Coding\\output1\\audio\\input_1.flac"
        target_sr = 16000  # Example target sample rate

        audio, sr = load_and_resample_audio(audio_file, target_sr)

        print(f"Audio shape: {audio.shape}")
        print(f"Sample rate: {sr}")
    """
    try:
        audio, sr = librosa.load(audio_path, sr=target_sample_rate, mono=True)
        return audio, sr

    except Exception:
        pass
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            waveform = resampler(waveform)
        audio = waveform.numpy()
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        return audio, target_sample_rate

    except Exception:
        pass
    try:
        audio, sr = sf.read(audio_path)
        if sr != target_sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sample_rate)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono if stereo
        return audio, target_sample_rate

    except Exception:
        pass

    raise RuntimeError(f"Failed to load audio file: {audio_path}")


def extract_mfcc_features(
    audio_data: np.ndarray,
    sample_rate: int,
    num_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Extracts MFCC features from audio data, including delta and delta-delta features,
    along with energy.

    Args:
        audio_data: Audio data as a NumPy array.
        sample_rate: The sample rate of the audio data.
        num_mfcc: Number of MFCC coefficients to extract per frame.
        n_fft: Length of the FFT window.
        hop_length: Number of samples between successive frames.

    Returns:
        A 2D NumPy array of shape (num_frames, num_mfcc * 3 + 1),
        where each row represents a frame and contains the MFCC, delta, delta-delta,
        and energy features.
    """
    mfccs = librosa.feature.mfcc(
        y=audio_data, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    # Calculate energy
    energy = librosa.feature.rms(
        y=audio_data, frame_length=n_fft, hop_length=hop_length
    )
    # Stack features horizontally
    features = np.concatenate(
        (mfccs, delta_mfccs, delta2_mfccs, energy), axis=0
    )
    # Transpose to get (num_frames, num_features) shape
    # features = features.T

    return features

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
    

if __name__ == "__main__":
    audio_file = "C:/Users/heman/Desktop/Coding/output1/audio/input_1.flac"
    target_sr = 16000  
    audio, sr = load_and_resample_audio(audio_file, target_sr)
    features = extract_mfcc_features(
        audio, target_sr, num_mfcc=13, n_fft=2048, hop_length=512
    )
    feature_s=standardize_mfcc(features ,target_length=40)
    print(f"Audio shape: {audio.shape}")
    print(f"Sample rate: {sr}")
    print(f"{features.shape}")
    print(feature_s.shape)