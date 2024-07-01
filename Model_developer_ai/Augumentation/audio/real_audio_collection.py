import datetime
import platform
import subprocess
import logging
import numpy as np

from typing import Optional, Tuple, Union, Iterator, Dict, List
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.ndarray:
    """
    Helper function to read an audio file through ffmpeg.

    Args:
        bpayload (bytes): Audio data as bytes.
        sampling_rate (int): Desired sampling rate.

    Returns:
        np.ndarray: Audio data as a NumPy array.

    Raises:
        ValueError: If ffmpeg is not found or the audio file is invalid.
    """
    ffmpeg_command = [
        "ffmpeg",
        "-i", "pipe:0",
        "-ac", "1",
        "-ar", str(sampling_rate),
        "-f", "f32le",
        "-hide_banner",
        "-loglevel", "quiet",
        "pipe:1",
    ]

    try:
        with subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as ffmpeg_process:
            output_stream = ffmpeg_process.communicate(bpayload)
    except FileNotFoundError as error:
        logger.error("ffmpeg was not found but is required to load audio files from filename")
        raise ValueError("ffmpeg was not found but is required to load audio files from filename") from error

    audio = np.frombuffer(output_stream[0], np.float32)
    if audio.shape[0] == 0:
        logger.error("Invalid or malformed audio file")
        raise ValueError(
            "Soundfile is either not in the correct format or is malformed. Ensure that the soundfile has "
            "a valid audio file extension (e.g. wav, flac or mp3) and is not corrupted. If reading from a remote "
            "URL, ensure that the URL is the full address to **download** the audio file."
        )
    
    logger.info(f"Successfully read audio file with {audio.shape[0]} samples")
    return audio

def ffmpeg_microphone(
    sampling_rate: int,
    chunk_length_s: float,
    format_for_conversion: str = "f32le",
) -> Iterator[bytes]:
    """
    Helper function to read raw microphone data.

    Args:
        sampling_rate (int): Desired sampling rate.
        chunk_length_s (float): Length of each audio chunk in seconds.
        format_for_conversion (str, optional): Audio format for conversion. Defaults to "f32le".

    Yields:
        bytes: Raw audio data.

    Raises:
        ValueError: If the format_for_conversion is not supported.
    """
    size_of_sample = 4 if format_for_conversion == "f32le" else 2 if format_for_conversion == "s16le" else None
    if size_of_sample is None:
        logger.error(f"Unsupported format: {format_for_conversion}")
        raise ValueError(f"Unhandled format `{format_for_conversion}`. Please use `s16le` or `f32le`")

    system = platform.system()
    format_, input_ = {
        "Linux": ("alsa", "default"),
        "Darwin": ("avfoundation", ":0"),
        "Windows": ("dshow", _get_microphone_name()),
    }.get(system, (None, None))

    if format_ is None:
        logger.error(f"Unsupported operating system: {system}")
        raise ValueError(f"Unsupported operating system: {system}")

    ffmpeg_command = [
        "ffmpeg",
        "-f", format_,
        "-i", input_,
        "-ac", "1",
        "-ar", str(sampling_rate),
        "-f", format_for_conversion,
        "-fflags", "nobuffer",
        "-hide_banner",
        "-loglevel", "quiet",
        "pipe:1",
    ]

    chunk_len = int(round(sampling_rate * chunk_length_s)) * size_of_sample
    yield from _ffmpeg_stream(ffmpeg_command, chunk_len)

def ffmpeg_microphone_live(
    sampling_rate: int,
    chunk_length_s: float,
    stream_chunk_s: Optional[float] = None,
    stride_length_s: Optional[Union[Tuple[float, float], float]] = None,
    format_for_conversion: str = "f32le",
) -> Iterator[Dict[str, Union[int, np.ndarray, bool, Tuple[int, int]]]]:
    """
    Helper function to read audio from the microphone file through ffmpeg.

    Args:
        sampling_rate (int): The sampling rate to use when reading the data from the microphone.
        chunk_length_s (float): The length of the maximum chunk of audio to be returned.
        stream_chunk_s (float, optional): The length of the minimal temporary audio to be returned.
        stride_length_s (float or Tuple[float, float], optional): The length of the striding to be used.
        format_for_conversion (str, optional): The name of the format of the audio samples. Defaults to "f32le".

    Yields:
        Dict: A dictionary containing audio data and metadata.

    Raises:
        ValueError: If the format_for_conversion is not supported.
    Example:
            if __name__ == "__main__":
            # Example usage
            try:
                # Read audio from file
                with open(r"C:\Users\heman\Desktop\Coding\output1\audio\input_1.flac", "rb") as audio_file:
                    audio_data = ffmpeg_read(audio_file.read(), sampling_rate=16000)
                logger.info(f"Audio data shape: {audio_data.shape}")

                # Stream from microphone
                for chunk in ffmpeg_microphone_live(sampling_rate=16000, chunk_length_s=2.0, stream_chunk_s=10):
                    logger.info(f"Received audio chunk: {chunk['raw'].shape}")

            except Exception as e:
                logger.error(f"An error occurred: {str(e)}")
    """
    chunk_s = stream_chunk_s or chunk_length_s
    microphone = ffmpeg_microphone(sampling_rate, chunk_s, format_for_conversion=format_for_conversion)

    dtype = np.float32 if format_for_conversion == "f32le" else np.int16 if format_for_conversion == "s16le" else None
    if dtype is None:
        logger.error(f"Unsupported format: {format_for_conversion}")
        raise ValueError(f"Unhandled format `{format_for_conversion}`. Please use `s16le` or `f32le`")

    size_of_sample = 4 if format_for_conversion == "f32le" else 2
    chunk_len = int(round(sampling_rate * chunk_length_s)) * size_of_sample

    if stride_length_s is None:
        stride_length_s = chunk_length_s / 6
    if isinstance(stride_length_s, (int, float)):
        stride_length_s = [stride_length_s, stride_length_s]

    stride_left = int(round(sampling_rate * stride_length_s[0])) * size_of_sample
    stride_right = int(round(sampling_rate * stride_length_s[1])) * size_of_sample

    audio_time = datetime.datetime.now()
    delta = datetime.timedelta(seconds=chunk_s)

    for item in chunk_bytes_iter(microphone, chunk_len, stride=(stride_left, stride_right), stream=True):
        item["raw"] = np.frombuffer(item["raw"], dtype=dtype)
        item["stride"] = (item["stride"][0] // size_of_sample, item["stride"][1] // size_of_sample)
        item["sampling_rate"] = sampling_rate

        if datetime.datetime.now() > audio_time + 10 * delta:
            logger.warning("Audio processing is falling behind. Skipping chunk.")
            continue

        audio_time += delta
        yield item
def chunk_bytes_iter(
    iterator: Iterator[bytes],
    chunk_len: int,
    stride: Tuple[int, int],
    stream: bool = False
) -> Iterator[Dict[str, Union[bytes, Tuple[int, int], bool]]]:
    """
    Reads raw bytes from an iterator and creates chunks of specified length.

    Args:
        iterator (Iterator[bytes]): Iterator yielding raw audio bytes.
        chunk_len (int): Length of each chunk in bytes.
        stride (Tuple[int, int]): Left and right stride lengths in bytes.
        stream (bool): If True, yield partial results. Defaults to False.

    Yields:
        Dict: A dictionary containing raw audio data and metadata.

    Raises:
        ValueError: If stride is too large compared to chunk_len.
    """
    acc = b""
    stride_left, stride_right = stride
    if stride_left + stride_right >= chunk_len:
        logger.error(f"Invalid stride: ({stride_left}, {stride_right}) vs chunk_len {chunk_len}")
        raise ValueError(
            f"Stride needs to be strictly smaller than chunk_len: ({stride_left}, {stride_right}) vs {chunk_len}"
        )
    _stride_left = 0

    for raw in iterator:
        acc += raw
        if stream and len(acc) < chunk_len:
            stride = (_stride_left, 0)
            yield {"raw": acc[:chunk_len], "stride": stride, "partial": True}
        else:
            while len(acc) >= chunk_len:
                stride = (_stride_left, stride_right)
                item = {"raw": acc[:chunk_len], "stride": stride}
                if stream:
                    item["partial"] = False
                yield item
                _stride_left = stride_left
                acc = acc[chunk_len - stride_left - stride_right:]

    if len(acc) > stride_left:
        item = {"raw": acc, "stride": (_stride_left, 0)}
        if stream:
            item["partial"] = False
        yield item

def _ffmpeg_stream(ffmpeg_command: List[str], buflen: int) -> Iterator[bytes]:
    """
    Internal function to create the generator of data through ffmpeg.

    Args:
        ffmpeg_command (List[str]): FFmpeg command as a list of strings.
        buflen (int): Buffer length in bytes.

    Yields:
        bytes: Raw audio data.

    Raises:
        ValueError: If ffmpeg is not found.
    """
    bufsize = 2**24  # 16MB
    try:
        with subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, bufsize=bufsize) as ffmpeg_process:
            while True:
                raw = ffmpeg_process.stdout.read(buflen)
                if raw == b"":
                    break
                yield raw
    except FileNotFoundError as error:
        logger.error("ffmpeg was not found but is required to stream audio files from filename")
        raise ValueError("ffmpeg was not found but is required to stream audio files from filename") from error

def _get_microphone_name() -> str:
    """
    Retrieve the microphone name in Windows.

    Returns:
        str: Microphone name or "default" if not found.
    """
    command = ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", ""]

    try:
        ffmpeg_devices = subprocess.run(command, text=True, stderr=subprocess.PIPE, encoding="utf-8")
        microphone_lines = [line for line in ffmpeg_devices.stderr.splitlines() if "(audio)" in line]

        if microphone_lines:
            microphone_name = microphone_lines[0].split('"')[1]
            logger.info(f"Using microphone: {microphone_name}")
            return f"audio={microphone_name}"
    except FileNotFoundError:
        logger.warning("ffmpeg was not found. Please install it or make sure it is in your system PATH.")

    return "default"
