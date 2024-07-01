import os
import json
from typing import Dict, Any
from pathlib import Path

# Define the base directory where all data will be stored
BASE_DIR = Path("data_collections")

# Define the structure of the metadata
MetaData = Dict[str, Any]
Data = Dict[str, Any]

# Define the structure of the text data to be stored in the JSON file
TextData = Dict[str, Any]
AudioMetaData = Dict[str, Any] 
ImageMetaData = Dict[str, Any]
VideoMetaData = Dict[str, Any]


def initialize_user_data(meta_data: MetaData) -> None:
    """
    Initializes the data collection folders for a new user based on the provided metadata.

    :param meta_data: A dictionary containing user metadata such as 'id'.
    """
    user_id = meta_data.get('id')
    if user_id is None:
        raise ValueError("Metadata must contain an 'id' key.")

    user_dir = BASE_DIR / str(user_id)
    # Create directories for each type of data
    for data_type in ['text-data', 'image-data', 'video-data', 'audio-data']:
        (user_dir / data_type).mkdir(parents=True, exist_ok=True)


def append_text_data(user_id: str, text_data: TextData) -> None:
    """
    Appends new text data to an existing JSON file within the text-data folder for a specific user.
    Creates a new JSON file if it does not exist.

    :param user_id: The unique identifier for the user.
    :param text_data: A dictionary containing the text data to append.
    """
    text_data_dir = BASE_DIR / user_id / 'text-data'
    text_data_file = text_data_dir / 'user_text_data.json'

    if not text_data_file.exists():
        with open(text_data_file, 'w') as file:
            json.dump([], file)  # Initialize with an empty list

    with open(text_data_file, 'r+') as file:
        existing_data = json.load(file)
        existing_data.append(text_data)
        file.seek(0)  # Go to the beginning of the file
        json.dump(existing_data, file, indent=4)



def append_audio_data(user_id: str, audio_data: AudioMetaData):
    """
    Appends new audio data metadata to an existing JSON file in the audio-data folder.

    Args:
        user_id: The unique ID of the user
        audio_data: Metadata for the new audio clip, containing keys like:
            - file_name: The name of the audio file 
            - duration: Length of audio in seconds
            - sample_rate: Sample rate of audio in Hz
            - transcript: Text transcript of audio (if applicable)
            - description: Any other descriptive info about the audio
    """
    audio_data_dir = BASE_DIR / user_id / 'audio-data'
    audio_data_file = audio_data_dir / 'user_audio_data.json'

    if not audio_data_file.exists():
        with open(audio_data_file, 'w') as f:
            json.dump([], f)
    
    with open(audio_data_file, 'r+') as f:
        existing_data = json.load(f)
        existing_data.append(audio_data)
        f.seek(0)
        json.dump(existing_data, f, indent=4)
        


def append_video_data(user_id: str, video_data: VideoMetaData):
    """
    Appends new video metadata to JSON file in video-data folder.

    Args:
        user_id: Unique ID of user
        video_data: Metadata for new video, containing keys like:
            - file_name: Name of video file
            - duration: Length of video in seconds 
            - frame_rate: Frames per second
            - resolution: Video resolution as widthxheight
            - description: Any other descriptive info
    """
    video_data_dir = BASE_DIR / user_id / 'video-data'
    video_data_file = video_data_dir / 'user_video_data.json'

    if not video_data_file.exists():
        with open(video_data_file, 'w') as f:
            json.dump([], f)
            
    with open(video_data_file, 'r+') as f:
        existing_data = json.load(f)
        existing_data.append(video_data)
        f.seek(0)
        json.dump(existing_data, f, indent=4)


def append_image_data(user_id: str, image_data: ImageMetaData):
    """
    Appends new image metadata to JSON file in image-data folder.

    Args: 
        user_id: Unique ID of user
        image_data: Metadata for new image, containing keys like:
            - file_name: Name of image file
            - size: Image size in pixels as widthxheight
            - description: Any other descriptive info 
    """
    image_data_dir = BASE_DIR / user_id / 'image-data'
    image_data_file = image_data_dir / 'user_image_data.json'

    if not image_data_file.exists():
        with open(image_data_file, 'w') as f:
            json.dump([], f)
            
    with open(image_data_file, 'r+') as f:
        existing_data = json.load(f)
        existing_data.append(image_data)
        f.seek(0)
        json.dump(existing_data, f, indent=4)
        



# Example usage
if __name__ == '__main__':
    # Initialize user data
    
    user_metadata = {
        'id': '12346',
    }
    initialize_user_data(user_metadata)

    Append a new text data entry
    new_text_data = {
        'query': 'What is the weather today?',
        'response': 'The weather is sunny with a high of 85 degrees.',
        'rewards': 10,
        'instructions': 'Please provide the weather forecast.',
        'prompts': 'Weather, Sunny, Forecast',
        'chain_of_thoughts': 'Checked weather API for current conditions.'
    }
    append_text_data(user_metadata['id'], new_text_data)
    new_audio = {
        'file_name': 'recording_1.wav',
        'duration': 10.5, 
        'sample_rate': 16000,
        'transcript': 'Hello this is a test recording',
        'description': 'First sample recording' 
    }

    append_audio_data(user_metadata['id'], new_audio)
    new_video = {
        'file_name': 'video_1.mp4',
        'duration': 60,
        'frame_rate': 30,
        'resolution': '1920x1080',
        'description': 'Sample video',
        "private": False
    }

    append_video_data(user_metadata['id'], new_video)
    new_image = {
        'file_name': 'image_1.jpg',
        'size': '1024x768',
        'description': 'Sample image'
    }
    append_image_data(user_metadata['id'], new_image)