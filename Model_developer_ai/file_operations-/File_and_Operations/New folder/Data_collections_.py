import os
import json
from typing import Dict, List, Union, Tuple, Any

# Define data types for the metadata and data
MetaData = Dict[str, Union[str, int]]
TextData = Dict[str, Union[str, int, List[str]]]
ImageData = Dict[str, Union[str, Tuple[int, int, int], Dict[str, str]]]
VideoData = Dict[str, Union[str, Tuple[int, int, int, int], Dict[str, str]]]
AudioData = Dict[str, Union[str, Tuple[int, int], Dict[str, str]]]

# Define the Data class to hold the data for each id
class Data:
    def __init__(self, id: str):
        self.id = id
        self.text_data = {}
        self.image_data = {}
        self.video_data = {}
        self.audio_data = {}

    def add_text_data(self, query: str, response: str, rewards: int,
                      instructions: List[str], prompts: List[str],
                      chain_of_thoughts: List[str]):
        """
        Add text data to the text_data dictionary for the current id.
        
        Args:
            query (str): The user query.
            response (str): The response to the query.
            rewards (int): The rewards for the response.
            instructions (List[str]): The instructions for generating the response.
            prompts (List[str]): The prompts used to generate the response.
            chain_of_thoughts (List[str]): The chain of thoughts used to generate the response.
        """
        text_data = {
            "query": query,
            "response": response,
            "rewards": rewards,
            "instructions": instructions,
            "prompts": prompts,
            "chain_of_thoughts": chain_of_thoughts,
        }
        self.text_data[len(self.text_data)] = text_data

    def add_image_data(self, image_path: str, size: Tuple[int, int, int],
                       description: Dict[str, str]):
        """
        Add image data to the image_data dictionary for the current id.
        
        Args:
            image_path (str): The path to the image file.
            size (Tuple[int, int, int]): The size of the image (height, width, channels).
            description (Dict[str, str]): A dictionary of additional metadata about the image.
        """
        image_data = {
            "image_path": image_path,
            "size": size,
            "description": description,
        }
        self.image_data[len(self.image_data)] = image_data

    def add_video_data(self, video_path: str, size: Tuple[int, int, int, int],
                       description: Dict[str, str]):
        """
        Add video data to the video_data dictionary for the current id.
        
        Args:
            video_path (str): The path to the video file.
            size (Tuple[int, int, int, int]): The size of the video (height, width, channels, frames).
            description (Dict[str, str]): A dictionary of additional metadata about the video.
        """
        video_data = {
            "video_path": video_path,
            "size": size,
            "description": description,
        }
        self.video_data[len(self.video_data)] = video_data

    def add_audio_data(self, audio_path: str, sample_rate: int,
                       description: Dict[str, str]):
        """
        Add audio data to the audio_data dictionary for the current id.
        
        Args:
            audio_path (str): The path to the audio file.
            sample_rate (int): The sample rate of the audio.
            description (Dict[str, str]): A dictionary of additional metadata about the audio.
        """
        audio_data = {
            "audio_path": audio_path,
            "sample_rate": sample_rate,
            "description": description,
        }
        self.audio_data[len(self.audio_data)] = audio_data

# Define the DataManager class to manage the data for multiple ids
class DataManager:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.data = {}

    def add_data(self, id: str, meta_data: MetaData):
        """
        Add a new Data object to the DataManager for the given id.
        
        Args:
            id (str): The id for the new Data object.
            meta_data (MetaData): The metadata for the new Data object.
        """
        self.data[id] = Data(id)
        self.data[id].meta_data = meta_data

    def add_text_data(self, id: str, query: str, response: str, rewards: int,
                      instructions: List[str], prompts: List[str],
                      chain_of_thoughts: List[str]):
        """
        Add text data to the Data object for the given id.
        
        Args:
            id (str): The id of the Data object to add text data to.
            query (str): The user query.
            response (str): The response to the query.
            rewards (int): The rewards for the response.
            instructions (List[str]): The instructions for generating the response.
            prompts (List[str]): The prompts used to generate the response.
            chain_of_thoughts (List[str]): The chain of thoughts used to generate the response.
        """
        self.data[id].add_text_data(query, response, rewards, instructions, prompts, chain_of_thoughts)

    def add_image_data(self, id: str, image_path: str, size: Tuple[int, int, int],
                       description: Dict[str, str]):
        """
        Add image data to the Data object for the given id.
        
        Args:
            id (str): The id of the Data object to add image data to.
            image_path (str): The path to the image file.
            size (Tuple[int, int, int]): The size of the image (height, width, channels).
            description (Dict[str, str]): A dictionary of additional metadata about the image.
        """
        self.data[id].add_image_data(image_path, size, description)

    def add_video_data(self, id: str, video_path: str, size: Tuple[int, int, int],
                       description: Dict[str, str]):
        
        """
        add video data to the Data object for the given id.
        
        Args:
            id (str): The id of the Data object to add video data to.
            video_path (str): The path to the video file.
            size (Tuple[int, int, int, int]): The size of the video (height, width, channels, frames).
            description (Dict[str, str]): A dictionary of additional metadata about the video.
        """
        
        self.data[id].add_video_data(video_path, size, description)

