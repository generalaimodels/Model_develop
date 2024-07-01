import os
import json
import torch
import warnings
import nltk
from tqdm.notebook import tqdm
from datasets import load_dataset, DatasetDict
from typing import List, Dict, Union, Optional, Any, Tuple
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM, 
                          AutoModelForMaskedLM,
                          AutoModelForSeq2SeqLM,
                          BitsAndBytesConfig,
                          pipeline
                          )
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

nltk.download('punkt')

def Advanced_Data_Loader(input: Union[str, Dict[str, str]], format: Optional[str] = None, split_ratios: Optional[Dict[str, float]] = None) -> Optional[DatasetDict]:
    """
    Loads a dataset from a given input path or dictionary specifying file paths and splits it.

    :param input: A string representing the dataset name or directory, or a dictionary containing file paths.
    :param format: The format of the dataset if loading from a file (e.g., 'csv' or 'json').
    :param split_ratios: A dictionary with keys 'train', 'test', and 'eval' containing split ratios.
    :return: A loaded and split dataset or None in case of failure.
    """
    if split_ratios is None:
        split_ratios = {'train': 0.8, 'test': 0.1, 'eval': 0.1}

    try:
        # Load the dataset
        if isinstance(input, dict) and format in ['csv', 'json']:
            dataset = load_dataset(format, data_files=input)
        elif isinstance(input, str) and format == 'text':
            dataset = load_dataset(format, data_dir=input)
        elif isinstance(input, str) and format is None:
            dataset = load_dataset(input)
        else:
            warnings.warn("Invalid input or format. Please provide a valid dataset name, directory, or file paths.")
            return None
    except FileNotFoundError as e:
        warnings.warn(str(e))
        return None

    # Split the dataset
    if dataset:
        split_dataset = dataset['train'].train_test_split(test_size=split_ratios['test'] + split_ratios['eval'])
        test_eval_dataset = split_dataset['test'].train_test_split(test_size=split_ratios['eval'] / (split_ratios['test'] + split_ratios['eval']))
        dataset = DatasetDict({
            'train': split_dataset['train'],
            'test': test_eval_dataset['train'],
            'eval': test_eval_dataset['test']
        })

    print("Splits: ", dataset.keys())
    print("Columns: ", {split: dataset[split].column_names for split in dataset.keys()})
    return dataset

def Serialize_Dataset_To_Json(dataset: Dict[str, Any], file_path: str) -> None:
    """
    Serializes the dataset with all its samples to a JSON file.

    Args:
        dataset (Dict[str, Any]): The dataset dictionary to serialize.
        file_path (str): The path to the output JSON file.

    Returns:
        None
    """
    # Convert the dataset to a serializable format by extracting the rows
    dataset_to_serialize = {}
    for split in dataset.keys():
        dataset_to_serialize[split] = dataset[split].to_dict()

    # Serialize the dataset to JSON
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(dataset_to_serialize, json_file, indent=4, ensure_ascii=False)
def Serialize_Dataset_To_Json_Allcolumns(dataset: Dict[str, Any],file_path:str):

  dataset_to_serialize={}
  for column in dataset.column_names:
    dataset_to_serialize[column]=dataset[column]
  with open(file_path,'w', encoding='utf-8') as josn_file:

    json.dump(dataset_to_serialize,josn_file,indent=4,ensure_ascii=False)
    

EMBEDDING_MODEL_NAME = "thenlper/gte-small"
READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]
FOLDER_PATH=""
DATASET_NAME=''
TEXT_COLUMN=''
LABEL_COLUMN=''
USER_QUERY=''
NO_OF_TOP_K_RESULTS=10

#================================== * FROM THE LOCAL FOLDER * =======================================================================================
Folder_Documents=DirectoryLoader(FOLDER_PATH,glob="**/[!.]*",show_progress=True,use_multithreading=True,silent_errors=True)
DOCUMENTS_DATA_BASE=Folder_Documents.load()
#==================================  * HUGGING_FACE * ================================================================================================
dataset=Advanced_Data_Loader(DATASET_NAME)
RAW_KNOWLEDGE_BASE=[
    LangchainDocument(page_content=doc[TEXT_COLUMN],metadat={"source":doc[LABEL_COLUMN]})
    for doc in tqdm(dataset)
]

def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique

#======================================== * DATASET BASE * ====================================

RAW_KNOWLEDGE_COMPLETE=[RAW_KNOWLEDGE_BASE,DOCUMENTS_DATA_BASE]

for DATA_BASE in RAW_KNOWLEDGE_COMPLETE:
    docs_processed = split_documents(
    512,  # We choose a chunk size adapted to our model
    DATA_BASE,
    tokenizer_name=EMBEDDING_MODEL_NAME,
)
    

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
)

KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
    docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
)

#======================================== * SEARCHING FORM KNOWLEDGE BASE * =====================================
retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=USER_QUERY, k=NO_OF_TOP_K_RESULTS)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

READER_LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=500,
)

prompt_in_chat_format = [
    {
        "role": "system",
        "content": """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.""",
    },
    {
        "role": "user",
        "content": """Context:
{context}
---
Now here is the question you need to answer.

Question: {question}""",
    },
]
RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
    prompt_in_chat_format, tokenize=False, add_generation_prompt=True
)

retrieved_docs_text = [
    doc.page_content for doc in retrieved_docs
]  # we only need the text of the documents
context = "\nExtracted documents:\n"
context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])

final_prompt = RAG_PROMPT_TEMPLATE.format(
    question=USER_QUERY, context=context
)

answer = READER_LLM(final_prompt)[0]["generated_text"]
#======================================== * RESULT * =====================================
print(answer)
