import torch

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from typing import List, Optional
from tqdm import tqdm
from langchain.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
import dataset_collection




import argparse
from typing import List

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process arguments for the script.")

    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="thenlper/gte-small",
        help="Name of the embedding model to use (default: 'thenlper/gte-small').",
    )
    parser.add_argument(
        "--reader_model_name",
        type=str,
        default="HuggingFaceH4/zephyr-7b-beta",
        help="Name of the reader model to use (default: 'HuggingFaceH4/zephyr-7b-beta').",
    )
    parser.add_argument(
        "--dir_path",
        type=str,
        default="E:/LLMS/Fine-tuning/",
        help="Path to the directory containing the input files (default: 'E:/LLMS/Fine-tuning/').",
    )
    parser.add_argument(
        "--dir_output",
        type=str,
        default="E:/LLMS/Fine-tuning/output",
        help="Path to the directory for storing the output files (default: 'E:/LLMS/Fine-tuning/output').",
    )
    parser.add_argument(
        "--csv_file_path",
        type=str,
        default="E:/LLMS/Fine-tuning/csv_file.csv",
        help="Path to the CSV file (default: 'E:/LLMS/Fine-tuning/csv_file.csv').",
    )
    parser.add_argument(
        "--folder_to_process",
        type=str,
        nargs="+",
        default=[],
        help="List of folders to process (default: [DIR_OUTPUT]).",
    )

    args = parser.parse_args()

    if not args.folder_to_process:
        args.folder_to_process = [args.dir_output]

    return args
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
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
READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
DIR_PATH="E:/LLMS/Fine-tuning/"
DIR_OUTPUT="E:/LLMS/Fine-tuning/output"
CSV_FILE_PATH="E:/LLMS/Fine-tuning/csv_file.csv "
FOLDER_TO_PROCESS=[DIR_OUTPUT]

ext_files = data_load_cleaning.get_files_with_extensions(DIR_PATH)
data_load_cleaning.write_to_csv(CSV_FILE_PATH, ext_files)
data_load_cleaning.process_pdfs_from_csv(csv_path=CSV_FILE_PATH, output_folder=DIR_OUTPUT)
data_load_cleaning.process_files_txtfile(DIR_PATH,  DIR_OUTPUT)
data_load_cleaning.reformat_txt_files(FOLDER_TO_PROCESS)
dataset=data_load_cleaning.loading_folder_using_datasets(folder_path=f'{DIR_OUTPUT}/reformatted')

ds=dataset['train']

RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc["text"])
    for doc in tqdm(ds,desc="KNOWLEDGE_BASE_COLLECTIONS")
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

docs_processed = split_documents(
    512,  # We choose a chunk size adapted to our model
    RAW_KNOWLEDGE_BASE,
    tokenizer_name=EMBEDDING_MODEL_NAME,
)

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=False,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
)
KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
    docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    READER_MODEL_NAME, quantization_config=bnb_config
)
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

user_query=""
retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
retrieved_docs_text = [
    doc.page_content for doc in retrieved_docs
]  # we only need the text of the documents
context = "\nExtracted documents:\n"
context += "".join(
    [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
)
final_prompt = RAG_PROMPT_TEMPLATE.format(
    question=user_query, context=context
)

# Redact an answer
answer = READER_LLM(final_prompt)[0]["generated_text"]


    
import torch
from typing import List, Optional
from tqdm import tqdm

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

import data_load_cleaning


EMBEDDING_MODEL_NAME = "thenlper/gte-small"
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
READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
DIR_PATH = "E:/LLMS/Fine-tuning/"
DIR_OUTPUT = "E:/LLMS/Fine-tuning/output"
CSV_FILE_PATH = "E:/LLMS/Fine-tuning/csv_file.csv"
FOLDER_TO_PROCESS = [DIR_OUTPUT]


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


def main():
    try:
        ext_files = data_load_cleaning.get_files_with_extensions(DIR_PATH)
        data_load_cleaning.write_to_csv(CSV_FILE_PATH, ext_files)
        data_load_cleaning.process_pdfs_from_csv(csv_path=CSV_FILE_PATH, output_folder=DIR_OUTPUT)
        data_load_cleaning.process_files_txtfile(DIR_PATH, DIR_OUTPUT)
        data_load_cleaning.reformat_txt_files(FOLDER_TO_PROCESS)
        dataset = data_load_cleaning.loading_folder_using_datasets(folder_path=f'{DIR_OUTPUT}/reformatted')

        ds = dataset['train']

        RAW_KNOWLEDGE_BASE = [
            LangchainDocument(page_content=doc["text"])
            for doc in tqdm(ds, desc="KNOWLEDGE_BASE_COLLECTIONS")
        ]

        docs_processed = split_documents(
            512,  # We choose a chunk size adapted to our model
            RAW_KNOWLEDGE_BASE,
            tokenizer_name=EMBEDDING_MODEL_NAME,
        )

        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=False,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
        )
        KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
            docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            READER_MODEL_NAME, quantization_config=bnb_config
        )
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

        user_query = ""
        retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
        retrieved_docs_text = [
            doc.page_content for doc in retrieved_docs
        ]  # we only need the text of the documents
        context = "\nExtracted documents:\n"
        context += "".join(
            [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
        )
        final_prompt = RAG_PROMPT_TEMPLATE.format(
            question=user_query, context=context
        )

        # Redact an answer
        answer = READER_LLM(final_prompt)[0]["generated_text"]

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()