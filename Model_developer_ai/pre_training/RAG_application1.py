import sys 
from pathlib import Path
current_file = Path(__file__).resolve()
package_root = current_file.parents[4]
sys.path.append(str(package_root))
print(package_root)
import dataset_collection
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from tqdm import tqdm

from typing import List, Optional

EMBEDDING_MODEL_NAME: str = "thenlper/gte-small"
READER_MODEL_NAME: str = "HuggingFaceH4/zephyr-7b-beta"
DIR_PATH: str = "E:/LLMS/Fine-tuning/"
DIR_OUTPUT: str = "E:/LLMS/Fine-tuning/output"
CSV_FILE_PATH: str = "E:/LLMS/Fine-tuning/csv_file.csv"
FOLDER_TO_PROCESS: List[str] = [DIR_OUTPUT]
MARKDOWN_SEPARATORS: List[str] = [
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


def load_data() -> List[str]:
    """Load data from directory and write to CSV file"""
    ext_files =  dataset_collection.get_files_with_extensions(DIR_PATH)
    dataset_collection.write_to_csv(CSV_FILE_PATH, ext_files)
    dataset_collection.process_pdfs_from_csv(csv_path=CSV_FILE_PATH, output_folder=DIR_OUTPUT)
    dataset_collection.process_files_txtfile(DIR_PATH, DIR_OUTPUT)
    dataset_collection.reformat_txt_files(FOLDER_TO_PROCESS)
    dataset =  dataset_collection.loading_folder_using_datasets(folder_path=f"{DIR_OUTPUT}/reformatted")
    return dataset['train']

def create_knowledge_base(train_data: List[dict]) -> List[LangchainDocument]:
    """Create knowledge base from training data"""
    knowledge_base: List[LangchainDocument] = []
    for doc in tqdm(train_data, desc="KNOWLEDGE_BASE_COLLECTIONS"):
        knowledge_base.append(LangchainDocument(page_content=doc["text"]))
    return knowledge_base

def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
) -> List[LangchainDocument]:
    """Split documents into chunks of maximum size `chunk_size` tokens"""
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )
    docs_processed: List[LangchainDocument] = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])
    unique_texts: dict = {}
    docs_processed_unique: List[LangchainDocument] = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)
    return docs_processed_unique

def create_embedding_model() -> HuggingFaceEmbeddings:
    """Create Hugging Face embeddings model"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=False,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
    )

def create_faiss_index(docs: List[LangchainDocument], embedding_model: HuggingFaceEmbeddings) -> FAISS:
    """Create FAISS index from documents and embeddings model"""
    return FAISS.from_documents(docs, embedding_model, distance_strategy=DistanceStrategy.COSINE)

def load_reader_model() -> pipeline:
    """Load reader model and tokenizer"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
    return pipeline(model=model, tokenizer=tokenizer, task="text-generation", do_sample=True, temperature=0.2, repetition_penalty=1.1, return_full_text=False, max_new_tokens=500)

def create_rag_prompt_template() -> str:
    """Create RAG prompt template"""
    prompt_in_chat_format = [
        {"role": "system", "content": """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer."""},
        {"role": "user", "content": """Context:
{context}
---
Now here is the question you need to answer.

Question: {question}"""},
    ]
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
    return tokenizer.apply_chat_template(prompt_in_chat_format, tokenize=False, add_generation_prompt=True)

def query_knowledge_base(user_query: str, knowledge_vector_database: FAISS) -> List[LangchainDocument]:
    """Query knowledge base and retrieve top-k documents"""
    retrieved_docs = knowledge_vector_database.similarity_search(query=user_query, k=5)
    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])
    return context

def main():
    try:
        train_data = load_data()
        knowledge_base = create_knowledge_base(train_data)
        docs_processed = split_documents(chunk_size=512, knowledge_base=knowledge_base)
        embedding_model = create_embedding_model()
        faiss_index = create_faiss_index(docs_processed, embedding_model)
        reader_model = load_reader_model()
        rag_prompt_template = create_rag_prompt_template()
        user_query = ""
        context = query_knowledge_base(user_query, faiss_index)
        final_prompt = rag_prompt_template.format(question=user_query, context=context)
        answer = reader_model(final_prompt)[0]["generated_text"]
        print(answer)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


# import argparse
# import torch
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from langchain.docstore.document import Document as LangchainDocument
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores.faiss import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores.utils import DistanceStrategy
# from typing import List, Optional
# from tqdm import tqdm
# import data_load_cleaning

# # Constants
# EMBEDDING_MODEL_NAME = "thenlper/gte-small"
# READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
# MARKDOWN_SEPARATORS = [
#     "\n#{1,6} ",
#     "```\n",
#     "\n\\*\\*\\*+\n",
#     "\n---+\n",
#     "\n___+\n",
#     "\n\n",
#     "\n",
#     " ",
#     "",
# ]

# def process_data(dir_path: str, dir_output: str, csv_file_path: str, folder_to_process: List[str]) -> None:
#     ext_files = data_load_cleaning.get_files_with_extensions(dir_path)
#     data_load_cleaning.write_to_csv(csv_file_path, ext_files)
#     data_load_cleaning.process_pdfs_from_csv(csv_path=csv_file_path, output_folder=dir_output)
#     data_load_cleaning.process_files_txtfile(dir_path,  dir_output)
#     data_load_cleaning.reformat_txt_files(folder_to_process)
#     dataset = data_load_cleaning.loading_folder_using_datasets(folder_path=f'{dir_output}/reformatted')
#     return dataset['train']

# def split_documents(
#     chunk_size: int,
#     knowledge_base: List[LangchainDocument],
#     tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
#     ) -> List[LangchainDocument]:
#     """
#     Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
#     """
#     text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
#         AutoTokenizer.from_pretrained(tokenizer_name),
#         chunk_size=chunk_size,
#         chunk_overlap=int(chunk_size / 10),
#         add_start_index=True,
#         strip_whitespace=True,
#         separators=MARKDOWN_SEPARATORS,
#     )

#     docs_processed = []
#     for doc in knowledge_base:
#         docs_processed += text_splitter.split_documents([doc])

#     # Remove duplicates
#     unique_texts = {}
#     docs_processed_unique = []
#     for doc in docs_processed:
#         if doc.page_content not in unique_texts:
#             unique_texts[doc.page_content] = True
#             docs_processed_unique.append(doc)

#     return docs_processed_unique

# def main():
#     parser = argparse.ArgumentParser(description='Process some integers.')
#     parser.add_argument('--dir_path', type=str, required=True)
#     parser.add_argument('--dir_output', type=str, required=True)
#     parser.add_argument('--csv_file_path', type=str, required=True)
#     parser.add_argument('--folder_to_process', type=list, required=True)
#     args = parser.parse_args()

#     ds = process_data(args.dir_path, args.dir_output, args.csv_file_path, args.folder_to_process)

#     RAW_KNOWLEDGE_BASE = [
#         LangchainDocument(page_content=doc["text"])
#         for doc in tqdm(ds,desc="KNOWLEDGE_BASE_COLLECTIONS")
#     ]

#     docs_processed = split_documents(
#         512,  # We choose a chunk size adapted to our model
#         RAW_KNOWLEDGE_BASE,
#         tokenizer_name=EMBEDDING_MODEL_NAME,
#     )

#     embedding_model = HuggingFaceEmbeddings(
#         model_name=EMBEDDING_MODEL_NAME,
#         multi_process=False,
#         model_kwargs={"device": "cuda"},
#         encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
#     )
#     KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
#         docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
#     )

#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#         READER_MODEL_NAME, quantization_config=bnb_config
#     )
#     tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

#     READER_LLM = pipeline(
#         model=model,
#         tokenizer=tokenizer,
#         task="text-generation",
#         do_sample=True,
#         temperature=0.2,
#         repetition_penalty=1.1,
#         return_full_text=False,
#         max_new_tokens=500,
#     )
#     prompt_in_chat_format = [
#         {
#             "role": "system",
#             "content": """Using the information contained in the context,
#     give a comprehensive answer to the question.
#     Respond only to the question asked, response should be concise and relevant to the question.
#     Provide the number of the source document when relevant.
#     If the answer cannot be deduced from the context, do not give an answer.""",
#         },
#         {
#             "role": "user",
#             "content": """Context:
#     {context}
#     ---
#     Now here is the question you need to answer.

#     Question: {question}""",
#         },
#     ]
#     RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
#         prompt_in_chat_format, tokenize=False, add_generation_prompt=True
#     )

#     user_query=""
#     retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
#     retrieved_docs_text = [
#         doc.page_content for doc in retrieved_docs
#     ]  # we only need the text of the documents
#     context = "\nExtracted documents:\n"
#     context += "".join(
#         [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
#     )
#     final_prompt = RAG_PROMPT_TEMPLATE.format(
#         question=user_query, context=context
#     )

#     # Redact an answer
#     answer = READER_LLM(final_prompt)[0]["generated_text"]

# if __name__ == "__main__":
#     main()
