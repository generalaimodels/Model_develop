# from langchain_community.document_loaders import TextLoader

# loader = TextLoader("codding_snipplet.md")
# loader.load()

# from langchain_community.document_loaders.csv_loader import CSVLoader


# loader = CSVLoader(file_path='C:/Users/heman/Desktop/Deep learning/file_paths.csv')
# data = loader.load()

# from langchain_community.document_loaders import DirectoryLoader
# from langchain.text_splitter import CharacterTextSplitter



# loader = DirectoryLoader('file_operations', glob= "**/[!.]*",show_progress=True, use_multithreading=True,loader_cls=TextLoader,silent_errors=True) #,txt ,.py, .csv ,.pdf,.md,.csv,.json



# docs = loader.load()

# from langchain.text_splitter import CharacterTextSplitter

# splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=30)

# chunked_docs = splitter.split_documents(docs)

# print(len(chunked_docs))
# print(len(docs))

# from langchain_community.document_loaders import UnstructuredHTMLLoader
# loader = UnstructuredHTMLLoader("https://python.langchain.com/docs/modules/data_connection/document_loaders/html")
# data = loader.load()