from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma;
from langchain_chroma import Chroma
import os
import shutil
import ollama


markdown_path = "./data/books"
loader = DirectoryLoader(
    path=markdown_path,
    glob="*.md",
    show_progress=True,
    loader_cls=TextLoader,
    silent_errors=True
)
data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 500,
    add_start_index=True,
    length_function = len
)
docs = text_splitter.split_documents(data)
# print(f"split {len(data)} documents into {len(docs)} chunk ")
# print(docs[10].metadata)

CHROMA_PATH = "chroma"

# clear out database first
# if os.path.exists(CHROMA_PATH) :
#     shutil.rmtree(CHROMA_PATH)

# db = Chroma.from_documents(
#     docs,
#     OllamaEmbeddings(model="llama3.2"),
#     persist_directory=CHROMA_PATH
# )

# db.persist()
# print(f"Saved {len(docs)} chunks to {CHROMA_PATH}")