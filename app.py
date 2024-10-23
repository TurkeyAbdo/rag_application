from langchain_community.document_loaders import DirectoryLoader

DATA_PATH = "data/books"

loader = DirectoryLoader(DATA_PATH,glob="*.md")
documents = loader.load()

print(documents)