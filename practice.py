from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyMuPDFLoader("gunaho ka devta.pdf")

data = loader.lazy_load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=20,
)

splitter_text = []

for doc in data:
    chunks = splitter.split_documents([doc])
    splitter_text.extend(chunks)

print(splitter_text)