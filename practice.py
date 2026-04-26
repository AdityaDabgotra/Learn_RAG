from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyMuPDFLoader("gunaho ka devta.pdf")

data = loader.lazy_load()

splitter_text = []
for i in data:
    page = RecursiveCharacterTextSplitter(
        chunk_size = 350,
        chunk_overlap = 20,
    )
    
    splitter_text.append(page.split_documents(i))

print(splitter_text)

