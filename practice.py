from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("gunaho ka devta.pdf")

data = loader.lazy_load()

for i in data:
    print(i)