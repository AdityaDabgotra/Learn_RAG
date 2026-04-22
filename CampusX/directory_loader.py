from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

loader = DirectoryLoader(
    path="../data/pdf",
    glob="*.pdf",     #"**/*" for all file irrespective of type
    loader_cls=PyMuPDFLoader
)

# pdf = loader.load()
pdf = loader.lazy_load()

print(type(pdf))
print(len(pdf))
print(pdf[57].page_content)

