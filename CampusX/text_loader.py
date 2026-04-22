from langchain_community.document_loaders import TextLoader

loader = TextLoader("../data/text_files/text1.txt",encoding="utf-8")

docs = loader.load()
print(docs)