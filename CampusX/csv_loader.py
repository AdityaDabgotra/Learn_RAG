from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("../data/csv/file.csv")

data = loader.load()

# print(data)
print(type(data))
print(len(data))
print(data[0].page_content)
print(data[0].metadata)
