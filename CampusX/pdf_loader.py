from langchain_community.document_loaders import PyMuPDFLoader

doc = PyMuPDFLoader("../data/pdf/Lecture 3 Operators in JAVA.pdf")

pdf = doc.load()

# print(pdf)
# print(type(pdf))
# print(len(pdf))

print(doc[0].page_content)
print(doc[0].metadata)
