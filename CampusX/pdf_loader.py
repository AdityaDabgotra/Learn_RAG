from langchain_community.document_loaders import PyMuPDFLoader

doc = PyMuPDFLoader("../data/pdf/Lecture 3 Operators in JAVA.pdf")

pdf = doc.load()

print(pdf)
print(type(pdf))