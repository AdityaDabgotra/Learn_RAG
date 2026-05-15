from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

documents = [
    Document(page_content="langchain is good"),
    Document(page_content="Chroma vector store is a store"),
    Document(page_content="Embeddings convert text into vectors"),
    Document(page_content="Open AI is an excellent model"),
]

embeddings = HuggingFaceEmbeddings()

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="my_Collection"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 1})

query = "What is Chroma store"

results = retriever.invoke(query)

for i,doc in enumerate(results):
    print(f"------Result------")
    print(doc.page_content)

