from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(page_content="Langchain is easy to work with LLms"),
    Document(page_content="Langchain is used to build LLm based application"),
    Document(page_content="Chroma is used to store and search document embeddings"),
    Document(page_content="Embeddings are vector representation of text"),
    Document(page_content="MMR helps to get diverse results when doing similarity search"),
    Document(page_content="Langchain support Chroma, FAISS,Pinecone, and more"),
]

embeddings = HuggingFaceEmbeddings()

vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embeddings,
)

retriever = vector_store.as_retriever(
    search_type = 'mmr',
    search_kwargs={"k":3,"lambda_mult":1}
)

query = "What is langchain?"

results = retriever.invoke(query)

for i,doc in enumerate(results):
    print(f"Result {i+1}")
    print(doc.page_content)


