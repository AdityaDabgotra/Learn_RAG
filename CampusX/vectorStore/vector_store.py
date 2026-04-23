
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document

doc1 = Document(
    page_content = "RCB is Lollipop",
    metadata={"team":"Royal Challengers Bangalore"}
)
doc2 = Document(
    page_content = "MI is ambani ka paisa",
    metadata={"team":"Mumbai Indians"}
)
doc3 = Document(
    page_content = "Chennai Chuttar Kings are test players",
    metadata={"team":"Chennai Super Kings"}
)
doc4 = Document(
    page_content = "Punjab is playing Good",
    metadata={"team":"Punjab Kings"}
)

docs = [doc1,doc2,doc3,doc4]

vector_store = Chroma(
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    persist_directory = "chroma_db",
    collection_name = "IPL"
)

# add documents
vector_store.add_documents(docs)

#view documents
vector_store.get(include=['embeddings','documents','metadata'])

#search documents
print(vector_store.similarity_search(
    query = "who is lollipop",
    k = 1
))

#search with similarity score
print(vector_store.similarity_search_with_score(
    query = "who is lollipop",
    k = 1
))

#update documents
updated_doc1 = Document(
    page_content = "RCB 49 ,RCB trophy 1 CSK ban 2",
    metadata = {"team":"Royal Challengers Bangalore"}
)
vector_store.update_document(document_id='6e6ba6f0-263a-464e-baff-c0735332fa1d',document=updated_doc1)

print(vector_store.similarity_search(
    query = "trophy",
    k = 1
))