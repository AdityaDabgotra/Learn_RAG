from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

def process():
    print("Pdf loading started")
    loader = PyMuPDFLoader("gunaho ka devta.pdf")

    data = loader.lazy_load()
    print("Pdf loaded")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=20,
    )

    splitter_text = []

    for doc in data:
        chunks = splitter.split_documents([doc])
        splitter_text.extend(chunks)

    print("PDF splitted")
    vector_store = Chroma(
        embedding_function=HuggingFaceEmbeddings(model_name="l3cube-pune/indic-sentence-bert-nli"),
        persist_directory="Chroma DB",
        collection_name="Gunaho-ka-devta"
    )
    print("Embeddings Created")

    vector_store.add_documents(splitter_text)

    print("Added to Vector DB")

    return vector_store


def retrieval(question,vector_store):
    response = vector_store.similarity_search(
        query = question,
        k = 5
    )
    return response

if __name__ == "__main__":
    store = process()
    print("Document Scanned Successfully !!\n\n")
    while(True):
        ques = input("Enter your Question:\t")
        if ques.lower().strip() == 'exit':
            break
        response = retrieval(ques,store)
        for ans in response:
            print(ans.page_content)

