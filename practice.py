from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()


# -------- CONFIG --------
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "In_the_Silence_You_Left_Behind"
EMBED_MODEL = "l3cube-pune/indic-sentence-bert-nli"


# -------- LOAD / CREATE VECTOR DB --------
def get_vector_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if os.path.exists(PERSIST_DIR):
        print("Loading existing vector DB...")
        return Chroma(
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_NAME,
        )

    print("No DB found. Creating new one...")

    loader = PyMuPDFLoader("In the Silence You Left Behind.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separators=["\n\n", "\n", "।", ".", " ", ""],
    )
    split_docs = splitter.split_documents(docs)
    print(f"Split into {len(split_docs)} chunks")

    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )
    vector_store.add_documents(split_docs)
    print("Vector DB created and saved.")

    return vector_store


# -------- BUILD QA CHAIN --------
def build_qa_chain(vector_store: Chroma):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_output_tokens=512,
    )
    print(retriever)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant for the English novel "In the Silence You Left Behind".

The context below is extracted from the novel (in English).
The user may ask in English .

Rules:
- Answer ONLY from the context provided.
- Reply in the same language/style as the question.
- If the answer is not in the context, say "I dont have any Idea."
- Be concise.

Context:
{context}

Question:
{question}

Answer:""",
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# -------- MAIN --------
if __name__ == "__main__":
    store = get_vector_store()
    qa_chain = build_qa_chain(store)

    print("\n✅ Ready for questions! (type 'exit' to quit)\n")

    while True:
        question = input("Enter your question: ").strip()

        if not question:
            continue
        if question.lower() == "exit":
            break

        try:
            answer = qa_chain.invoke(question)
            print(f"\nAnswer:\n{answer.strip()}\n")
        except Exception as e:
            print(f"\n[Error] {e}\n")