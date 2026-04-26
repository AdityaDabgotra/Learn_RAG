from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_anthropic import ChatAnthropic
import os


# -------- CONFIG --------
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "gunaho-ka-devta"
EMBED_MODEL = "l3cube-pune/indic-sentence-bert-nli"

# Set your API key (or set ANTHROPIC_API_KEY env variable)
os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"


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

    loader = PyMuPDFLoader("gunaho ka devta.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
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

    # Claude Haiku — fast and cheap, great for RAG
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0.3,      # lower = more factual, less hallucination
        max_tokens=512,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant for the Hindi novel "Gunaho Ka Devta".

The context below is extracted from the novel (in Hindi).
The user may ask in Hinglish (Hindi written in Roman script).

Rules:
- Answer ONLY from the context provided.
- Reply in the same language/style as the question (Hindi or Hinglish).
- If the answer is not in the context, say "Mujhe is baare mein context mein koi information nahi mili."
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