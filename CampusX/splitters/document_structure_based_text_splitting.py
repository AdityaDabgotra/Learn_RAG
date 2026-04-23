from langchain_text_splitters import MarkdownTextSplitter

splitter = MarkdownTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0,
)

text = """
### learning RAG

functions of rag:
- `rag_retrieve()`: Retrieve relevant documents from a knowledge base based on a query.
- `rag_generate()`: Generate a response based on the retrieved documents and the query.
- `rag()`: A high-level function that combines retrieval and generation in one step.

### Example usage
"""
data = splitter.split_text(text)

print(data)