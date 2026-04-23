from langchain_text_splitters import CharacterTextSplitter

text = """Artificial Intelligence (AI) is a transformative technology that has the potential to revolutionize various aspects of our lives. At its core, AI refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. This encompasses a wide range of capabilities, including problem-solving, understanding natural language, recognizing patterns, and making decisions.

AI is already being utilized in numerous fields, such as healthcare, finance, and transportation. In healthcare, AI algorithms can analyze medical data to assist in diagnosing diseases and recommending treatments. In finance, AI systems are employed for fraud detection and algorithmic trading, enhancing efficiency and security. The transportation sector is witnessing the rise of autonomous vehicles, which rely heavily on AI to navigate and make real-time decisions.

Despite its benefits, the rise of AI also raises ethical concerns, including job displacement and privacy issues. As AI continues to evolve, it is crucial to address these challenges and ensure that its development aligns with societal values. The future of AI holds immense promise, and with responsible innovation, it can lead to significant advancements that improve our quality of life and drive economic growth."""

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator = ''
)

result = splitter.split_text(text)
# result = splitter.split_documents(docs)  for splitting document types

print(result)
print(len(result))