from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type='standard_deviation',
    breakpoint_threshold_amount=1
)

sample = """
Farmers are the backbone of any nation, working tirelessly to produce the food that sustains millions. They face numerous challenges such as unpredictable weather, rising costs, and limited access to modern technology. Despite these difficulties, their dedication ensures food security and supports the economy, making their role both vital and admirable.Cricket is one of the most popular sports in the world, especially in countries like India. It brings people together, creating a sense of unity and excitement during matches. From local street games to international tournaments, cricket is not just a sport but an emotion that inspires teamwork, discipline, and passion among players and fans alike.

Terrorism is a serious global issue that threatens peace and stability across nations. It involves the use of violence and fear to achieve political or ideological goals, often targeting innocent civilians. Combating terrorism requires strong international cooperation, awareness, and efforts to address its root causes, promoting peace and understanding in society.
"""

docs = text_splitter.create_documents([sample])
print(len(docs))
print(docs)