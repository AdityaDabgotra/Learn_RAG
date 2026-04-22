from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(
    template="Write a summary for the following paragraph - {paragraph}",
    input_variables=['paragraph']
)

parser = StrOutputParser()

loader = TextLoader("../data/text_files/text1.txt",encoding="utf-8")

docs = loader.load()
print(docs)
print(type(docs))
print(len(docs))

print(docs[0])
print(docs[0].page_content)


chain = prompt | model | parser

print(chain.invoke({'paragraph':docs[0].page_content}))