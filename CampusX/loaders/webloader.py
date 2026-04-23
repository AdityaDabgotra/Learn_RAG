from langchain_community.document_loaders import WebBaseLoader

url = "https://en.wikipedia.org/wiki/Ashoka"

loader = WebBaseLoader(url)
website = loader.load()

# print(website)
print(len(website))
print(website[0].page_content)