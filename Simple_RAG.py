# web based loader
HF_TOKEN = ''
import bs4
import os
access_token = os.environ.get(HF_TOKEN)
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Vector embedding
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma


# Load, chunk and index the content of the html page
loader=WebBaseLoader(web_paths=('https://lilianweng.github.io/posts/2023-06-23-agent/',),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                     class_=("post-title", "post-content","post-header"))), )

text_documents=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents =text_splitter.split_documents(text_documents)

db=Chroma.from_documents(documents[:5], HuggingFaceBgeEmbeddings())
query ="who are the authors of attention all you need papar?"
result=db.similarity_search(query)
print(result[0].page_content)





