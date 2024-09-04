from pyasn1_modules.rfc6031 import at_pskc_issuer

HF_TOKEN = ''
import bs4
import os
access_token = os.environ.get(HF_TOKEN)
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools import WikipediaQueryRun
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
# Vector embedding
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
# Agent
from langchain.agents import create_openapi_agent



#### Wikipedia data retrival tool ####
api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

#### Web-based data retrival tool (Custom tool)####
loader=WebBaseLoader('https://docs.smith.langchain.com/')
docs=loader.load()
documents=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
vectordb=FAISS.from_documents(documents, HuggingFaceBgeEmbeddings())
retriever=vectordb.as_retriever()
retriever_tool=create_retriever_tool(retriever, 'langsmith_search',
                                     "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!" )

#### Arxiv Tool ###
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

###### Combine all the tools ######
tools=[wiki, arxiv, retriever_tool]

###### Agents ######
from dotenv import load_dotenv
load_dotenv()
os.environ['OPEN_API_KEY']=os.getenv('OPEN_API_KEY')
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openapi_agent
from langchain.agents import AgentExecutor

llm=ChatOpenAI(model='gpt-3.5-turbo-0125', temperature=0)

prompt=hub.pull('hwchase17/openai-functions-agent')
agent=create_openapi_agent(llm, tools, prompt)

### Agent executer ###
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)
result=agent_executor.invoke({"input":"Tell me about Langsmith"})
print(result)
