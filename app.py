from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from pydantic import BaseModel
import os, yaml, uvicorn

# Read the app.yaml file
with open('config/app.yaml', 'r') as file:
    config = yaml.safe_load(file)


os.environ["OPENAI_API_KEY"] = config.get('openai-api-key')
os.environ["LANGCHAIN_API_KEY"] = config.get('lanhchain-api-key')
# Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ['HUGGINGFACEHUB_API_TOKEN']=config.get('hugginngface_token')


def load_documents_and_split(file_name):
    """
    Loads a PDF document, splits it into chunks, and returns the split documents.

    Args:
        file_name (str): The path to the PDF file to be loaded.

    Returns:
        list: A list of split document chunks.
    """
    loader = PyPDFLoader(file_name)
    doc = loader.load()
    # Document Transformation - Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(doc)
    return split_documents

hf_embeddings=HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",      
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':False},
)


huggingface_embeddings=HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",      
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':False},
)
hf=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-v0.1",
    temperature=0.7, max_new_tokens=1024
)


# Modifying this to OLlama Embeddings for free access
def create_embeddings(split_documents):
    """
    Creates embeddings for the given split documents using FAISS and OpenAIEmbeddings.

    Args:
        split_documents (list): A list of documents that have been split into smaller parts.

    Returns:
        FAISS: A FAISS vector database containing the embeddings of the split documents.
    """
    vector_db = FAISS.from_documents(split_documents, hf_embeddings)
    return vector_db



def create_hf_embeddings(split_documents):
    """
    Creates a FAISS vector database from the given split documents using Hugging Face embeddings.

    Args:
        split_documents (list): A list of documents that have been split into smaller chunks.

    Returns:
        FAISS: A FAISS vector database containing the embeddings of the split documents.
    """
    vector_db = FAISS.from_documents(split_documents, huggingface_embeddings)
    return vector_db





# Load and split documents
split_documents = load_documents_and_split('data/ConceptsofBiology.pdf')
vector_db = create_embeddings(split_documents)
vector_db_hf = create_hf_embeddings(split_documents)

# Incorporating LLMs to the RAG flow - Locally running llama3.2 from Ollama
model_llama = OllamaLLM(model='llama3.2')
model_gpt = ChatOpenAI(model='gpt-3.5-turbo')
 

# Prompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the questions based only on the provided context. 
Think step by step before providing a detailed answer. 
<context> {context} </context>
Question: {input}""")

hf_prompt_template="""
Use the following piece of context to answer the questions asked.
Provide the answer based on the context {context}
Question:{question}
"""


# Create chains
doc_chain_llama = create_stuff_documents_chain(model_llama,prompt= prompt)
doc_chain_gpt = create_stuff_documents_chain(model_gpt,prompt= prompt)


retriever = vector_db.as_retriever()
retriever=vector_db_hf.as_retriever(search_type="similarity",search_kwargs={"k":3})



prompt=PromptTemplate(template=hf_prompt_template,input_variables=["context","question"])


retriever_chain_llama = create_retrieval_chain(retriever, doc_chain_llama)
retriever_chain_gpt = create_retrieval_chain(retriever, doc_chain_gpt)
retrievalQA=RetrievalQA.from_chain_type(
    llm=hf,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt":prompt}
)


app=FastAPI(
    title="Langchain RAG Server - Biology Concepts",
    version="1.0",
    decsription="API for a user interactions with a chat - Biology Concepts"

)

class QueryRequest(BaseModel):
    query: str

# Endpoint for Ollama Llama3.2
@app.post("/ollama-llama3.2")
async def ollama_llama(request: QueryRequest):
    response = retriever_chain_llama.invoke({"input" : request.query})
    return response['answer']

# Endpoint for OpenAI GPT3.5 Turbo
# @app.post("/gpt")
# async def gpt(request: QueryRequest):
#     response = retriever_chain_gpt.invoke({"input" : request.query})
#     return response['answer']


# Endpoint for Mistral-HF
@app.post("/mistral")
async def mistral(request: QueryRequest):
    response = retrievalQA.invoke({"query" : request.query})
    return response['result']

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)

