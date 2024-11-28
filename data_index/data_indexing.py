"""Data indexing module on FAISS"""

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from models.embeddings import hf1_embeddings, hf2_embeddings

# VECTORSTORE = None

def load_documents_and_split(file_name):
    """Document loader using recursive char splitter"""
    loader = PyPDFLoader(file_name)
    doc = loader.load()
    # Document Transformation - Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(doc)
    return split_documents

def create_hf1embeddings(split_documents):
    """Embeddings using all-mpnet-base-v2"""
    vector_db = FAISS.from_documents(split_documents, hf1_embeddings)
    vector_db.save_local("data_index/indexed_data")
    return vector_db

def create_hf2embeddings(split_documents):
    """Embeddings using all-MiniLM-l6-v2"""
    vector_db = FAISS.from_documents(split_documents, hf2_embeddings)
    return vector_db

def create_load_vectorstore(file_name):
    '''....'''
    return create_hf1embeddings(load_documents_and_split(file_name))
