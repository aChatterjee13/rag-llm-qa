'''Gets the retrieval'''

from langchain_community.vectorstores import FAISS
from models.embeddings import hf1_embeddings

def get_retriever():
    '''....'''
    VECTORSTORE = FAISS.load_local(
        folder_path="data_index/indexed_data/", embeddings=hf1_embeddings,
        allow_dangerous_deserialization=True
    )
    return VECTORSTORE.as_retriever()
