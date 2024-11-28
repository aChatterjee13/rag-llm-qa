'''Embeddings'''
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

hf1_embeddings=HuggingFaceBgeEmbeddings(
    model_name=
    "sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device':'cpu'}, encode_kwargs={'normalize_embeddings':False}
)

hf2_embeddings=HuggingFaceBgeEmbeddings(
    model_name=
    "sentence-transformers/all-MiniLM-l6-v2", 
    model_kwargs={'device':'cpu'}, encode_kwargs={'normalize_embeddings':False})
