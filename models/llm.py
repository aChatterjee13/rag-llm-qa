'''LLM Model'''

from langchain_ollama import OllamaLLM
# Incorporating LLMs to the RAG flow - Locally running llama3.2 from Ollama
model_llama = OllamaLLM(model='llama3.2')
