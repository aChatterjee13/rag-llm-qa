"""Define prompt templates for various RAG implementations"""
import yaml
from langchain_core.prompts import ChatPromptTemplate

# Read the app.yaml file
with open('config/prompts.yaml', encoding="utf-8") as file:
    config = yaml.safe_load(file)

def get_prompt_template_llama():
    """Get prompt template for llama"""
    prompt = config.get('llama_prompt')
    prompt_template = ChatPromptTemplate.from_template(prompt)
    return prompt_template

def get_promt_template_crag():
    """Get prompt template for CRAG pipeline"""
    crag_template = config.get('crag_prompt_template')
    return crag_template

def get_retrieval_grading_prompt():
    """Get retrieval grading template for CRAG pipeline"""
    retrieval_grading_prompt = config.get('crag_retrieval_grading')
    return retrieval_grading_prompt

def get_rewrite_temnplate():
    """Get rewrite template for CRAG pipeline"""
    rewrite_template = config.get('rewrite_template')
    return rewrite_template
