'''CRAG - Tools'''

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

from src.prompting import get_promt_template_crag, get_retrieval_grading_prompt
from src.prompting import  get_rewrite_temnplate
from models.llm import model_llama

def rag_chain():
    '''rag chain'''
    prompt = PromptTemplate(
        template=get_promt_template_crag(),
        input_variables=["generation", "question", "context"],
    )
    return prompt | model_llama | StrOutputParser()

def retrieval_grader():
    '''grading retrievals'''
    prompt = PromptTemplate(
        template=get_retrieval_grading_prompt(),
        input_variables=["question", "document"],
    )
    return prompt | model_llama | JsonOutputParser()

def web_search_tool():
    '''Web searching'''
    return TavilySearchResults(k=3)

def question_rewriter():
    '''Questions Rewriter'''
    re_write_prompt = PromptTemplate(
        template=get_rewrite_temnplate(),
        input_variables=["generation", "question"],
    )
    return re_write_prompt | model_llama | StrOutputParser() 
