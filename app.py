"""Main python file"""
from typing import List
import os
import yaml
import uvicorn

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
# from langchain import hub
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.messages import HumanMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from src.prompting import get_prompt_template_llama
from src.data_model import QueryRequest
from src.retriever import get_retriever
from src.crag_nodes import retrieve, grade_documents, generate
from src.crag_nodes import transform_query, web_search, decide_to_generate
from models.llm import model_llama

# Read the app.yaml file
with open('config/app.yaml', encoding="utf-8") as file:
    config = yaml.safe_load(file)

os.environ["OPENAI_API_KEY"] = config.get('openai-api-key')
os.environ["LANGCHAIN_API_KEY"] = config.get('langchain-api-key')
# Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ['HUGGINGFACEHUB_API_TOKEN']=config.get('hugginngface_token')
os.environ['TAVILY_API_KEY'] = config.get('tavily_token')


def rag_response(query):
    ''' General RAG response'''
    # Create chains for normal RAG

    prompt = get_prompt_template_llama()
    doc_chain = create_stuff_documents_chain(model_llama,prompt= prompt)
    retriever = get_retriever()
    retriever_chain = create_retrieval_chain(retriever, doc_chain)
    response = retriever_chain.invoke({"input" : query})
    return response


def crag_response():
    '''CRAG response'''

    class GraphState(TypedDict):
        '''Graph State class'''
        question: str
        generation: str
        web_search: str
        documents: List[str]

    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search_node", web_search)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


app=FastAPI(
    title="Langchain RAG Server - Biology Concepts",
    version="1.0",
    decsription="API for a user interactions with a chat - Biology Concepts"

)
# Endpoint for Ollama Llama3.2
@app.post("/ollama-llama3.2")
async def ollama_llama(request: QueryRequest):
    """Endpoint definition for LLama Local LLM"""
    response = rag_response(request.query)
    return response['answer']

#Endpoint for Corrective RAG pipeline
@app.post("/corrective-llama3.2")
async def corrective_rag(request: QueryRequest):
    """Endpoint definition for LLama Local LLM - Corrective RAG"""
    input_dict = {"question": request.query}
    response = crag_response().invoke(input_dict)
    # print(response)
    return response['generation']

@app.post("/upload/")
async def upload_file_and_index(file: UploadFile = File(...)):
    '''....'''
    try:
        # Save the file or process it here
        content = await file.read()
        with open(f"data_index/{file.filename}", "wb") as f:
            f.write(content)
        # create_load_vectorstore(f"data_index/{file.filename}")
        return JSONResponse(content={"message": f"File '{file.filename}' uploaded successfully"},
                            status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)

