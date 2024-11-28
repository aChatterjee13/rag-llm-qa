'''CRAG nodes'''

from langchain.schema import Document
from src.crag_tools import retrieval_grader
from src.crag_tools import rag_chain
from src.crag_tools import web_search_tool
from src.crag_tools import question_rewriter
from src.retriever import get_retriever

def decide_to_generate(state):
    '''...'''
    web_search = state["web_search"]
    if web_search == "Yes":
        return "transform_query"
    else:
        return "generate"
    
def generate(state):
    '''....'''
    question = state["question"]
    documents = state["documents"]

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = rag_chain()
    generation = chain.stream({"context": format_docs(documents), "question": question})
    return {"context": documents, "question": question, "generation": generation}

def grade_documents(state):
    '''...'''
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"
    grader = retrieval_grader()
    for d in documents:
        score = grader.invoke({"question": question, "document": d.page_content})
        grade = score["score"]
        if grade == "yes":
            filtered_docs.append(d)
        else:
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def retrieve(state):
    '''....'''
    question = state["question"]
    documents = get_retriever().invoke(question)
    return {"documents": documents, "question": question}

def web_search(state):
    '''....'''
    question = state["question"]
    documents = state["documents"]
    web_results = str(web_search_tool().run(question))
    web_results = Document(page_content=web_results, metadata={"source": "Web Search"})
    documents.append(web_results)
    return {"documents": documents, "question": question}

def transform_query(state):
    '''....'''
    question = state["question"]
    documents = state["documents"]
    formatted_docs = "\n\n".join(doc.page_content for doc in documents)
    better_question = question_rewriter().invoke(
        {"question": question, "documents": formatted_docs}
    )
    return {"documents": documents, "question": better_question}
