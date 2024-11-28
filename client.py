"""Client module for testing the RAG backend"""
import requests
import streamlit as st

def get_ollama_response(input_text):
    """Response from Ollama local LLM"""
    response=requests.post("http://localhost:8000/ollama-llama3.2",
                           json={'query': input_text},timeout=300)
    print(response)
    return response.json()


def get_corrective_ollama_response(input_text):
    """Response from Ollama local LLM + Corrective RAG pipeline"""
    response=requests.post( "http://localhost:8000/corrective-llama3.2",
                           json={'query': input_text},timeout=300)
    return response.json()



st.title('Concepts of Biology - QA Chatbot')

uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt", "xlsx", "json","pdf"])

if uploaded_file is not None:
    st.write(f"Filename: {uploaded_file.name}")

    # Option to upload the file to FastAPI
    if st.button("Upload File to Index"):
        try:
            # Send the file to the FastAPI server
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post("http://localhost:8000/upload/", files=files, timeout=300)
            if response.status_code == 200:
                st.success(response.json().get("message"))
            else:
                st.error(f"Error: {response.status_code} - {response.json().get('error')}")
    
        except requests.ConnectionError as e:
            st.error(f"An error occurred: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")


user_input = st.text_input("Ask anything about Biology")
if st.button("Ollama Llama3.2 LLM"):
    result = get_ollama_response(user_input)
    st.write("Response from Llama3.2:", result)

if st.button("Corrective Ollama Llama3.2 LLM"):
    result = get_corrective_ollama_response(user_input)
    st.write("Response from Corrective RAG Llama3.2:", result)
