import requests
import streamlit as st

def get_huggingface_response(input_text):
    response=requests.post("http://localhost:8000/mistral",
    json={'query':input_text})
    return response.json()

def get_ollama_response(input_text):
    response=requests.post(
    "http://localhost:8000/ollama-llama3.2",
    json={'query': input_text})
    return response.json()

# def get_openai_gpt_response(input_text):
#     response=requests.post(
#     "http://localhost:8000/gpt",
#     json={'input':{'query':input_text}})
#     return response.json()


st.title('Concepts of Biology - QA Chatbot')

user_input = st.text_input("Ask anything about Biology")
if st.button("Hugging Face LLM"):
    result = get_huggingface_response(user_input)
    st.write("Response from Mistral:", result)

if st.button("Ollama Llama3.2 LLM"):
    result = get_ollama_response(user_input)
    st.write("Response from Llama3.2:", result)

# if st.button("OpenAI GPT LLM"):
#     result = get_openai_gpt_response(user_input)
#     st.write("Response from GPT3.5:", result)
